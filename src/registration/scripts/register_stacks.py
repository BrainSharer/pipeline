from __future__ import annotations

import argparse
from collections import defaultdict
import os
import glob
import shutil
import sys

import dask.array as da
from dask import delayed
from dask.diagnostics import ProgressBar

import numpy as np
import tifffile
from tqdm import tqdm
import zarr
import SimpleITK as sitk
import math
from itertools import product
from typing import Sequence, Tuple, Optional
from pathlib import Path
from cloudvolume import CloudVolume
from taskqueue.taskqueue import LocalTaskQueue
import igneous.task_creation as tc
import pandas as pd
from timeit import default_timer as timer



PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())


from library.utilities.utilities_registration import create_affine_parameters
from library.utilities.utilities_process import M_UM_SCALE
from library.image_manipulation.image_manager import ImageManager
from library.image_manipulation.neuroglancer_manager import NumpyToNeuroglancer
from library.image_manipulation.precomputed_manager import NgPrecomputedMaker
from registration.scripts.sitk_helpers import _channel_count, _resample_rgb_block, _resample_scalar_block, _spatial_chunks, _spatial_shape, compute_affine_padding, compute_chunk_source_region, compute_registration_metrics, create_tissue_mask, make_registration_image, normalize_tiff_array
from library.controller.sql_controller import SqlController


class StackRegistration:

    RGB_CHANNELS = 3


    def __init__(self, moving, fixed, downsample=1, registration_channel="luminance", debug=False):
        self.moving = moving
        self.fixed = fixed
        self.downsample = downsample
        self.scratch_dir = "/data/pipeline_tmp"
        self.base_path = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data"
        self.reg_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration"
        self.preview_path = os.path.join(self.reg_path, self.moving, f'{self.moving}_{self.fixed}_registered.{self.downsample}.nii')
        self.moving_tif_path = os.path.join(self.base_path, self.moving, 'preps', 'C1', f'source_aligned.{self.downsample}')
        self.fixed_tif_path = os.path.join(self.base_path, self.fixed, 'preps', 'C1', f'source_aligned.{self.downsample}')
        self.registered_zarr_path = os.path.join(self.scratch_dir, self.moving, f'{self.moving}_{self.fixed}_registered.{self.downsample}.zarr')
        self.moving_zarr_path = os.path.join(self.scratch_dir, self.moving, f'source_aligned.{self.downsample}.zarr')
        self.fixed_zarr_path = os.path.join(self.scratch_dir, self.fixed, f'source_aligned.{self.downsample}.zarr')
        self.registered_tif_path = os.path.join(self.scratch_dir, self.moving, f'registered.{self.downsample}')

        moving_brain_controller = SqlController(self.moving)
        self.moving_xy_resolution = moving_brain_controller.scan_run.resolution
        self.moving_z_resolution = moving_brain_controller.scan_run.zresolution

        fixed_brain_controller = SqlController(self.fixed)
        self.fixed_xy_resolution = fixed_brain_controller.scan_run.resolution
        self.fixed_z_resolution = fixed_brain_controller.scan_run.zresolution

        if self.moving == 'Allen':
            self.moving_spacing = [ round(self.moving_xy_resolution,2), round(self.moving_xy_resolution,2), self.moving_z_resolution ]
        else:
            self.moving_spacing = [ round(self.moving_xy_resolution*self.downsample,2), round(self.moving_xy_resolution*self.downsample,2), self.moving_z_resolution ]

        if self.fixed == 'Allen':     
            self.fixed_spacing = [ round(self.fixed_xy_resolution,2), round(self.fixed_xy_resolution,2), self.fixed_z_resolution ]
        else:
            self.fixed_spacing = [ round(self.fixed_xy_resolution*self.downsample,2), round(self.fixed_xy_resolution*self.downsample,2), self.fixed_z_resolution ]

        self.transform_path = os.path.join(self.reg_path, f"{self.moving}_{self.fixed}.tfm")
        if registration_channel not in {
            "luminance",
            "red",
            "green",
            "blue",
        }:
            raise ValueError("registration_channel must be one of: 'luminance', 'red', 'green', 'blue'")

        self.registration_channel = registration_channel
        self.image_manager = ImageManager(self.moving_tif_path)


        self.debug = debug


    def create_zarr(self):
        """
        Convert the TIFF stack into Zarr while preserving the channel
        dimension for RGB images.
        """

        if not os.path.exists(self.moving_tif_path):
            raise FileNotFoundError(
                f"Input path does not exist: "
                f"{self.moving_tif_path}"
            )

        divisors = {
            1: 32,
            4: 8,
            8: 4,
            16: 4,
            32: 2,
        }

        divisor = divisors.get(self.downsample,4,)
        chunk_x = max(1, self.image_manager.width // divisor)
        chunk_y = self.image_manager.height
        chunk_z = max(1, self.image_manager.len_files // divisor)

        dask_imgs = (StackRegistration.build_dask_array_from_folder(self.moving_tif_path))

        if dask_imgs.ndim == 3:
            # Grayscale
            rechunks = (chunk_z,chunk_y,chunk_x)

        elif dask_imgs.ndim == 4:
            # RGB/RGBA
            rechunks = (chunk_z, chunk_y, chunk_x,dask_imgs.shape[-1],)

        else:
            raise ValueError(f"Unsupported Dask volume shape: {dask_imgs.shape}")

        print(f"Input volume shape={dask_imgs.shape}")
        print(f"Using chunks={rechunks}")

        dask_imgs = dask_imgs.rechunk(rechunks)

        print(f"Dask shape={dask_imgs.shape}")
        print(f"Dask chunks={dask_imgs.chunksize}")

        os.makedirs(os.path.dirname(self.moving_zarr_path),exist_ok=True,)

        with ProgressBar():
            dask_imgs.to_zarr(self.moving_zarr_path,overwrite=True)

        source = zarr.open(self.moving_zarr_path,mode="r",)

        print(source.info)
        print(f"Created source Zarr with {_channel_count(source)} channel(s)")



    def create_transform(self):
        if self.downsample < 32:
            print(f'Downsample is too high: {self.downsample}')
            return
        if os.path.exists(self.transform_path):
            print(f"Transform file {self.transform_path} already exists, skipping transform creation")            
            return
        else:
            print(f'Creating affine registration in {self.transform_path}')

        if not os.path.exists(self.moving_tif_path):
            print(f'Exiting, missing: {self.moving_tif_path}')
            return
        if not os.path.exists(self.fixed_tif_path):
            print(f'Exiting, missing: {self.fixed_tif_path}')
            return
        if self.debug:
            print(f'Using moving data from {self.moving_tif_path}')
            print(f'Using fixed data from {self.fixed_tif_path}')
            return
        fixed_sitk = StackRegistration.create_sitk_volume(self.fixed_tif_path, self.registration_channel)
        moving_sitk = StackRegistration.create_sitk_volume(self.moving_tif_path, self.registration_channel)
        moving_sitk.SetSpacing(self.moving_spacing)
        fixed_sitk.SetSpacing(self.fixed_spacing)
        moving_points = self.get_points_from_db(self.moving)
        fixed_points = self.get_points_from_db(self.fixed)
        print(f'\nMoving sitk info size={moving_sitk.GetSize()} spacing={moving_sitk.GetSpacing()} dimension={moving_sitk.GetNumberOfComponentsPerPixel()}')
        print(f'Fixed sitk info size={fixed_sitk.GetSize()} spacing={fixed_sitk.GetSpacing()} channels={fixed_sitk.GetNumberOfComponentsPerPixel()}')
        affine_transform = StackRegistration.affine_registration(fixed_sitk, moving_sitk, fixed_points, moving_points)
        sitk.WriteTransform(affine_transform, self.transform_path)

    def get_points_from_db(self, brain):
        points = {}
        points['MD585'] = [
            [0.013953426233794197, 0.005983398409168919, 0.004514850459482039],
            [0.013912231044199516, 0.00587667296259302, 0.004994991750687443],
            [0.010316675919632297, 0.004565379106554093, 0.005053235069885642],
            [0.010229780198797628, 0.004591973307192101, 0.005283702359346643],
            [0.010645997385917916, 0.004575319550817625, 0.004934697554697554],
            [0.0105896270302566, 0.0046109504822593106, 0.005430269662921348],
            [0.011249664145238389, 0.005594378255743184, 0.003731213397316619],
            [0.011399400908132013, 0.005490763213847054, 0.006449849692970397],
            [0.011970730882949332, 0.005583081898770125, 0.00461111735769501],
            [0.01197703295006206, 0.005566141666052378, 0.005313953488372093],
            [0.012019832940339115, 0.006695419642352017, 0.0039904568049567695],
            [0.012250718389703262, 0.0064394844261743325, 0.006205794031031602],
            [0.011667528423428991, 0.005971317043107613, 0.003957394113845973],
        ]

        points['DK55'] = [
            [0.01445113, 0.00665242, 0.00582],
            [0.014330899999999999, 0.0059365600000000004, 0.00524],
            [0.0112075, 0.005005760000000001, 0.00462],
            [0.0111728, 0.00496067, 0.0049],
            [0.011525200000000001, 0.00504396, 0.00446],
            [0.011474, 0.00498683, 0.00494],
            [0.0121846, 0.00624508, 0.0034],
            [0.0121989, 0.00601109, 0.00632],
            [0.0127477, 0.00621566, 0.0045],
            [0.012691899999999999, 0.00617466, 0.00524],
            [0.0130807, 0.0073236, 0.00366],
            [0.0129273, 0.007183800000000001, 0.0063],
            [0.012650950000000001, 0.0061815, 0.00402],
        ]
        db_points = None

        xy_resolution = {}
        if brain == 'MD585':
            xy_resolution = 0.452 * self.downsample
        else:
            xy_resolution = 0.325 * self.downsample

        try:
            db_points = points[brain]
        except KeyError:
            return None

        data = []
        for (x,y,z) in db_points:
            # Perform operation on the 3 numbers
            x *= (M_UM_SCALE / xy_resolution)
            y *= (M_UM_SCALE / xy_resolution)
            z *= (M_UM_SCALE / self.fixed_z_resolution)
            data.append((x,y,z))

        return data

    def create_registered_tiles(self):
        start_time = timer()
        if os.path.exists(self.registered_zarr_path):
            print(f'Remove zarr output exists: {self.registered_zarr_path}')
            shutil.rmtree(self.registered_zarr_path)
            
        if not os.path.exists(self.transform_path):
            print(f"Transform file {self.transform_path} does not exist, cannot create registered volume")
            exit(0)
        transform = sitk.ReadTransform(self.transform_path)
        print('path', self.transform_path)
        print(f'matrix {transform}')
        if not os.path.exists(self.moving_zarr_path):
            print(f'Missing: {self.moving_zarr_path}')
            exit(0)
        source = zarr.open(self.moving_zarr_path, mode='r')
        print(source.info)
        paddings = {}
        paddings[32] = (32,32,32)
        paddings[16] = (32, 0, 256)
        paddings[8] = (32, 0, 512)
        paddings[4] = (256, 0, 256)
        paddings[1] = (32, 0, 1024)
        #chunks = 1,height,width/4
        #exp 1 divisor 4, 32,32,32 big gaps in the X
        #exp 2 divisor 4, 32,32,64 still gaps create registered tiles took 274.49 seconds
        #exp 3 divisor 4, 32,32,128 almost no gaps create registered tiles took 295.76 seconds
        #exp 4,divisor 8, 32,32,128, no good, the spinal cord gets lopped off
        #exp 5,divisor 4, 1,4 horrible
        #exp 6,chunks 16,height/4,width/4 padding=32,32,32 gaps in x,y,z create registered tiles took 35.48 seconds
        #exp 7,chunks 16,height/4,width/4 padding=(4, 38, 72) too many gaps everywhere, create registered tiles took 16.05 seconds
        #exp 8, chunks 64,64,64 padding=32,16,16, horrible, took 1m37.794s
        #exp 9, chunks 64,64,64 padding=16,32,32, horrible, took 1m44.388s
        #exp 10, chunks (1, 1234, 1164), padding 4,32,32, horrible
        #exp 11, chunks (1, 1234, 1164), padding 4,32,291
        #exp 12, chunks (1, 1234, 1164), padding 32,32,291, end lopped off
        #exp 13, chunks (1, 1234, 582), (32, 32, 145), end lopped off
        #exp 13, chunks (1, 1234, 582), (32, 32, 291), end lopped off
        #exp 14, chunks (1, 1234, 582), (64,64,291), end almost all there took 12m27.673s
        #exp 15, chunks (1, 1234, 582), (64,64,64),gaps in X but on lopping, took 9m20.104s
        #exp 16, chunks (1, 1234, 582), (64,4,291) no gaps very small part of spinal cord missing, took 12m53.123s
        #exp 17, chunks (1, 1234, 582), (32,4,291) no gaps, lots of spinal cord missing, took 6m57.830
        #exp 18, chunks (1, 1234, 582), (64,4,64) gaps no lopping 9m46.253s
        #exp 19, chunks (1, 1234, 582), (64,32,64) gaps no lopping 9m31.962s
        #exp 20, chunks (1, 1234, 582), (64,64,64) gaps small lopping
        #exp 21, chunks (57, 154, 291), (64,64,64) gaps, no lopping
        #exp 21, chunks (57, 154, 291), (57, 154, 291), works! 2m25.254s
        #exp 22, chunks (1, 523, 930), (32, 523, 930), works
        #exp 23, chunks (60, 523, 930),(30, 261, 465), works 1m45.298s DK50
        #exp 23, chunks (60, 1047, 930),(30, 523, 465), works 1m19.535s DK50
        #exp 24, chunks (57, 1234, 1164), (28, 617, 582), little in the midsection got lopped off, 1m43.639s, DK62
        #exp 25, chunks (230, 1234, 1164), (115, 617, 582), little in the midsection got lopped off,1m43.639s, DK62
        #exp 26, chunks (57, 1234, 291), (32,32,32), little chopped, 2m DK62
        #exp 27, chunks (57, 1234, 291), (32,64,64), big gaps in X 1m53.235s DK62
        #exp 28, chunks (57, 1234, 291), (32,32,64), big gaps and lopped off 1m53.235s DK62
        #exp 29, chunks (57, 1234, 291), (28, 617, 145), little in the midsection got lopped off,1m59.325s
        #exp 30, chunks (32,32,32), (32,32,32) useless
        #exp 31, chunks (len_files/divisor, height, width/divisor) (chunks/2), works well 3m50.423s DK62

        chunk_z = source.chunks[0]
        chunk_y = source.chunks[1]
        chunk_x = source.chunks[2]
        padding = (chunk_z//2, chunk_y//2, chunk_x//2)
        print(f'Using padding of {padding}')

        target = zarr.open(
            self.registered_zarr_path,
            mode="w",
            shape=source.shape,
            chunks=source.chunks,
            dtype=source.dtype)        

        StackRegistration.process_volume(
            source,
            target,
            transform,
            padding_zyx=padding,
            spacing_zyx=self.fixed_spacing[::-1],
            background_color=self.image_manager.get_bgcolor()
        )
        
        registered_volume = zarr.open(self.registered_zarr_path, mode='r')
        print(registered_volume.info)
        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        print(f"create registered tiles took {total_elapsed_time} seconds")


    def create_tifs(self):
        """
        Convert registered Zarr back to individual TIFF files.

        RGB remains RGB.
        Grayscale remains grayscale.
        """

        if os.path.exists(self.registered_tif_path):
            shutil.rmtree(self.registered_tif_path)

        os.makedirs(self.registered_tif_path, exist_ok=True)

        if not os.path.exists(self.registered_zarr_path):
            raise FileNotFoundError(self.registered_zarr_path)

        volume = zarr.open(self.registered_zarr_path,mode="r",)
        channels = (_channel_count(volume))
        nz = volume.shape[0]
        print(f"Writing {nz} TIFFs with {channels} channel(s)")

        for z in tqdm(range(nz), desc="Creating TIFFs"):
            image = np.asarray(volume[z])
            output_path = os.path.join(self.registered_tif_path,f"{z:03d}.tif",)
            tifffile.imwrite(output_path,image,)

        print(f"Finished writing TIFFs to {self.registered_tif_path}")


    def create_neuroglancer(self):
            
        neuroglancer_path = os.path.join(self.base_path, self.moving, 'www', 'neuroglancer_data') 
        rechunkme_path = os.path.join(neuroglancer_path, 'registered_rechunkme')
        progress_dir = os.path.join(neuroglancer_path, 'registered_progress')
        image_manager = ImageManager(self.registered_tif_path)
        lfiles = image_manager.len_files
        if not os.path.exists(self.registered_tif_path):
            print(f'Missing: {self.registered_tif_path}')
            exit(0)
        if lfiles < 10:
            print(f'No registered TIFs to work with: {lfiles}')
            exit(0)

        # chunk 
        chunks = [image_manager.height//4, image_manager.width//4, 1] # 1796x984

        scales = self.moving_spacing
        scales = tuple(int(s*1000) for s in scales) # convert from microns to nanometers for neuroglancer
        print(f'scales={scales} downsample={self.downsample}')
        num_channels = image_manager.num_channels
        # neuroglancer does not support boolean dtype
        dtype = image_manager.dtype
        print(f'volume_size={image_manager.volume_size} ndim={image_manager.ndim} dtype={dtype} num_channels={num_channels}')
        print(f'Creating initial pretransfer data with chunks={chunks}')
        ng = NumpyToNeuroglancer(
            self.moving,
            None,
            scales,
            "image",
            dtype,
            num_channels=num_channels,
            chunk_size=chunks,
        )
        if os.path.exists(progress_dir) and len(os.listdir(progress_dir)) > 0 and os.path.exists(self.registered_tif_path):
            print("Transfer task has already been completed")
        else:
            ng.init_precomputed(rechunkme_path, image_manager.volume_size)
            for i, f in enumerate(tqdm(image_manager.files, desc="Processing images")):
                filepath = os.path.join(self.registered_tif_path, f)
                ng.process_image(file_key=[i, filepath, progress_dir, False, 0, 0])

            ng.precomputed_vol.cache.flush()

        base_chunks = [64, 64, 16]

        scales, resolutions, chunks = NgPrecomputedMaker.compute_mipmaps((self.moving_spacing), base_chunks)
        mips = len(scales) - 1  # number of downsampled levels to create (excluding the original)
        registered_outpath = os.path.join(neuroglancer_path, f'registered.{self.downsample}')
        outpath = f"file://{registered_outpath}"
        if not os.path.exists(rechunkme_path):
            print(f"DIR {rechunkme_path} does not exist, exiting.")
            sys.exit()
        
        cloudpath = f"file://{rechunkme_path}"
        workers = 4
        tq = LocalTaskQueue(parallel=workers)

        print(f'Writing transfer task with {workers} workers')
        # I have been having trouble with newer versions of cloud volume and the sharded transfer tasks.  

        task = tc.create_transfer_tasks(cloudpath, dest_layer_path=outpath, max_mips=mips, chunk_size=chunks[0], mip=0, skip_downsamples=True)

        print(f'Creating transfer task with chunks={chunks[0]} to layer {outpath}')

        tq.insert(task)
        tq.execute()
        print('Finished transfer task')

        for mip in range(0, mips):
            factor = scales[mip]
            resolution = resolutions[mip]
            chunk_mip = chunks[mip]
            cv = CloudVolume(outpath, mip)
            print(f'Creating downsample task at mip={mip} factor={factor} with chunks={chunk_mip} resolution = {resolution}')

            task = tc.create_downsampling_tasks(cv.layer_cloudpath, mip=mip, num_mips=1, compress=True, factor=factor, chunk_size=chunk_mip)
            
            tq.insert(task)            
            tq.execute()
        print('Finished creating precomputed data')        

    def run_tiles(self):
        self.create_registered_tiles()
        self.create_tifs()
        self.create_neuroglancer()
        print("Finished running tiles, tifs and neuroglancer")

    def test_mips(self):
        scales = self.moving_spacing
        scales = tuple(int(s*1000) for s in scales) # convert from microns to nanometers for neuroglancer
        print(f'Before computing scales={scales} downsample={self.downsample}')
        base_chunks = [64,64,16]

        scales, resolutions, chunks = NgPrecomputedMaker.compute_mipmaps((x,y,z), base_chunks)
        for s,r,c in zip(scales, resolutions, chunks):
            print(f'scale={s}, resolution={r}, chunk={c}')



    @staticmethod
    def read_tiff_delayed(path: str):
        """
        Return a delayed object that reads a single tiff (using tifffile) and returns numpy array.
        """
        @delayed
        def _read(p):
            arr = tifffile.imread(p)
            # ensure shape is (Z=1?, Y, X, C) or (Y, X, C)
            return normalize_tiff_array(arr)
        return _read(path)

    @staticmethod
    def build_dask_array_from_folder(
        folder: str,
        pattern: str = "*.tif",
        sample_index: int = 0,
    ):
        """
        Build a Dask array from a directory containing one TIFF per section.

        Returns:

            grayscale:
                (Z,Y,X)

            RGB:
                (Z,Y,X,3)
        """

        files = sorted(
            glob.glob(
                os.path.join(folder, pattern)
            )
        )

        if not files:
            raise FileNotFoundError(
                f"No TIFF files found in {folder} "
                f"matching {pattern}"
            )

        sample = tifffile.imread(files[sample_index])
        sample = normalize_tiff_array(sample)

        if sample.ndim == 2:
            height, width = sample.shape
            channels = 1
        else:
            height, width, channels = sample.shape

        dtype = sample.dtype

        print(
            f"TIFF stack: {len(files)} sections, "
            f"shape={sample.shape}, dtype={dtype}, "
            f"channels={channels}"
        )

        delayed_reads = [
            StackRegistration.read_tiff_delayed(path)
            for path in files
        ]

        slices = []

        for delayed_image in delayed_reads:

            if channels == 1:
                shape = (height, width)
            else:
                shape = (
                    height,
                    width,
                    channels,
                )

            image = da.from_delayed(
                delayed_image,
                shape=shape,
                dtype=dtype,
            )

            slices.append(image)

        volume = da.stack(
            slices,
            axis=0,
        )

        return volume
    

    @staticmethod
    def build_dask_array_from_folderV0(folder: str, pattern: str = "*.tif", sample_index: int = 0):
        """
        Read a folder of single-plane TIFFs (450 files typical), create a Dask array
        with shape (Z, Y, X, C) or (Z, Y, X) depending on images.
        Returns (dask_array, metadata) where metadata includes dtype, shape, channels info.
        """
        files = sorted(glob.glob(os.path.join(folder, pattern)))
        if len(files) == 0:
            raise FileNotFoundError(f"No tiff files found in {folder} matching {pattern}")

        # read a sample to infer shape/dtype
        print(f"Reading files from {folder}")
        sample = tifffile.imread(files[sample_index])
        sample = np.asarray(sample)
        # normalize sample dims -> (Y, X) or (Y, X, C)
        if sample.ndim == 2:
            height, width = sample.shape
            channels = 1
        elif sample.ndim == 3:
            # Could be (Z, Y, X) if multi-page, but user said single-plane slides
            # assume (Y, X, C) if len==3 and last dim <= 4
            if sample.shape[0] <= 4 and len(files) == 1:
                # one file with channels first? fallback
                height, width, channels = sample.shape
            else:
                height, width = sample.shape[:2]
                channels = sample.shape[2] if sample.ndim == 3 else 1
        else:
            raise ValueError("Unexpected sample tiff dimensions: %s" % (sample.shape,))

        dtype = sample.dtype

        # create delayed readers for each file
        delayed_reads = [StackRegistration.read_tiff_delayed(p) for p in files]

        # wrap each into dask array chunk -> assume each file is a single slice (Y,X[,C])
        # We'll create array of shape (z, y, x, c) if channels>1 else (z, y, x)
        sample_arr = sample
        if channels == 1 and sample_arr.ndim == 3 and sample_arr.shape[2] == 1:
            sample_arr = sample_arr[..., 0]

        # create a single-chunk dask array per file and stack them
        da_slices = []
        for d in delayed_reads:
            # build from_delayed with shape of sample slice
            if channels == 1:
                shp = (height, width)
            else:
                shp = (height, width, channels)
            arr = da.from_delayed(d, shape=shp, dtype=dtype)
            da_slices.append(arr)

        return da.stack(da_slices, axis=0)  # shape (Z, Y, X) or (Z, Y, X, C)

    @staticmethod
    def write_zarr(volume, outfile):
        volume.to_zarr(
            outfile,
            overwrite=True,
            compressor=zarr.Blosc(
                cname="zstd",
                clevel=5,
                shuffle=2
            ),
        )

    @staticmethod
    def open_zarr(path):
        return da.from_zarr(path)


    @staticmethod
    def affine_registration(fixed, moving, fixed_landmarks=None, moving_landmarks=None):
        def command_iteration(method):
            print(f"Iteration: {method.GetOptimizerIteration()} ")
            print(f"Metric Value: {method.GetMetricValue()}")
            print("-" * 20)

        if fixed.GetNumberOfComponentsPerPixel() == 3:
            fixed = sitk.VectorIndexSelectionCast(fixed, 1)
        if moving.GetNumberOfComponentsPerPixel() == 3:
            moving = sitk.VectorIndexSelectionCast(moving, 1)
        fixed = sitk.Cast(fixed, sitk.sitkFloat32)
        moving = sitk.Cast(moving, sitk.sitkFloat32)
        fixed.SetOrigin((0.0, 0.0, 0.0))
        moving.SetOrigin((0.0, 0.0, 0.0))
        fixed.SetDirection(np.eye(3).flatten())
        moving.SetDirection(np.eye(3).flatten())        
        print(f"Affine registration with fixed image size {fixed.GetSize()} and moving image size {moving.GetSize()}")
        print(f"Fixed image spacing: {fixed.GetSpacing()}, Moving image spacing: {moving.GetSpacing()}")


        registration = sitk.ImageRegistrationMethod()
        #registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        # Correlation works much better than MattesMutual!!!
        registration.SetMetricAsCorrelation()
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(0.25)
        registration.SetInterpolator(sitk.sitkLinear)

        registration.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=300,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        # ------------------------------------------------------------
        # Initial transform
        # ------------------------------------------------------------

        if fixed_landmarks is not None and moving_landmarks is not None:
            print(f"Using {len(fixed_landmarks)} corresponding landmarks for affine initialization")
            initial_transform = (
                StackRegistration.landmark_affine_initialization(
                    fixed,
                    moving,
                    fixed_landmarks,
                    moving_landmarks)
            )
        elif fixed_landmarks is not None or moving_landmarks is not None:
            raise ValueError("Both fixed_landmarks and moving_landmarks must be supplied.")
        else:
            print("No landmarks supplied; using geometry initialization.")
            initial_transform = sitk.CenteredTransformInitializer(
                fixed,
                moving,
                sitk.AffineTransform(fixed.GetDimension()),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )        
        #registration.SetInitialTransform(initial_transform)
        #registration.SetOptimizerScalesFromPhysicalShift()

        registration.SetOptimizerScalesFromPhysicalShift()
        registration.SetShrinkFactorsPerLevel([4, 2, 1])
        registration.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration.SetInitialTransform(initial_transform, inPlace=True)
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        registration.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration))
        affine_transform = registration.Execute(fixed, moving)

        print("Affine done. Final metric:", registration.GetMetricValue())
        print("Optimizer's stopping condition: ", registration.GetOptimizerStopConditionDescription())

        return affine_transform



    @staticmethod
    def sitk_to_dask(image):
        arr = sitk.GetArrayFromImage(image)
        return da.from_array(
            arr,
            chunks=(32,512,512)
        )

    @staticmethod
    def create_sitk_volume(input_path: str, registration_channel: str = "luminance"):
        files = sorted(glob.glob(os.path.join(input_path, "*.tif")))
        if not files or len(files) == 0:
            print(f'No tifs in {input_path}')
            exit(0)
        slices = []
        for f in tqdm(files, desc="Creating sitk volume"):
            img = tifffile.imread(f)
            img = make_registration_image(img, registration_channel, )
            slices.append(img.astype(np.float32))
        arr = np.stack(slices, axis=0)
        return sitk.GetImageFromArray(arr)

    @staticmethod
    def save_tiffs(volume, directory):
        os.makedirs(directory, exist_ok=True)
        nz = volume.shape[0]
        slices = []

        for z in tqdm(range(nz), desc="Saving TIFs"):
            img = volume[z].compute()
            outpath = os.path.join(directory, f"{z:03d}.tif")
            tifffile.imwrite(outpath, img)
            slices.append(img)

        arr = np.stack(slices, axis=0)
        stack_path = os.path.join(directory, "stack.tif")
        tifffile.imwrite(stack_path, arr)
        print(f"Saved registered volume as individual tiffs in {directory} and as stack in {stack_path}")




    @staticmethod
    def process_volume(
        zarr_src,
        zarr_dst,
        transform,
        padding_zyx,
        spacing_zyx,
        background_color
    ):
        """
        Process an entire Zarr volume one chunk at a time.

        Parameters
        ----------
        zarr_src : zarr.Array
            Source volume.
        zarr_dst : zarr.Array
            Destination volume.
        transform : sitk.Transform
            Global transform.
        padding_zyx : tuple
            Padding added around each chunk.
        """

        shape = zarr_src.shape
        chunks = zarr_src.chunks

        nblocks = tuple(
            math.ceil(shape[i] / chunks[i])
            for i in range(3)
        )

        iterator = product(range(nblocks[0]),range(nblocks[1]),range(nblocks[2]))


        iterator = tqdm(
            iterator,
            total=nblocks[0] * nblocks[1] * nblocks[2],
            desc="Warping chunks", disable=False)

        for block_index in iterator:

            StackRegistration.process_block(
                zarr_src=zarr_src,
                zarr_dst=zarr_dst,
                block_index=block_index,
                transform=transform,
                padding_zyx=padding_zyx,
                spacing_zyx=spacing_zyx,
                background_color=background_color
            )

    @staticmethod
    def _zyx_to_xyz(values: Sequence[float]) -> Tuple[float, float, float]:
        """Convert (z, y, x) ordering to SimpleITK's (x, y, z) ordering."""
        if len(values) != 3:
            raise ValueError(f"Expected 3 values, got {len(values)}")
        z, y, x = values
        return float(x), float(y), float(z)

    @staticmethod
    def _compute_block_bounds(
        block_index: Sequence[int],
        shape_zyx: Sequence[int],
        chunks_zyx: Sequence[int],
        padding_zyx: Sequence[int],
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
        """
        Compute core and padded block bounds in z,y,x order.

        Returns:
            core_start, core_stop, pad_start, pad_stop
        """
        if len(block_index) != 3:
            raise ValueError("block_index must be (z_block, y_block, x_block)")
        if len(shape_zyx) != 3 or len(chunks_zyx) != 3 or len(padding_zyx) != 3:
            raise ValueError("shape_zyx, chunks_zyx, and padding_zyx must each have length 3")

        core_start = []
        core_stop = []
        pad_start = []
        pad_stop = []

        for i in range(3):
            start = int(block_index[i] * chunks_zyx[i])
            stop = min(start + int(chunks_zyx[i]), int(shape_zyx[i]))
            p = int(padding_zyx[i])

            core_start.append(start)
            core_stop.append(stop)
            pad_start.append(max(0, start - p))
            pad_stop.append(min(int(shape_zyx[i]), stop + p))

        return tuple(core_start), tuple(core_stop), tuple(pad_start), tuple(pad_stop)

    @staticmethod
    def process_block(
        zarr_src,
        zarr_dst,
        block_index: Sequence[int],
        transform: sitk.Transform,
        padding_zyx: Sequence[int],
        spacing_zyx: Sequence[float],
        background_color: int
    ):
        """
        Resample one spatial Z,Y,X block.

        Supports:

            grayscale:
                source.shape == (Z,Y,X)

            RGB:
                source.shape == (Z,Y,X,3)

            RGBA:
                source.shape == (Z,Y,X,4)

        The channel dimension is never included in spatial padding.
        """

        if zarr_src.shape != zarr_dst.shape:
            raise ValueError(
                "Source and destination shapes must match."
            )

        if len(zarr_src.shape) not in (3, 4):
            raise ValueError(
                f"Expected 3-D or 4-D Zarr, "
                f"got {zarr_src.shape}"
            )

        channels = (
            1
            if len(zarr_src.shape) == 3
            else zarr_src.shape[-1]
        )

        shape_zyx = _spatial_shape(
            zarr_src
        )

        chunks_zyx = _spatial_chunks(
            zarr_src
        )

        (
            core_start,
            core_stop,
            pad_start,
            pad_stop,
        ) = StackRegistration._compute_block_bounds(
            block_index=block_index,
            shape_zyx=shape_zyx,
            chunks_zyx=chunks_zyx,
            padding_zyx=padding_zyx,
        )

        core_size_zyx = tuple(
            core_stop[i] - core_start[i]
            for i in range(3)
        )

        if any(
            size <= 0
            for size in core_size_zyx
        ):
            raise ValueError(
                f"Invalid block size: "
                f"{core_size_zyx}")

        # --------------------------------------------------------------
        # READ SOURCE BLOCK
        # --------------------------------------------------------------

        if channels == 1:

            src = zarr_src[
                pad_start[0]:pad_stop[0],
                pad_start[1]:pad_stop[1],
                pad_start[2]:pad_stop[2],
            ]

        else:

            src = zarr_src[
                pad_start[0]:pad_stop[0],
                pad_start[1]:pad_stop[1],
                pad_start[2]:pad_stop[2],
                :,
            ]

        src = np.asarray(src)

        # --------------------------------------------------------------
        # REFERENCE GEOMETRY
        # --------------------------------------------------------------

        spacing_xyz = (
            StackRegistration
            ._zyx_to_xyz(spacing_zyx))

        origin_xyz = (0.0,0.0,0.0)

        source_origin_xyz = (
            origin_xyz[0] + pad_start[2] * spacing_xyz[0],
            origin_xyz[1] + pad_start[1] * spacing_xyz[1],
            origin_xyz[2] + pad_start[0] * spacing_xyz[2],
        )

        core_origin_xyz = (
            origin_xyz[0] + core_start[2] * spacing_xyz[0],
            origin_xyz[1] + core_start[1] * spacing_xyz[1],
            origin_xyz[2] + core_start[0] * spacing_xyz[2])

        reference = sitk.Image(
            int(core_size_zyx[2]),
            int(core_size_zyx[1]),
            int(core_size_zyx[0]),
            sitk.sitkFloat32,
        )

        reference.SetSpacing(spacing_xyz)
        reference.SetOrigin(core_origin_xyz)
        reference.SetDirection(np.eye(3).flatten())

        # --------------------------------------------------------------
        # RESAMPLE
        # --------------------------------------------------------------

        if channels == 1:

            out_core = (
                _resample_scalar_block(
                    src_zyx=src,
                    src_origin_xyz=source_origin_xyz,
                    reference=reference,
                    transform=transform,
                    background_color=background_color
                )
            )

            out_core = out_core.astype(zarr_dst.dtype,copy=False)

        else:

            out_core = (
                _resample_rgb_block(
                    src_zyxc=src,
                    src_origin_xyz=source_origin_xyz,
                    reference=reference,
                    transform=transform,
                    output_dtype=zarr_dst.dtype,
                    background_color=background_color
                )
            )

        # --------------------------------------------------------------
        # WRITE
        # --------------------------------------------------------------

        z_slice = slice(core_start[0],core_stop[0])
        y_slice = slice(core_start[1],core_stop[1])
        x_slice = slice(core_start[2],core_stop[2])

        if channels == 1:

            zarr_dst[
                z_slice,
                y_slice,
                x_slice,
            ] = out_core

        else:

            zarr_dst[
                z_slice,
                y_slice,
                x_slice,
                :,
            ] = out_core

        return (z_slice, y_slice, x_slice)




    def status(self):
        print(f'Moving spacing: {self.moving_spacing}')
        print(f'Fixed spacing: {self.fixed_spacing}')
        brain_paths = [self.moving_zarr_path, self.fixed_zarr_path, self.registered_zarr_path]
        for brain_path in brain_paths:
            if os.path.exists(brain_path):
                zarr_data = zarr.open(brain_path, mode='r')
                print(brain_path)
                print(zarr_data.info)
                channels = (_channel_count(zarr_data))
                print(f"Channels: {channels}")

            else:
                print(f'Missing: {brain_path}')
        if os.path.exists(self.transform_path):
            print(f'Found transform: {self.transform_path}')
        else:
            print(f'Missing transform: {self.transform_path}')

    def get_volume(self, brain):
        tif_path = os.path.join(self.base_path, brain, 'preps', 'C1', f'source_aligned.{self.downsample}')
        if not os.path.exists(tif_path):
            print(f"Missing: {tif_path}")
            exit(1)

        image = StackRegistration.create_sitk_volume(tif_path)
        return image


    def validate_registration(self):

        #moving
        moving_path = os.path.join(self.reg_path, self.moving, f'source.{self.downsample}.nii')
        if os.path.exists(moving_path):
            moving_sitk = sitk.ReadImage(moving_path)
            print(f'Loading existing moving sitk {moving_path}')
        else:
            moving_sitk = self.get_volume(self.moving)
            moving_sitk.SetSpacing(self.moving_spacing)
            sitk.WriteImage(sitk.Cast(moving_sitk, sitk.sitkUInt16), moving_path)
            print(f'Wrote moving image to: {moving_path}')

        if self.debug:
            print(f'Moving spacing: {moving_sitk.GetSpacing()}, ndim: {moving_sitk.GetDimension()}, channels: {moving_sitk.GetNumberOfComponentsPerPixel()}')
            print(f'\tsize: {moving_sitk.GetSize()}')
            return

        # fixed    
        fixed_path = os.path.join(self.reg_path, self.fixed, f'source.{self.downsample}.nii')
        if os.path.exists(fixed_path):
            fixed_sitk = sitk.ReadImage(fixed_path)
            print(f'Loading existing fixed sitk {fixed_path}')
        else:
            fixed_sitk = self.get_volume(self.fixed)
            fixed_sitk.SetSpacing(self.fixed_spacing)
            sitk.WriteImage(sitk.Cast(fixed_sitk, sitk.sitkUInt16), fixed_path)
            print(f'Wrote fixed image to: {fixed_path}')
        registered_mask_path = os.path.join(self.reg_path, self.moving, f'registered_mask.{self.downsample}.nii')
        if os.path.exists(self.preview_path):
            registered_image = sitk.ReadImage(self.preview_path)
            print(f'Loading existing registered image {self.preview_path}')
        else:            
            if not os.path.exists(self.transform_path):
                print(f"Missing: {self.transform_path}")
                exit(1)
            transform = sitk.ReadTransform(self.transform_path)

            resample = sitk.ResampleImageFilter()
            resample.SetTransform(transform)
            resample.SetInterpolator(sitk.sitkLinear)
            resample.SetReferenceImage(fixed_sitk)
            resample.SetDefaultPixelValue(0)
            registered_image = resample.Execute(moving_sitk)
            registered_image.SetSpacing(self.fixed_spacing)
            sitk.WriteImage(sitk.Cast(registered_image, sitk.sitkUInt16), self.preview_path)
            print(f'Wrote resampled image to: {self.preview_path}')

        ##### moving mask
        moving_mask_path = os.path.join(self.reg_path, self.moving, f'mask.{self.downsample}.nii')
        if os.path.exists(moving_mask_path):
            moving_mask = sitk.ReadImage(moving_mask_path)
            moving_mask.SetSpacing(self.moving_spacing)
            print(f'Loading existing moving_mask {moving_mask_path}')
        else:
            moving_mask = create_tissue_mask(moving_sitk, threshold=20)
            moving_mask.SetSpacing(self.moving_spacing)
            sitk.WriteImage(sitk.Cast(moving_mask, sitk.sitkUInt8), moving_mask_path)
            print(f'Finished creating moving mask to {moving_mask_path}')
        ##### fixed mask
        fixed_mask_path = os.path.join(self.reg_path, self.fixed, f'mask.{self.downsample}.nii')
        if os.path.exists(fixed_mask_path):
            fixed_mask = sitk.ReadImage(fixed_mask_path)
            fixed_mask.SetSpacing(self.fixed_spacing)
            print(f'Loading existing fixed_mask {fixed_mask_path}')
        else:
            fixed_mask = create_tissue_mask(fixed_sitk, threshold=20)
            fixed_mask.SetSpacing(self.fixed_spacing)
            sitk.WriteImage(sitk.Cast(fixed_mask, sitk.sitkUInt8), fixed_mask_path)
            print(f'Finished creating fixed mask to {fixed_mask_path}')
            
        if os.path.exists(registered_mask_path):
            registered_mask = sitk.ReadImage(registered_mask_path)
            registered_mask.SetSpacing(self.fixed_spacing)
        else:
            registered_mask = create_tissue_mask(registered_image, threshold=20)
            print('Finished creating registered mask')
            registered_mask.SetSpacing(self.fixed_spacing)
            sitk.WriteImage(sitk.Cast(registered_mask, sitk.sitkUInt8), registered_mask_path)

        print(f'Size in voxels, fixed: {fixed_mask.GetSize()} moving: {registered_image.GetSize()}')
        print(f'Resolution in micrometers fixed: {fixed_mask.GetSpacing()} moving: {registered_image.GetSpacing()}')

        metrics = compute_registration_metrics(
            fixed_mask,
            registered_mask,
        )

        print(f"Dice:                  {metrics.dice:.4f}")
        print(f"Jaccard:               {metrics.jaccard:.4f}")
        print(f"Centroid displacement: {metrics.centroid_distance:.3f} µm")

        print()
        print(f"Fixed volume:          {metrics.fixed_volume:,.1f} µm³")
        print(f"Moving volume:         {metrics.moving_volume:,.1f} µm³")
        print(f"Intersection volume:   {metrics.intersection_volume:,.1f} µm³")        


        """
        intersection = sitk.And(fixed_mask, moving_mask)
        print(f'intersection size: {intersection.GetSize()} depth: {intersection.GetDepth()} spacing: {intersection.GetSpacing()}')

        fixed_only = fixed_mask & ~moving_mask
        print(f'fixed only size: {fixed_only.GetSize()} depth: {fixed_only.GetDepth()} spacing: {fixed_only.GetSpacing()}')
        moving_only = moving_mask & ~fixed_mask        
        print(f'moving only size: {moving_only.GetSize()} depth: {moving_only.GetDepth()} spacing: {moving_only.GetSpacing()}')
        """

    def testing(self):
        transform = sitk.ReadTransform(self.transform_path)
        print(f'Transform: {self.transform_path}')

        #DK52_6N_L_points_m = (0.0131708,0.0063383599999999995,0.0044)
        #mxum, myum, mzum = (p * M_UM_SCALE for p in DK52_6N_L_points_m)
        #DK52_6N_L_voxels = (mxum/self.moving_xy_resolution, myum/self.moving_xy_resolution, mzum/self.moving_z_resolution)
        #print('DK52 6NL voxels', DK52_6N_L_voxels)
        DK52_voxels = (32449,12798,80)
        print('DK52 voxels', DK52_voxels)
        mx, my,mz = DK52_voxels 
        mxum = mx * self.moving_xy_resolution
        myum = my * self.moving_xy_resolution
        mzum = mz * self.moving_z_resolution
        xt, yt, zt = transform.GetInverse().TransformPoint((mxum, myum, mzum)) # transformed data to fixed space in µm
        #print('reg points um', xt, yt, zt)
        registered_voxels = xt / self.fixed_xy_resolution, yt / self.fixed_xy_resolution, zt / self.fixed_z_resolution
        print('reg voxels',registered_voxels)

        DK55_voxels = (27251,11924,112)
        print('DK55 voxels', DK55_voxels)
        return
        fx,fy,fz = DK55_voxels
        fxum = fx * 0.325
        fyum = fy * 0.325
        fzum = fz * 20

        xt, yt, zt = transform.GetInverse().TransformPoint((fxum, fyum, fzum)) # transformed data to fixed space in µm
        #print('reg points um', xt, yt, zt)
        registered_voxels = xt / self.fixed_xy_resolution, yt / self.fixed_xy_resolution, zt / self.fixed_z_resolution
        print('Inv voxels',registered_voxels)
        return

        
        
        xx = abs(xt - fxum)
        yy = abs(yt - fyum)
        zz = abs(zt - fzum)
        print('diff', xx, yy, zz)
        distance = math.dist((xt, yt, zt),
                    (fxum, fyum, fzum))
        print('distance', distance)
        return
        
        
        xt, yt, zt = transform.GetInverse().TransformPoint(DK52_6N_L_points_um) # transformed data to fixed space in µm
        print('inv points um', xt, yt, zt)
        xx = abs(xt - fx)
        yy = abs(yt - fy)
        zz = abs(zt - fz)
        print('diff', xx, yy, zz)
        distance = math.dist((xt, yt, zt),
                    (fx, fy, fz))
        print('inv distance', distance)        





    def convert_points(self):
        """ Session IDs for volumes in the DB
        --  DK55 IDs, 5N_L=8464, 5N_R=8465, 6N_L=8527, 6N_R=8526
        -- MD585 IDs, 5N_L=7934, 5N_R=7935, 6N_L=7936, 6N_R=7937
        """
        #
        def create_com_from_db(session_id, transform):
            sqlController = SqlController(self.moving)
            registered_polygons = defaultdict(list)
            annotation_session = sqlController.get_annotation_by_id(session_id=session_id)
            if annotation_session is None:
                print(f'No annotations found for {session_id=}')
                return

            annotation = annotation_session.annotation
            # first test data to make sure it has the right keys
            try:
                data = annotation["childJsons"]
            except KeyError as ke:
                print(f'No data for {annotation_session.FK_prep_id} was found. {ke}')
                return


            for row in data:
                if 'childJsons' not in row:
                    return
                for child in row['childJsons']:
                    xm0,ym0,zm0 = child['pointA']
                    #convert to um from meters
                    xm0 = xm0 * M_UM_SCALE   # in µm
                    ym0 = ym0 * M_UM_SCALE      # in µm
                    zm0 = zm0 * M_UM_SCALE      # in µm
                    # need to convert these to the same physical space as the transform = 1/32
                    #xm0 = xm0 / self.fixed_xy_resolution
                    #ym0 = ym0 / self.fixed_xy_resolution
                    #zm0 = zm0 / self.fixed_z_resolution


                    if transform is None:
                        xt, yt, zt = (xm0, ym0, zm0) # transformed data to fixed space in µm
                    else:
                        xt, yt, zt = transform.GetInverse().TransformPoint((xm0, ym0, zm0)) # transformed data to fixed space in µm
                        """
                        points_hr = np.array([
                            [xm0,ym0, zm0],
                        ])
                        
                        registered_points = self.transform_points_between_spacings(
                            points_hr,
                            transform,
                            source_spacing=(0.325, 0.325, 20.0),
                            target_spacing=(0.325, 0.325, 20.0),)
                        xt = registered_points[0,0]
                        yt = registered_points[0,1]
                        zt = registered_points[0,2]
                        
                        #xt, yt, zt = transform.GetInverse().TransformPoint((xm0, ym0, zm0)) # transformed data to fixed space in µm
                        """
                    #xt /= self.fixed_xy_resolution
                    #yt /= self.fixed_xy_resolution
                    #zt /= self.fixed_z_resolution

                    section = int(np.round(zt))
                    registered_polygons[section].append((yt, xt))

            zyx_list = []
            for section, points in registered_polygons.items():
                section_points = [(section, y,x) for y,x in points]
                for section_point in section_points:
                    zyx_list.append(section_point)

            return np.mean(zyx_list, axis=0)

        #fixed_sessions = {'5N_L':8464, '5N_R':8465, '6N_L':8527, '6N_R':8526}
        #moving_sessions = {'5N_L':7934, '5N_R':7935, '6N_L':7936, '6N_R':7937}
        moving_sessions = {'SC':7432, '6N_L':7411}
        fixed_sessions = {'SC':7706, '6N_L':611}
        transform = sitk.ReadTransform(self.transform_path)
        points_hr = np.array([
            [34332,12924,233],
            [32087,19040,84],
        ])

        points_registration = self.transform_points_between_spacings(
            points_hr,
            transform.GetInverse(),
            source_spacing=(0.325, 0.325, 20.0),
            target_spacing=(0.325, 0.325, 20.0),
        )
        print(points_registration)
        print()
        print('want DK55 center',35705,12798,230)
        print('want DK55 right',29575,15397,74)

        data = []

        for (moving_structure, moving_session_id), (fixed_structure, fixed_session_id) in zip(moving_sessions.items(), fixed_sessions.items()):
            try:
                mz,my,mx = create_com_from_db(session_id=moving_session_id, transform=transform)
            except:
                print(f'Error getting com for {moving_session_id}')    
             
            fz,fy,fx = create_com_from_db(session_id=fixed_session_id, transform=None)
            distance = math.dist((mx, my, mz),
                                (fx, fy, fz))        
            row_dict = {
                    "Moving": self.moving,
                    "Structure": moving_structure,
                    "mx": mx,
                    "my": my,
                    "mz": mz,
                    "Fixed": self.fixed,
                    "Structure": fixed_structure,
                    "fx": fx,
                    "fy": fy,
                    "fz": fz,
                    "change x": abs(mx - fx),
                    "change y": abs(my - fy),
                    "change z": abs(mz - fz),
                    "Distance um": distance
                }
            
            data.append(row_dict)
 
        df = pd.DataFrame(data=data)
        df = df.round(0)
        df['mx'] = df['mx'].astype(int)
        df['my'] = df['my'].astype(int)
        df['mz'] = df['mz'].astype(int)
        df['fx'] = df['fx'].astype(int)
        df['fy'] = df['fy'].astype(int)
        df['fz'] = df['fz'].astype(int)
        print(df.head(20))

    def create_masks(self):
        def create_mask(image):
            mask = sitk.BinaryThreshold(
                image,
                lowerThreshold=222,
                upperThreshold=255,
                insideValue=0,
                outsideValue=255
            )

            radius = [48,48]

            mask = sitk.BinaryMorphologicalClosing(mask, kernelRadius=radius)
            mask = sitk.BinaryMorphologicalOpening(mask, radius)
            mask = sitk.Cast(mask, sitk.sitkUInt8)

            return mask

        masked_path = os.path.join(self.base_path, self.moving,  'preps', 'C1', 'masked')
        if os.path.exists(masked_path):
            shutil.rmtree(masked_path)
        os.makedirs(masked_path, exist_ok=True)
        files = sorted(os.listdir(self.moving_tif_path))
        for f in tqdm(files, desc="creating 2D masks", disable=self.debug):
            tif_path = os.path.join(self.moving_tif_path, f)
            image = sitk.ReadImage(tif_path, sitk.sitkUInt8)
            #image = sitk.VectorIndexSelectionCast(image, 1)
            mask = create_mask(image)
            masked_image = sitk.Mask(image, mask)
            masked_file_path = os.path.join(masked_path, f)
            sitk.WriteImage(masked_image, masked_file_path)


    @staticmethod
    def transform_points_between_spacings(
        points,
        transform,
        source_spacing,
        target_spacing,
        source_origin=(0.0, 0.0, 0.0),
        target_origin=(0.0, 0.0, 0.0),
    ):
        """
        Transform voxel-index coordinates from one image space to another.

        Parameters
        ----------
        points : (N, 3) ndarray
            Points in (x, y, z) voxel-index coordinates.

        transform : sitk.Transform
            SimpleITK transform operating in physical coordinates.

        source_spacing : tuple
            (sx, sy, sz) physical spacing of the source points.

        target_spacing : tuple
            (sx, sy, sz) physical spacing of the target image.

        source_origin : tuple
            Physical origin of source image.

        target_origin : tuple
            Physical origin of target image.

        Returns
        -------
        target_points : (N, 3) ndarray
            Transformed points in target voxel-index coordinates.
        """

        points = np.asarray(points, dtype=np.float64)
        source_spacing = np.asarray(source_spacing, dtype=np.float64)
        target_spacing = np.asarray(target_spacing, dtype=np.float64)

        source_origin = np.asarray(source_origin, dtype=np.float64)
        target_origin = np.asarray(target_origin, dtype=np.float64)

        # Voxel index -> physical coordinates
        physical_points = (
            points * source_spacing
            + source_origin
        )

        # Apply SimpleITK transform in physical space
        transformed_physical = np.asarray([
            transform.TransformPoint(tuple(p))
            for p in physical_points
        ])

        # Physical coordinates -> target voxel indices
        target_points = (
            transformed_physical - target_origin
        ) / target_spacing

        return target_points


    def test_itk(self):
        import itk

        #moving
        moving_path = os.path.join(self.reg_path, self.moving, f'source.{self.downsample}.nii')
        if os.path.exists(moving_path):
            moving_sitk = sitk.ReadImage(moving_path)
            print(f'Loading existing moving sitk {moving_path}')
        else:
            moving_sitk = self.get_volume(self.moving)
            moving_sitk.SetSpacing(self.moving_spacing)
            sitk.WriteImage(sitk.Cast(moving_sitk, sitk.sitkUInt16), moving_path)
            print(f'Wrote moving image to: {moving_path}')
        print(f'Moving spacing: {moving_sitk.GetSpacing()}, ndim: {moving_sitk.GetDimension()}, channels: {moving_sitk.GetNumberOfComponentsPerPixel()}')
        print(f'\tsize: {moving_sitk.GetSize()}')

        # fixed    
        fixed_path = os.path.join(self.reg_path, self.fixed, f'source.{self.downsample}.nii')
        if os.path.exists(fixed_path):
            fixed_sitk = sitk.ReadImage(fixed_path)
            print(f'Loading existing fixed sitk {fixed_path}')
        else:
            fixed_sitk = self.get_volume(self.fixed)
            fixed_sitk.SetSpacing(self.fixed_spacing)
            sitk.WriteImage(sitk.Cast(fixed_sitk, sitk.sitkUInt16), fixed_path)
            print(f'Wrote fixed image to: {fixed_path}')

        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixed_sitk)
        elastixImageFilter.SetMovingImage(moving_sitk)

        genericMap = create_affine_parameters(elastixImageFilter=elastixImageFilter)
        elastixImageFilter.SetParameterMap(genericMap)
        elastixImageFilter.SetParameter("MaximumNumberOfIterations", "300")
        elastixImageFilter.SetParameter("ResultImageFormat", "nii")
        elastixImageFilter.SetParameter("NumberOfResolutions", "6") #### Very important, less than 6 gives lousy results.
        elastixImageFilter.SetParameter("DefaultPixelValue", "0")
        #elastixImageFilter.SetParameter("ComputeZYX", "true")
        
        elastixImageFilter.PrintParameterMap
        elastixImageFilter.SetLogToFile(True)
        elastixImageFilter.LogToConsoleOff()

        elastixImageFilter.SetLogFileName('elastix.log')
        output_directory = "./elastixoutput"
        os.makedirs(output_directory, exist_ok=True)
        elastixImageFilter.SetOutputDirectory(output_directory)
        elastixImageFilter.PrintParameterMap()
        resultImage = elastixImageFilter.Execute()
                
        resultImage = sitk.Cast(sitk.RescaleIntensity(resultImage), sitk.sitkUInt16)

        sitk.WriteImage(resultImage, "reg.volume.nii")
        parameterMap = elastixImageFilter.GetTransformParameterMap()[0]
        TransformParameters = parameterMap['TransformParameters']
        center = parameterMap['CenterOfRotationPoint']
        print(f'TransformParameters: {TransformParameters}')
        print(f'Center of rotation point: {center}')
        R = [float(x) for x in TransformParameters[:9]]
        t = [float(x) for x in TransformParameters[9:]]
        c = [float(x) for x in center]

        affine_transform = sitk.AffineTransform(3)
        affine_transform.SetMatrix(R)
        affine_transform.SetTranslation(t)
        affine_transform.SetCenter(c)

        sitk.WriteTransform(affine_transform, self.transform_path)

        




                  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--moving', help='Enter the animal (moving)', required=True, type=str)
    parser.add_argument('--fixed', help='Enter the animal (fixed)', required=True, type=str)
    parser.add_argument("--task", help="Enter the task you want to perform", required=True, default="status", type=str)
    parser.add_argument("--downsample", help="Enter the downsample", required=False, default=1, type=int)
    parser.add_argument(
        "--registration-channel",
        default="luminance",
        choices=[
            "luminance",
            "red",
            "green",
            "blue",
        ],
        required=False,
        help=("Channel used to calculate the registration transform for RGB images. Default: luminance."),)

    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)

    args = parser.parse_args()
    moving_brain = args.moving
    fixed_brain = args.fixed
    task = str(args.task).strip().lower()
    downsample = args.downsample
    registration_channel = args.registration_channel
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    pipeline = StackRegistration(moving_brain, fixed_brain, downsample, registration_channel, debug)


    function_mapping = {
        "create_zarr": pipeline.create_zarr,
        "create_transform": pipeline.create_transform,
        "create_registered_tiles": pipeline.create_registered_tiles,
        "zarr2tif": pipeline.create_tifs,
        "create_neuroglancer": pipeline.create_neuroglancer,
        "run_tiles": pipeline.run_tiles,
        "status": pipeline.status,
        "test_mips": pipeline.test_mips,
        "validate": pipeline.validate_registration,
        "testing": pipeline.testing,
        "convert_points": pipeline.convert_points,
        "create_masks": pipeline.create_masks,
        "test_itk": pipeline.test_itk
    }

    if task in function_mapping:
        function_mapping[task]()
    else:
        print(f'{task} is not a correct task. Choose one of these:')
        for key in function_mapping.keys():
            print(f'\t{key}')
