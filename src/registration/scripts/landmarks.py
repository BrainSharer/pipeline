from __future__ import annotations

import argparse
import json
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
from timeit import default_timer as timer



PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())


from library.image_manipulation.image_manager import ImageManager
from library.image_manipulation.neuroglancer_manager import NumpyToNeuroglancer
from library.image_manipulation.precomputed_manager import NgPrecomputedMaker
from registration.scripts.sitk_helpers import compute_affine_padding, compute_chunk_source_region, compute_registration_metrics, create_tissue_mask
from library.controller.sql_controller import SqlController


class StackRegistration:

    def __init__(
        self,
        moving,
        fixed,
        downsample=1,
        debug=False,
        landmark_path=None,
    ):
        self.moving = moving
        self.fixed = fixed
        self.downsample = downsample
        self.landmark_path = landmark_path
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
        self.debug = debug


    def create_zarr(self):
        if not os.path.exists(self.moving_tif_path):
            print(f"Input path {self.moving_tif_path} does not exist for brain {self.moving}")
            exit(1)
        divisors = {}
        divisors[1] = 32
        divisors[4] = 16
        divisors[8] = 8
        divisors[16] = 4
        divisors[32] = 2
        try:
            divisor = divisors[self.downsample]
        except KeyError:
            divisor = 4
        image_manager = ImageManager(self.moving_tif_path)
        chunk_x = image_manager.width//divisor
        chunk_y = image_manager.height
        chunk_z = image_manager.len_files//divisor
        dims = image_manager.ndim
        print('dims', dims)
        if dims == 3:
            rechunks_zyx = (chunk_z, chunk_y, chunk_x, -1)
        else:
            rechunks_zyx = (chunk_z, chunk_y, chunk_x)
        if os.path.exists(self.moving_zarr_path):
            print(f"Output path {self.moving_zarr_path} already exists")
            print(f"\tfor brain {self.moving_zarr_path}, skipping zarr creation")
            return

        print(f'{self.moving} input {self.moving_tif_path}')
        print(f'{self.moving} output {self.moving_zarr_path}')


        dask_imgs = StackRegistration.build_dask_array_from_folder(self.moving_tif_path)
        print(f'Using chunks={rechunks_zyx}')
        dask_imgs = dask_imgs.rechunk(rechunks_zyx)
        print(f'Dask array shape: {dask_imgs.shape} chunk size = {dask_imgs.chunksize}')
        with ProgressBar():
            dask_imgs.to_zarr(self.moving_zarr_path, overwrite=True)
        del dask_imgs
        print(f"✅ Source zarr saved to {self.moving_zarr_path}")
        volume = zarr.open(self.moving_zarr_path, mode='r')
        print(volume.info)
        del volume


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
        fixed_sitk = StackRegistration.create_sitk_volume(self.fixed_tif_path)
        moving_sitk = StackRegistration.create_sitk_volume(self.moving_tif_path)
        moving_sitk.SetSpacing(self.moving_spacing)
        fixed_sitk.SetSpacing(self.fixed_spacing)
        print(f'\nMoving sitk info size={moving_sitk.GetSize()} spacing={moving_sitk.GetSpacing()} dimension={moving_sitk.GetNumberOfComponentsPerPixel()}')
        print(f'Fixed sitk info size={fixed_sitk.GetSize()} spacing={fixed_sitk.GetSpacing()} channels={fixed_sitk.GetNumberOfComponentsPerPixel()}')
        landmark_data = StackRegistration._load_landmarks(
            self.landmark_path
        ) if self.landmark_path else None

        if landmark_data is not None:
            print(
                f"Loaded {len(landmark_data['names'])} landmarks "
                f"from {landmark_data['path']}"
            )
            affine_transform = StackRegistration.affine_registration(
                fixed_sitk,
                moving_sitk,
                fixed_landmarks=landmark_data["fixed"],
                moving_landmarks=landmark_data["moving"],
                landmark_names=landmark_data["names"],
            )
        else:
            affine_transform = StackRegistration.affine_registration(
                fixed_sitk,
                moving_sitk,
            )

        sitk.WriteTransform(affine_transform, self.transform_path)


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

        if len(source.chunks) == 4:
            chunkz, chunky, chunkx, _ = source.chunks
            padding = (chunkz//2, chunky//2, chunkx//2, 1)
        else:
            chunkz, chunky, chunkx = source.chunks
            padding = (chunkz//2, chunky//2, chunkx//2)
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
        )
        
        registered_volume = zarr.open(self.registered_zarr_path, mode='r')
        print(registered_volume.info)
        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        print(f"create registered tiles took {total_elapsed_time} seconds")

    def create_tifs(self):
        if os.path.exists(self.registered_tif_path):
            shutil.rmtree(self.registered_tif_path)
        os.makedirs(self.registered_tif_path, exist_ok=True)
        if not os.path.exists(self.registered_zarr_path):
            print(f'No zarr at {self.registered_zarr_path}')
            return
        volume = zarr.open(self.registered_zarr_path, mode='r')
        nz = volume.shape[0]

        for z in tqdm(range(nz), desc="Creating TIFs"):
            img = volume[z]
            outpath = os.path.join(self.registered_tif_path, f"{z:03d}.tif")
            tifffile.imwrite(outpath, img)
        print(f'Finished writing tifs to {self.registered_tif_path}')


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
            return arr
        return _read(path)

    @staticmethod
    def build_dask_array_from_folder(folder: str, pattern: str = "*.tif", sample_index: int = 0):
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
    def _validate_landmarks(
        fixed_landmarks,
        moving_landmarks,
        min_landmarks=4,
    ):
        """
        Validate corresponding 3-D physical-space landmarks.

        Returns
        -------
        fixed : np.ndarray, shape (N, 3)
        moving : np.ndarray, shape (N, 3)
        """
        if fixed_landmarks is None or moving_landmarks is None:
            raise ValueError(
                "Both fixed_landmarks and moving_landmarks are required."
            )

        fixed = np.asarray(fixed_landmarks, dtype=np.float64)
        moving = np.asarray(moving_landmarks, dtype=np.float64)

        if fixed.ndim != 2 or fixed.shape[1] != 3:
            raise ValueError(
                f"fixed_landmarks must have shape (N, 3), got {fixed.shape}"
            )
        if moving.ndim != 2 or moving.shape[1] != 3:
            raise ValueError(
                f"moving_landmarks must have shape (N, 3), got {moving.shape}"
            )
        if fixed.shape != moving.shape:
            raise ValueError(
                "fixed_landmarks and moving_landmarks must have identical shapes: "
                f"{fixed.shape} != {moving.shape}"
            )
        if fixed.shape[0] < min_landmarks:
            raise ValueError(
                f"At least {min_landmarks} corresponding 3-D landmarks are required; "
                f"got {fixed.shape[0]}"
            )
        if not np.isfinite(fixed).all() or not np.isfinite(moving).all():
            raise ValueError("Landmarks must contain only finite values.")

        # Reject duplicate/near-duplicate points because they make an affine
        # initialization poorly conditioned.
        if fixed.shape[0] > 1:
            fixed_dist = np.linalg.norm(
                fixed[:, None, :] - fixed[None, :, :], axis=2
            )
            moving_dist = np.linalg.norm(
                moving[:, None, :] - moving[None, :, :], axis=2
            )
            fixed_nonzero = fixed_dist[np.triu_indices(fixed.shape[0], k=1)]
            moving_nonzero = moving_dist[np.triu_indices(moving.shape[0], k=1)]

            if np.any(fixed_nonzero <= 1e-6):
                raise ValueError("Fixed landmarks contain duplicate/near-duplicate points.")
            if np.any(moving_nonzero <= 1e-6):
                raise ValueError("Moving landmarks contain duplicate/near-duplicate points.")

        return fixed, moving

    @staticmethod
    def _landmark_affine(
        fixed_landmarks,
        moving_landmarks,
    ):
        """
        Create a SimpleITK affine transform initialized from corresponding
        physical-space landmarks.

        The resulting transform follows the same convention used by
        sitk.Resample: it maps fixed/output physical coordinates to
        moving/input coordinates.
        """
        fixed, moving = StackRegistration._validate_landmarks(
            fixed_landmarks,
            moving_landmarks,
        )

        transform = sitk.AffineTransform(3)

        transform = sitk.LandmarkBasedTransformInitializer(
            transform,
            fixed.flatten().tolist(),
            moving.flatten().tolist(),
        )

        return transform

    @staticmethod
    def _landmark_residuals(
        transform,
        fixed_landmarks,
        moving_landmarks,
    ):
        """
        Calculate landmark residuals for a transform.

        Because SimpleITK's resampling convention is output/fixed -> input/moving,
        fixed landmarks are transformed and compared with moving landmarks.
        """
        fixed, moving = StackRegistration._validate_landmarks(
            fixed_landmarks,
            moving_landmarks,
        )

        predicted = np.asarray(
            [transform.TransformPoint(tuple(p)) for p in fixed],
            dtype=np.float64,
        )

        residual_vectors = predicted - moving
        residuals = np.linalg.norm(residual_vectors, axis=1)

        return residuals, residual_vectors

    @staticmethod
    def _landmark_report(
        transform,
        fixed_landmarks,
        moving_landmarks,
        names=None,
    ):
        """
        Print and return landmark residual statistics.
        """
        fixed, moving = StackRegistration._validate_landmarks(
            fixed_landmarks,
            moving_landmarks,
        )

        residuals, residual_vectors = StackRegistration._landmark_residuals(
            transform,
            fixed,
            moving,
        )

        if names is None:
            names = [f"landmark_{i:02d}" for i in range(len(residuals))]
        if len(names) != len(residuals):
            raise ValueError("Landmark names must have the same length as landmarks.")

        rms = float(np.sqrt(np.mean(residuals ** 2)))
        mean = float(np.mean(residuals))
        median = float(np.median(residuals))
        maximum = float(np.max(residuals))

        print("\nLandmark residuals:")
        for name, residual in zip(names, residuals):
            print(f"  {name:24s}: {residual:10.3f} µm")

        print(
            f"Landmark RMS={rms:.3f} µm, "
            f"mean={mean:.3f} µm, "
            f"median={median:.3f} µm, "
            f"max={maximum:.3f} µm"
        )

        return {
            "rms": rms,
            "mean": mean,
            "median": median,
            "max": maximum,
            "residuals": residuals.tolist(),
            "residual_vectors": residual_vectors.tolist(),
            "names": list(names),
        }

    @staticmethod
    def _load_landmarks(path):
        """
        Load corresponding landmarks from JSON.

        Supported format
        ----------------
        {
          "landmarks": [
            {
              "name": "anterior",
              "fixed": [x, y, z],
              "moving": [x, y, z]
            }
          ]
        }

        Coordinates are physical SimpleITK coordinates in micrometers,
        ordered as (x, y, z).
        """
        if not path:
            return None

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Landmark file does not exist: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            entries = data.get("landmarks")
        elif isinstance(data, list):
            entries = data
        else:
            raise ValueError(
                "Landmark JSON must be either an object containing "
                "'landmarks' or a list of landmark objects."
            )

        if not isinstance(entries, list) or not entries:
            raise ValueError("No landmarks were found in the landmark JSON.")

        fixed = []
        moving = []
        names = []

        for i, item in enumerate(entries):
            if not isinstance(item, dict):
                raise ValueError(f"Landmark {i} must be an object.")

            if "fixed" not in item or "moving" not in item:
                raise ValueError(
                    f"Landmark {i} must contain both 'fixed' and 'moving'."
                )

            name = str(item.get("name", f"landmark_{i:02d}"))
            fixed_point = item["fixed"]
            moving_point = item["moving"]

            if len(fixed_point) != 3 or len(moving_point) != 3:
                raise ValueError(
                    f"Landmark '{name}' must contain 3-D fixed and moving coordinates."
                )

            fixed.append(fixed_point)
            moving.append(moving_point)
            names.append(name)

        fixed, moving = StackRegistration._validate_landmarks(
            fixed,
            moving,
        )

        return {
            "fixed": fixed,
            "moving": moving,
            "names": names,
            "path": str(path),
        }

    @staticmethod
    def index_to_physical_point(image, index_zyx):
        """
        Convert a numpy-style (z, y, x) voxel index to SimpleITK physical
        coordinates (x, y, z).
        """
        if len(index_zyx) != 3:
            raise ValueError("index_zyx must contain exactly three values.")

        z, y, x = [int(v) for v in index_zyx]
        return image.TransformIndexToPhysicalPoint((x, y, z))

    @staticmethod
    def physical_to_index_zyx(image, point_xyz):
        """
        Convert SimpleITK physical coordinates (x, y, z) to numpy-style
        (z, y, x) continuous index coordinates.
        """
        if len(point_xyz) != 3:
            raise ValueError("point_xyz must contain exactly three values.")

        continuous_index = image.TransformPhysicalPointToContinuousIndex(
            tuple(float(v) for v in point_xyz)
        )
        x, y, z = continuous_index
        return z, y, x

    @staticmethod
    def affine_registration(
        fixed,
        moving,
        fixed_landmarks=None,
        moving_landmarks=None,
        landmark_names=None,
        use_landmarks=True,
    ):
        """
        Register moving to fixed using an affine transform.

        If corresponding landmarks are supplied, they provide the initial
        affine transform. The intensity-based optimizer then refines that
        transform. Without landmarks, the existing geometry-based initializer
        is used.

        Landmark coordinates must be SimpleITK physical coordinates (x, y, z),
        in the same physical coordinate system as the images.

        Returns
        -------
        sitk.AffineTransform
            Final affine transform.
        """
        def command_iteration(method):
            print(
                f"Iteration: {method.GetOptimizerIteration()} "
                f"Metric: {method.GetMetricValue():.6f}"
            )

        if fixed.GetDimension() != 3 or moving.GetDimension() != 3:
            raise ValueError("affine_registration requires 3-D images.")

        if fixed.GetNumberOfComponentsPerPixel() == 3:
            fixed = sitk.VectorIndexSelectionCast(fixed, 1)
        if moving.GetNumberOfComponentsPerPixel() == 3:
            moving = sitk.VectorIndexSelectionCast(moving, 1)

        fixed = sitk.Cast(fixed, sitk.sitkFloat32)
        moving = sitk.Cast(moving, sitk.sitkFloat32)

        # The registration pipeline uses a common zero-origin Cartesian
        # coordinate system. Landmark coordinates must use this same system.
        fixed.SetOrigin((0.0, 0.0, 0.0))
        moving.SetOrigin((0.0, 0.0, 0.0))
        fixed.SetDirection(np.eye(3).flatten())
        moving.SetDirection(np.eye(3).flatten())

        print(
            f"Affine registration with fixed image size {fixed.GetSize()} "
            f"and moving image size {moving.GetSize()}"
        )
        print(f"Fixed image spacing: {fixed.GetSpacing()}")
        print(f"Moving image spacing: {moving.GetSpacing()}")

        if use_landmarks and (
            fixed_landmarks is not None or moving_landmarks is not None
        ):
            if fixed_landmarks is None or moving_landmarks is None:
                raise ValueError(
                    "Both fixed_landmarks and moving_landmarks must be supplied."
                )

            fixed_landmarks, moving_landmarks = StackRegistration._validate_landmarks(
                fixed_landmarks,
                moving_landmarks,
            )

            if landmark_names is None:
                landmark_names = [
                    f"landmark_{i:02d}"
                    for i in range(len(fixed_landmarks))
                ]

            print(
                f"Initializing affine transform from "
                f"{len(fixed_landmarks)} corresponding landmarks."
            )

            initial_transform = StackRegistration._landmark_affine(
                fixed_landmarks,
                moving_landmarks,
            )

            print("Initial landmark transform:")
            print(initial_transform)

            StackRegistration._landmark_report(
                initial_transform,
                fixed_landmarks,
                moving_landmarks,
                landmark_names,
            )

        else:
            print("Using geometry-based affine initialization.")

            initial_transform = sitk.CenteredTransformInitializer(
                fixed,
                moving,
                sitk.AffineTransform(fixed.GetDimension()),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )

        registration = sitk.ImageRegistrationMethod()

        # Correlation has worked better for this pipeline than Mattes MI.
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

        registration.SetOptimizerScalesFromPhysicalShift()
        registration.SetShrinkFactorsPerLevel([4, 2, 1])
        registration.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        registration.AddCommand(
            sitk.sitkIterationEvent,
            lambda: command_iteration(registration),
        )

        # Keep the initial transform separate so the landmark initialization
        # remains available for QC and debugging.
        final_transform = registration.Execute(
            fixed,
            moving,
            initial_transform,
        )

        print("\nAffine registration complete.")
        print(f"Final metric: {registration.GetMetricValue()}")
        print(
            "Optimizer stopping condition: ",
            registration.GetOptimizerStopConditionDescription(),
        )

        if fixed_landmarks is not None and moving_landmarks is not None:
            print("\nFinal landmark fit:")
            StackRegistration._landmark_report(
                final_transform,
                fixed_landmarks,
                moving_landmarks,
                landmark_names,
            )

        return final_transform

    @staticmethod
    def sitk_to_dask(image):
        arr = sitk.GetArrayFromImage(image)
        return da.from_array(
            arr,
            chunks=(32,512,512)
        )

    @staticmethod
    def create_sitk_volume(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.tif")))
        slices = []
        for f in tqdm(files, desc="Creating sitk volume"):
            img = tifffile.imread(f)
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
        spacing_zyx
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
                spacing_zyx=spacing_zyx
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
        spacing_zyx: Sequence[float]
    ) -> tuple[slice, slice, slice]:
        """
        Read one padded block from `zarr_src`, resample it with SimpleITK, and write
        the de-padded result into `zarr_dst`.

        Parameters
        ----------
        zarr_src : zarr.Array
            Source Zarr array in (z, y, x) order.
        zarr_dst : zarr.Array
            Destination Zarr array in (z, y, x) order. Must have the same shape as `zarr_src`.
        block_index : (int, int, int)
            Block index in chunk coordinates, in (z_block, y_block, x_block) order.
        transform : sitk.Transform
            Transform to use for resampling. By default, SimpleITK expects the transform
            to map output physical points to input physical points.
        padding_zyx : tuple[int, int, int]
            Extra padding around the block, in voxels, in (z, y, x) order.
        spacing_zyx : tuple[float, float, float]
            Voxel spacing in (z, y, x) order.

        Returns
        -------
        tuple[slice, slice, slice]
            The z, y, x slices written into `zarr_dst`.
        """
        if zarr_src.shape != zarr_dst.shape:
            raise ValueError("zarr_src and zarr_dst must have the same shape")

        if len(zarr_src.shape) != 3:
            raise ValueError("This function expects a 3D Zarr array in (z, y, x) order")

        shape_zyx = zarr_src.shape
        chunks_zyx = zarr_src.chunks if zarr_src.chunks is not None else zarr_dst.chunks
        if chunks_zyx is None:
            raise ValueError("Zarr chunks are required on at least one of zarr_src or zarr_dst")

        core_start, core_stop, pad_start, pad_stop = StackRegistration._compute_block_bounds(
            block_index=block_index,
            shape_zyx=shape_zyx,
            chunks_zyx=chunks_zyx,
            padding_zyx=padding_zyx,
        )

        # Read padded moving/source block in numpy order (z, y, x)
        src_pad = zarr_src[
            pad_start[0]:pad_stop[0],
            pad_start[1]:pad_stop[1],
            pad_start[2]:pad_stop[2],
        ]

        # Core output block size (z, y, x)
        core_size_zyx = (
            core_stop[0] - core_start[0],
            core_stop[1] - core_start[1],
            core_stop[2] - core_start[2],
        )
        #print(f'padding 0,1,2 {pad_start[0]}:{pad_stop[0]},{pad_start[1]}:{pad_stop[1]},{pad_start[2]}:{pad_stop[2]}')

        if any(s <= 0 for s in core_size_zyx):
            raise ValueError(f"Invalid core block size: {core_size_zyx}")

        # Convert spacing/origin from z,y,x to x,y,z for SimpleITK
        spacing_xyz = StackRegistration._zyx_to_xyz(spacing_zyx)
        origin_xyz = (0,0,0)

        # Build the moving image from the padded source block
        moving = sitk.GetImageFromArray(src_pad)
        moving = sitk.Cast(moving, sitk.sitkFloat32)
        moving.SetSpacing(spacing_xyz)
        moving.SetOrigin(
            (
                origin_xyz[0] + pad_start[2] * spacing_xyz[0],
                origin_xyz[1] + pad_start[1] * spacing_xyz[1],
                origin_xyz[2] + pad_start[0] * spacing_xyz[2],
            )
        )

        moving.SetDirection(np.eye(3).flatten())

        # Build a reference image for the core block
        reference = sitk.Image(
            int(core_size_zyx[2]),  # x
            int(core_size_zyx[1]),  # y
            int(core_size_zyx[0]),  # z
            sitk.sitkFloat32,
        )
        reference.SetSpacing(spacing_xyz)
        reference.SetOrigin(
            (
                origin_xyz[0] + core_start[2] * spacing_xyz[0],
                origin_xyz[1] + core_start[1] * spacing_xyz[1],
                origin_xyz[2] + core_start[0] * spacing_xyz[2],
            )
        )
        reference.SetDirection(np.eye(3).flatten())
        # Resample padded moving image into the core reference geometry
        resampled = sitk.Resample(
            moving,
            reference,
            transform,
            sitk.sitkLinear,
            0,
            sitk.sitkFloat32,
        )

        # Write the core block back to the destination Zarr
        out_core = sitk.GetArrayFromImage(resampled)
        out_core = np.asarray(out_core, dtype=zarr_dst.dtype)

        z_slice = slice(core_start[0], core_stop[0])
        y_slice = slice(core_start[1], core_stop[1])
        x_slice = slice(core_start[2], core_stop[2])

        zarr_dst[z_slice, y_slice, x_slice] = out_core
        return z_slice, y_slice, x_slice


    def status(self):
        print(f'Moving spacing: {self.moving_spacing}')
        print(f'Fixed spacing: {self.fixed_spacing}')
        brain_paths = [self.moving_zarr_path, self.fixed_zarr_path, self.registered_zarr_path]
        for brain_path in brain_paths:
            if os.path.exists(brain_path):
                zarr_data = zarr.open(brain_path, mode='r')
                print(brain_path)
                print(zarr_data.info)
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
            print(f'Loading existing fixed_sitk {moving_path}')
        else:
            moving_sitk = self.get_volume(self.moving)
            moving_sitk.SetSpacing(self.moving_spacing)
            sitk.WriteImage(sitk.Cast(moving_sitk, sitk.sitkUInt16), moving_path)
            print(f'Wrote moving image to: {moving_path}')

        # fixed    
        fixed_path = os.path.join(self.reg_path, self.fixed, f'source.{self.downsample}.nii')
        if os.path.exists(fixed_path):
            fixed_sitk = sitk.ReadImage(fixed_path)
            print(f'Loading existing fixed_sitk {fixed_path}')
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

    def test_padding(self):
        transform = sitk.ReadTransform(self.transform_path)
        matrix = np.array(transform.GetMatrix()).reshape(3, 3)
        print(matrix)
        translation = transform.GetTranslation()
        print(translation)

        chunk_shape = (57, 154, 582)
        padding = compute_affine_padding(
            matrix=matrix,
            translation=translation,
            chunk_shape=chunk_shape,
            spacing=self.spacing[::-1],
        )

        print(padding)
        region = compute_chunk_source_region(
            chunk_index=(0, 0, 0),
            chunk_shape=chunk_shape,
            source_shape=(460, 1234, 2328),
            padding=padding,
        )

        print("source start:", region.start)
        print("source stop: ", region.stop)
        print("source shape:", region.shape)                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--moving', help='Enter the animal (moving)', required=True, type=str)
    parser.add_argument('--fixed', help='Enter the animal (fixed)', required=True, type=str)
    parser.add_argument("--task", help="Enter the task you want to perform", required=True, default="status", type=str)
    parser.add_argument("--downsample", help="Enter the downsample", required=False, default=1, type=int)
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)
    parser.add_argument(
        "--landmarks",
        help=(
            "JSON file containing corresponding fixed/moving physical-space "
            "landmarks. Coordinates are (x,y,z)."
        ),
        required=False,
        default=None,
        type=str,
    )

    args = parser.parse_args()
    moving_brain = args.moving
    fixed_brain = args.fixed
    task = str(args.task).strip().lower()
    downsample = args.downsample
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    pipeline = StackRegistration(
        moving_brain,
        fixed_brain,
        downsample,
        debug,
        landmark_path=args.landmarks,
    )


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
        "test_padding": pipeline.test_padding
    }

    if task in function_mapping:
        function_mapping[task]()
    else:
        print(f'{task} is not a correct task. Choose one of these:')
        for key in function_mapping.keys():
            print(f'\t{key}')
