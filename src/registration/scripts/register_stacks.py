from __future__ import annotations

import argparse
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
from library.utilities.dask_utilities import closest_divisors_to_target


class StackRegistration:

    def __init__(self, moving, fixed, downsample=1, debug=False):
        self.moving = moving
        self.fixed = fixed
        self.downsample = downsample
        scratch_dir = "/data/pipeline_tmp"
        self.base_path = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data"
        self.reg_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration"
        self.moving_tif_path = os.path.join(self.base_path, self.moving, 'preps', 'C1', f'source_aligned.{self.downsample}')
        self.fixed_tif_path = os.path.join(self.base_path, self.fixed, 'preps', 'C1', f'source_aligned.{self.downsample}')
        self.registered_zarr_path = os.path.join(scratch_dir, self.moving, f'{self.moving}_{self.fixed}_registered.{self.downsample}.zarr')
        self.moving_zarr_path = os.path.join(scratch_dir, self.moving, f'source.{self.downsample}.zarr')
        self.fixed_zarr_path = os.path.join(scratch_dir, self.fixed, f'source.{self.downsample}.zarr')
        self.registered_tif_path = os.path.join(scratch_dir, self.moving, f'registered.{self.downsample}')
        self.xy_resolution = 0.325
        self.z_resolution = 20.0                
        self.transform_path = os.path.join(self.reg_path, f"{self.moving}_{self.fixed}.tfm")
        self.debug = debug


    def create_zarr(self):
        input_path = os.path.join(self.base_path, self.moving, 'preps', 'C1', f"source_aligned.{self.downsample}")
        if not os.path.exists(input_path):
            print(f"Input path {input_path} does not exist for brain {self.moving}")
            sys.exit(1)
        divisors = {}
        divisors[1] = 32
        divisors[8] = 8
        divisors[16] = 4
        divisors[32] = 4
        try:
            divisor = divisors[self.downsample]
        except KeyError:
            divisor = 2
        image_manager = ImageManager(input_path)
        #chunk_x = closest_divisors_to_target(image_manager.width, image_manager.width // divisor)
        #chunk_y = closest_divisors_to_target(image_manager.height, image_manager.height // divisor)
        if os.path.exists(self.moving_zarr_path):
            print(f"Output path {self.moving_zarr_path} already exists")
            print(f"\tfor brain {self.moving_zarr_path}, skipping zarr creation")
            return

        print(f'{self.moving} input {input_path}')
        print(f'{self.moving} output {self.moving_zarr_path}')


        dask_imgs = StackRegistration.build_dask_array_from_folder(input_path)
        rechunks_zyx = (1, image_manager.height, image_manager.width//divisor)
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
        if self.downsample > 32:
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
        moving_spacing = self.create_spacing(self.moving)
        fixed_spacing = self.create_spacing(self.fixed)
        moving_sitk.SetSpacing(moving_spacing)
        fixed_sitk.SetSpacing(fixed_spacing)
        print(f'\nMoving sitk info size={moving_sitk.GetSize()=} spacing={moving_sitk.GetSpacing()=}')
        print(f'Fixed sitk info size={fixed_sitk.GetSize()=} spacing={fixed_sitk.GetSpacing()}')
        affine_transform = StackRegistration.affine_registration(fixed_sitk, moving_sitk)
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
        fixed_spacing = (self.xy_resolution*self.downsample, self.xy_resolution*self.downsample, self.z_resolution)
        print(f'Spacing fixed image {fixed_spacing}')
        paddings = {}
        paddings[32] = (32, 0, 128)
        paddings[16] = (32, 0, 256)
        paddings[32] = (32, 0, 512)
        paddings[4] = (32, 0, 1024)
        #paddings[1] = (256, 256, 256)
        #paddings = (32,0,256)
        # 2 took 10 seconds, looks very chopped
        # 8 took 53 seconds, looks chopped
        # 16 took 93 seconds, just a little chopped
        # 24 took 139 seconds, looks good only when chunks = shape
        # 24,24,128 create registered tiles took 479.26 seconds

        try:
            padding = paddings[self.downsample]
        except KeyError:
            padding = (64,64,64)


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
            spacing_zyx=fixed_spacing[::-1],
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

        for z in tqdm(range(nz)):
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

        x = self.xy_resolution * self.downsample
        y = self.xy_resolution * self.downsample
        z = self.z_resolution
        scales = (x,y,z)
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
            for i, f in enumerate(image_manager.files):
                filepath = os.path.join(self.registered_tif_path, f)
                ng.process_image(file_key=[i, filepath, progress_dir, False, 0, 0])

            ng.precomputed_vol.cache.flush()

        base_chunks = [64, 64, 16]

        scales, resolutions, chunks = NgPrecomputedMaker.compute_mipmaps((x,y,z), base_chunks)
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
        x = self.xy_resolution * self.downsample
        y = self.xy_resolution * self.downsample
        z = self.z_resolution
        scales = (x,y,z)
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
    def affine_registration(fixed, moving):
        def command_iteration(method):
            print(f"Iteration: {method.GetOptimizerIteration()} ")
            print(f"Metric Value: {method.GetMetricValue()}")
            print("-" * 20)

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
    def create_sitk_volume(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.tif")))
        slices = []
        for f in tqdm(files):
            img = tifffile.imread(f)
            slices.append(img.astype(np.float32))
        arr = np.stack(slices, axis=0)
        return sitk.GetImageFromArray(arr)

    @staticmethod
    def save_tiffs(volume, directory):
        os.makedirs(directory, exist_ok=True)
        nz = volume.shape[0]
        slices = []

        for z in tqdm(range(nz)):
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


    def create_spacing(self, brain):
        # 1. Load your original medical image to grab original metadata
        size_info = {}
        size_info['DK52'] = (65500,35500,486)
        size_info['DK55'] = (60000,34000,485)

        moving_original_spacing_xyz = (self.xy_resolution, self.xy_resolution, self.z_resolution)

        # 2. Load the resized image generated by ImageMagick
        source_path = os.path.join(self.base_path, brain, 'preps', 'C1', f'source_aligned.{self.downsample}.zarr')
        if not os.path.exists(source_path):
            print(f'zarr does not exist: {source_path}')
            exit(0)
        source = zarr.open(source_path, mode='r')
        resized_img = sitk.GetImageFromArray(source)
        new_size = (resized_img.GetSize())

        # 3. Calculate new physical spacing
        new_spacing = [
            moving_original_spacing_xyz[i] * (size_info[brain][i] / new_size[i]) 
            for i in range(len(moving_original_spacing_xyz))
        ]


        return new_spacing        

    def get_zarr_info(self):
        brain_paths = [self.moving_zarr_path, self.fixed_zarr_path, self.registered_zarr_path]
        for brain_path in brain_paths:
            if os.path.exists(brain_path):
                zarr_data = zarr.open(brain_path, mode='r')
                print(brain_path)
                print(zarr_data.info)
            else:
                print(f'Missing: {brain_path}')

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--moving', help='Enter the animal (moving)', required=True, type=str)
    parser.add_argument('--fixed', help='Enter the animal (fixed)', required=True, type=str)
    parser.add_argument("--task", help="Enter the task you want to perform", required=True, default="status", type=str)
    parser.add_argument("--downsample", help="Enter the downsample", required=False, default=1, type=int)
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)

    args = parser.parse_args()
    moving_brain = args.moving
    fixed_brain = args.fixed
    task = str(args.task).strip().lower()
    downsample = args.downsample
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    pipeline = StackRegistration(moving_brain, fixed_brain, downsample, debug)


    function_mapping = {
        "create_zarr": pipeline.create_zarr,
        "create_transform": pipeline.create_transform,
        "create_registered_tiles": pipeline.create_registered_tiles,
        "zarr2tif": pipeline.create_tifs,
        "create_neuroglancer": pipeline.create_neuroglancer,
        "run_tiles": pipeline.run_tiles,
        "get_info": pipeline.get_zarr_info,
        "test_mips": pipeline.test_mips
    }

    if task in function_mapping:
        function_mapping[task]()
    else:
        print(f'{task} is not a correct task. Choose one of these:')
        for key in function_mapping.keys():
            print(f'\t{key}')
