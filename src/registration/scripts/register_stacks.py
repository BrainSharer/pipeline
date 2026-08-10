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
import cloudvolume
from cloudvolume import CloudVolume
from taskqueue.taskqueue import LocalTaskQueue
import igneous.task_creation as tc



PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())


from library.image_manipulation.image_manager import ImageManager
from library.utilities.utilities_process import write_image
from library.image_manipulation.neuroglancer_manager import NumpyToNeuroglancer
from library.image_manipulation.precomputed_manager import NgPrecomputedMaker


class StackRegistration:

    def __init__(self, moving, fixed, downsample=1, debug=False):
        self.moving = moving
        self.fixed = fixed
        self.downsample = downsample
        self.base_path = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data"
        self.reg_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration"
        self.ds_moving_path = os.path.join(self.base_path, self.moving, 'preps', 'C1', f'thumbnail_aligned.{self.downsample}')
        self.ds_fixed_path = os.path.join(self.base_path, self.fixed, 'preps', 'C1', f'thumbnail_aligned.{self.downsample}')
        self.full_moving_path = os.path.join(self.base_path, self.moving, 'preps', 'C1', 'thumbnail_aligned.64')
        self.full_fixed_path = os.path.join(self.base_path, self.fixed, 'preps', 'C1', 'thumbnail_aligned.64')
        self.output_zarr = os.path.join(self.base_path, self.moving, 'preps', 'C1', f'{self.moving}_{self.fixed}_registered.zarr')
        self.registered_tif_path = os.path.join(self.base_path, self.moving, 'preps', 'C1', 'thumbnail_registered')
        self.xy_resolution = 0.325
        self.full_xy_resolution = self.xy_resolution * 64
        self.ds_xy_resolution = self.xy_resolution * self.downsample
        self.z_resolution = 20.0
                
        self.transform_path = os.path.join(self.reg_path, f"{self.moving}_{self.fixed}.tfm")
        self.full_xy_resolution /= 1000
        self.ds_xy_resolution /= 1000
        self.z_resolution /= 1000

        self.debug = debug


    def create_zarr(self):
        for brain in [self.moving, self.fixed]:
            if self.downsample > 1:
                aligned = f"thumbnail_aligned.{self.downsample}"
                aligned_zarr = f"thumbnail_aligned.{self.downsample}.zarr"
                rechunker = 8
            else:
                aligned = "full_aligned"
                aligned_zarr = "full_aligned.zarr"
                rechunker = 8
            input_path = os.path.join(self.base_path, brain, 'preps', 'C1', aligned)
            output_path = os.path.join(self.base_path, brain, 'preps', 'C1', aligned_zarr)
            if not os.path.exists(input_path):
                print(f"Input path {input_path} does not exist for brain {brain}")
                sys.exit(1)
            if os.path.exists(output_path):
                print(f"Output path {output_path} already exists")
                print(f"\tfor brain {brain}, skipping zarr creation")
                continue

            print(f'{brain} input {input_path}')
            print(f'{brain} output {output_path}')


            dask_imgs = build_dask_array_from_folder(input_path)
            rechunks_zyx = (1, dask_imgs.shape[1], dask_imgs.shape[2] // rechunker)
            dask_imgs = dask_imgs.rechunk(rechunks_zyx)
            print(f'Dask array shape: {dask_imgs.shape} chunk size = {dask_imgs.chunksize}')
            with ProgressBar():
                dask_imgs.to_zarr(output_path, overwrite=True)
            del dask_imgs
            print(f"✅ Downsampled stack saved to {output_path}")
            volume = zarr.open(output_path, mode='r')
            print(volume.info)
            del volume


    def create_test_volumes(self):
        # fixed
        input_path = os.path.join(self.base_path, self.fixed, 'preps', 'C1', f"thumbnail_aligned.{self.downsample}")
        if not os.path.exists(input_path):
            print(f"Input path {input_path} does not exist for brain {self.fixed}")
            sys.exit(1)

        slices = []
        files = sorted(os.listdir(input_path))
        for f in tqdm(files):
            inpath = os.path.join(input_path, f)
            img = tifffile.imread(inpath)
            slices.append(img)

        arr = np.stack(slices, axis=0)
        outpath = os.path.join(self.base_path, self.fixed, 'preps', f'C1', f'volume.{self.downsample}.tif')
        tifffile.imwrite(outpath, arr)
        print(f'Wrote fixed tif to {outpath}')
        # registered
        input_path = os.path.join(self.base_path, self.moving, 'preps', 'C1', f"registered.{self.downsample}")
        if not os.path.exists(input_path):
            print(f"Input path {input_path} does not exist for brain {self.moving}")
            sys.exit(1)

        slices = []
        files = sorted(os.listdir(input_path))
        for f in tqdm(files):
            inpath = os.path.join(input_path, f)
            img = tifffile.imread(inpath)
            slices.append(img)

        arr = np.stack(slices, axis=0)
        outpath = os.path.join(self.base_path, self.moving, 'preps', f'C1', f'registered.{self.downsample}.tif')
        tifffile.imwrite(outpath, arr)
        print(f'Wrote registered tif to {outpath}')


    def check_image(self, brain):
        # 1. Load your original medical image to grab original metadata
        size_info = {}
        size_info['DK52'] = (65500,35500,486)
        size_info['DK55'] = (60000,34000,485)

        moving_original_spacing_xyz = (0.325, 0.325, 20)

        # 2. Load the resized image generated by ImageMagick
        source_path = os.path.join(self.base_path, brain, 'preps', 'C1', f'thumbnail_aligned.{self.downsample}.zarr')
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



    def create_transform(self):
        if os.path.exists(self.transform_path):
            print(f"Transform file {self.transform_path} already exists, skipping transform creation")            
            return
        else:
            print(f'Creating affine registration in {self.transform_path}')

        if not os.path.exists(self.ds_moving_path):
            print(f'Exiting, missing: {self.ds_moving_path}')
            return
        if not os.path.exists(self.ds_fixed_path):
            print(f'Exiting, missing: {self.ds_fixed_path}')
            return
        if self.debug:
            print(f'Using moving data from {self.ds_moving_path}')
            print(f'Using fixed data from {self.ds_fixed_path}')
            return
        fixed_sitk = create_sitk_volume(self.ds_fixed_path)
        moving_sitk = create_sitk_volume(self.ds_moving_path)
        moving_spacing = self.check_image(self.moving)
        fixed_spacing = self.check_image(self.fixed)
        moving_sitk.SetSpacing(moving_spacing)
        fixed_sitk.SetSpacing(fixed_spacing)
        print(f'\nMoving sitk info size={moving_sitk.GetSize()=} spacing={moving_sitk.GetSpacing()=}')
        print(f'Fixed sitk info size={fixed_sitk.GetSize()=} spacing={fixed_sitk.GetSpacing()}')
        affine_transform = affine_registration(fixed_sitk, moving_sitk)
        sitk.WriteTransform(affine_transform, self.transform_path)


    def create_registered_stack(self):
            
        if os.path.exists(self.transform_path):
            print(f'Using transform: {self.transform_path}')
        else:
            print(f"Transform file {self.transform_path} does not exist, cannot create registered volume")
            return
        transform = sitk.ReadTransform(self.transform_path)
        print('path', self.transform_path)
        print(f'matrix {transform}')
        moving_zarr_path = os.path.join(self.base_path, self.moving, 'preps', 'C1', f'thumbnail_aligned.{self.downsample}.zarr')
        moving_zarr = zarr.open(moving_zarr_path, mode='r')
        if os.path.exists(moving_zarr_path):
            print(f'Using moving zarr from: {moving_zarr_path}')
        else:
            print(f'Cannot find moving zarr: {moving_zarr_path}')
            exit(0)
        moving_image = sitk.GetImageFromArray(moving_zarr)
        moving_spacing = self.check_image(self.moving)
        moving_image.SetSpacing(moving_spacing)

        fixed_zarr_path = os.path.join(self.base_path, self.fixed, 'preps', 'C1', f'thumbnail_aligned.{self.downsample}.zarr')
        if os.path.exists(fixed_zarr_path):
            print(f'Using fixed zarr from: {fixed_zarr_path}')
        else:
            print(f'Cannot find fixed zarr: {fixed_zarr_path}')
            exit(0)
        fixed_zarr = zarr.open(fixed_zarr_path, mode='r')
        fixed_image = sitk.GetImageFromArray(fixed_zarr)
        fixed_spacing = self.check_image(self.fixed)
        fixed_image.SetSpacing(fixed_spacing)

        print(f'Size of moving image {moving_image.GetSize()} spacing: {moving_image.GetSpacing()}')
        print(f'Size of fixed image {fixed_image.GetSize()} spacing: {fixed_image.GetSpacing()}')

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetTransform(transform)
        resampled = resampler.Execute(moving_image)

        match_filter = sitk.HistogramMatchingImageFilter()
        resampled = match_filter.Execute(resampled, fixed_image)

        registered_outpath = os.path.join(self.base_path, self.moving, 'preps', 'C1', f'registered.{self.downsample}')
        if os.path.exists(registered_outpath):
            shutil.rmtree(registered_outpath)
            print('Removed existing', registered_outpath)
        os.makedirs(registered_outpath, exist_ok=True)
        size_z = resampled.GetSize()[2]

        for z in tqdm(range(size_z), desc="Writing registered TIFs", disable=self.debug):
            # Extract the 2D slice at index z
            slice_2d = resampled[:, :, z]
            # Generate a unique file name for each plane
            filepath = os.path.join(registered_outpath, f"{z:03d}.tif")
            # Write the 2D slice to disk
            sitk.WriteImage(slice_2d, filepath)

    
        print(f"Resampled moving images written to {registered_outpath}")
        sitk.WriteImage(slice_2d, filepath)



    def create_neuroglancer(self):
            
        registered_inpath = os.path.join(self.base_path, self.moving, 'preps', 'C1', f'registered.{self.downsample}')
        neuroglancer_path = os.path.join(self.base_path, self.moving, 'www', 'neuroglancer_data') 
        rechunkme_path = os.path.join(neuroglancer_path, 'registered_rechunkme')
        progress_dir = os.path.join(neuroglancer_path, 'registered_progress')
        image_manager = ImageManager(registered_inpath)
        # chunk 
        chunks = [image_manager.height//4, image_manager.width//4, 1] # 1796x984

        x = self.xy_resolution * self.downsample
        y = self.xy_resolution * self.downsample
        z = 20
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
        if os.path.exists(progress_dir) and len(os.listdir(progress_dir)) > 0 and os.path.exists(registered_inpath):
            print("Transfer task has already been completed")
        else:
            ng.init_precomputed(rechunkme_path, image_manager.volume_size)
            file_keys = []
            for i, f in enumerate(image_manager.files):
                filepath = os.path.join(registered_inpath, f)
                file_keys.append([i, filepath, progress_dir, False, 0, 0]) #added is_blank, height, width

            
            for file_key in file_keys:
                ng.process_image(file_key=file_key)

            ng.precomputed_vol.cache.flush()

        base_chunks = [64, 64, 64]

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




    def volume2stack(self):
        moving_path = os.path.join(self.base_path, self.moving, 'preps', 'C1', f'registered.{self.downsample}.tif')
        if os.path.exists(moving_path):
            print('Registered volume exists:', moving_path)
        else:
            print(f'Registered volume missing: {moving_path}')
            exit(0)
        fixed_path = os.path.join(self.base_path, self.fixed, 'preps', 'C1', f'volume.{self.downsample}.tif')
        if os.path.exists(fixed_path):
            print('Fixed volume exists:', fixed_path)
        else:
            print(f'Fixed volume missing: {fixed_path}')
            exit(0)

        fixed_image = sitk.ReadImage(fixed_path)
        resampled = sitk.ReadImage(moving_path)
        match_filter = sitk.HistogramMatchingImageFilter()
        resampled = match_filter.Execute(resampled, fixed_image)

        size_z = resampled.GetSize()[2]
        moving_outpath = os.path.join(self.base_path, self.moving, 'preps', 'C1', f'registered.{self.downsample}')
        if os.path.exists(moving_outpath):
            shutil.rmtree(moving_outpath)
            print('Removed existing', moving_outpath)
        os.makedirs(moving_outpath, exist_ok=True)
        size_z = resampled.GetSize()[2]

        for z in tqdm(range(size_z), desc="Writing registered TIFs", disable=self.debug):
            slice_2d = resampled[:, :, z]
            filepath = os.path.join(moving_outpath, f"{z:03d}.tif")
            sitk.WriteImage(slice_2d, filepath)

        print(f"Resampled moving images written to {moving_outpath}")




    def create_registered_volume_by_tiles(self):

        if os.path.exists(self.output_zarr):
            shutil.rmtree(self.output_zarr)
            
        if not os.path.exists(self.transform_path):
            print(f"Transform file {self.transform_path} does not exist, cannot create registered volume")
            return
        transform = sitk.ReadTransform(self.transform_path)
        print('path', self.transform_path)
        print(f'matrix {transform}')
        source = zarr.open(os.path.join(self.base_path, self.moving, 'preps', 'C1', f'thumbnail_aligned.{self.downsample}.zarr'), mode='r')
        print(source.info)
        image = sitk.GetImageFromArray(source)
        fixed_spacing = self.check_image(self.fixed)
        moving_spacing = self.check_image(self.moving)
        image.SetSpacing(moving_spacing)
        print(f'Spacing of moving image {image.GetSpacing()}')


        target = zarr.open(
            self.output_zarr,
            mode="w",
            shape=source.shape,
            chunks=source.chunks,
            dtype=source.dtype)        

        process_volume(
            source,
            target,
            transform,
            padding_zyx=(32, 32, 32),
            spacing_zyx=moving_spacing[::-1],
            use_inverse_transform=True
        )
        
        registered_volume = zarr.open(self.output_zarr, mode='r')
        print(registered_volume.info)

    def create_tifs(self):
        if os.path.exists(self.registered_tif_path):
            shutil.rmtree(self.registered_tif_path)
        os.makedirs(self.registered_tif_path, exist_ok=True)
        if not os.path.exists(self.output_zarr):
            print(f'No zarr at {self.output_zarr}')
            return
        volume = zarr.open(self.output_zarr, mode='r')
        nz = volume.shape[0]
        slices = []

        for z in tqdm(range(nz)):
            img = volume[z]
            outpath = os.path.join(self.registered_tif_path, f"{z:03d}.tif")
            tifffile.imwrite(outpath, img)
            slices.append(img)

        arr = np.stack(slices, axis=0)
        outpath = os.path.join(self.base_path, self.moving, 'preps', 'C1', 'registered.tif')
        tifffile.imwrite(outpath, arr)
        print(f'Wrote registered tif to {outpath}')

        return
        fixed_zarr_path = os.path.join(self.base_path, self.fixed, 'preps', 'C1', 'thumbnail_aligned.zarr')

        volume = zarr.open(fixed_zarr_path, mode='r')
        nz = volume.shape[0]
        slices = []

        for z in tqdm(range(nz)):
            img = volume[z]
            slices.append(img)

        arr = np.stack(slices, axis=0)
        outpath = os.path.join(self.reg_path, self.fixed, 'testing.tif')
        tifffile.imwrite(outpath, arr)
        print(f'Wrote registered tif to {outpath}')





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
    delayed_reads = [read_tiff_delayed(p) for p in files]

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

def open_zarr(path):
    return da.from_zarr(path)

def dask_to_sitk(volume, xy_resolution, xy_downsample):
    arr = volume.compute()
    img = sitk.GetImageFromArray(arr)
    spacing = (
            xy_resolution * xy_downsample,
            xy_resolution * xy_downsample,
            20.0
        )    
    img.SetSpacing(spacing)
    print(f"Converted dask array to SimpleITK image with shape {img.GetSize()} and spacing {img.GetSpacing()}")

    return img

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


def resample_full_resolution(moving_zarr,
                             fixed_zarr,
                             transform, xy_resolution):

    resampler = sitk.ResampleImageFilter()
    full_resolution_fixed = dask_to_sitk(fixed_zarr, xy_resolution, 1)
    full_resolution_moving = dask_to_sitk(moving_zarr, xy_resolution, 1)
    resampler.SetReferenceImage(full_resolution_fixed)
    resampler.SetTransform(transform)
    resampled = resampler.Execute(full_resolution_moving)
    
    arr = sitk.GetArrayFromImage(resampled)

    return da.from_array(
        arr,
        chunks=(1,512,512)
    )



def sitk_to_dask(image):
    arr = sitk.GetArrayFromImage(image)
    return da.from_array(
        arr,
        chunks=(32,512,512)
    )

def save_zarr(volume, outfile):

    volume.to_zarr(
        outfile,
        overwrite=True,
        compressor=zarr.Blosc(
            cname='zstd',
            clevel=5,
            shuffle=2
        )
    )

def create_sitk_volume(input_path):
    files = sorted(glob.glob(os.path.join(input_path, "*.tif")))
    slices = []
    for f in tqdm(files):
        img = tifffile.imread(f)
        slices.append(img.astype(np.float32))
    arr = np.stack(slices, axis=0)
    return sitk.GetImageFromArray(arr)

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

def process_volume(
    zarr_src,
    zarr_dst,
    transform,
    *,
    padding_zyx=(0, 32, 32),
    spacing_zyx=(1.0, 1.0, 1.0),
    origin_zyx=(0.0, 0.0, 0.0),
    direction_xyz=None,
    fill_value=0,
    interpolator=None,
    use_inverse_transform=True,
    progress=True,
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
    progress : bool
        Display a progress bar if tqdm is installed.
    """

    if interpolator is None:
        import SimpleITK as sitk
        interpolator = sitk.sitkLinear

    shape = zarr_src.shape
    chunks = zarr_src.chunks

    nblocks = tuple(
        math.ceil(shape[i] / chunks[i])
        for i in range(3)
    )

    iterator = product(
        range(nblocks[0]),
        range(nblocks[1]),
        range(nblocks[2]),
    )

    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(
                iterator,
                total=nblocks[0] * nblocks[1] * nblocks[2],
                desc="Warping chunks",
            )
        except ImportError:
            pass

    for block_index in iterator:

        process_block(
            zarr_src=zarr_src,
            zarr_dst=zarr_dst,
            block_index=block_index,
            transform=transform,
            padding_zyx=padding_zyx,
            spacing_zyx=spacing_zyx,
            origin_zyx=origin_zyx,
            direction_xyz=direction_xyz,
            fill_value=fill_value,
            interpolator=interpolator,
            use_inverse_transform=use_inverse_transform,
        )

def _zyx_to_xyz(values: Sequence[float]) -> Tuple[float, float, float]:
    """Convert (z, y, x) ordering to SimpleITK's (x, y, z) ordering."""
    if len(values) != 3:
        raise ValueError(f"Expected 3 values, got {len(values)}")
    z, y, x = values
    return float(x), float(y), float(z)


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


def process_block(
    zarr_src,
    zarr_dst,
    block_index: Sequence[int],
    transform: sitk.Transform,
    *,
    padding_zyx: Sequence[int] = (0, 16, 16),
    spacing_zyx: Sequence[float] = (1.0, 1.0, 1.0),
    origin_zyx: Sequence[float] = (0.0, 0.0, 0.0),
    direction_xyz: Optional[Sequence[float]] = None,
    fill_value: float = 0.0,
    interpolator: int = sitk.sitkLinear,
    use_inverse_transform: bool = False,
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
    origin_zyx : tuple[float, float, float]
        Physical origin of voxel (0,0,0) in (z, y, x) order.
    direction_xyz : optional sequence[float]
        3x3 direction cosine matrix flattened in row-major order, in SimpleITK's
        (x, y, z) coordinate convention. If None, identity is used.
    fill_value : float
        Background value used outside the transformed source.
    interpolator : int
        SimpleITK interpolator, e.g. sitk.sitkLinear or sitk.sitkNearestNeighbor.
    use_inverse_transform : bool
        If True, uses transform.GetInverse() before resampling.

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

    core_start, core_stop, pad_start, pad_stop = _compute_block_bounds(
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

    if any(s <= 0 for s in core_size_zyx):
        raise ValueError(f"Invalid core block size: {core_size_zyx}")

    # Convert spacing/origin from z,y,x to x,y,z for SimpleITK
    spacing_xyz = _zyx_to_xyz(spacing_zyx)
    origin_xyz = _zyx_to_xyz(origin_zyx)

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

    if direction_xyz is None:
        direction_xyz = (
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        )
    moving.SetDirection(tuple(float(v) for v in direction_xyz))
    # Optionally invert the transform. This is useful when the provided transform
    # maps source->target but Resample needs output->input.
    tx = transform
    if use_inverse_transform:
        tx = transform.GetInverse()

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
    reference.SetDirection(tuple(float(v) for v in direction_xyz))
    # Resample padded moving image into the core reference geometry
    resampled = sitk.Resample(
        moving,
        reference,
        tx,
        interpolator,
        fill_value,
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
        "create_registered_volume": pipeline.create_registered_volume_by_tiles,
        "create_tifs": pipeline.create_tifs,
        "create_registered_stack": pipeline.create_registered_stack,
        "volume2stack": pipeline.volume2stack,
        "create_neuroglancer": pipeline.create_neuroglancer,
        "create_test_volumes": pipeline.create_test_volumes
    }

    if task in function_mapping:
        function_mapping[task]()
    else:
        print(f'{task} is not a correct task. Choose one of these:')
        for key in function_mapping.keys():
            print(f'\t{key}')
