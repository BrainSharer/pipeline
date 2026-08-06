import argparse
from email.mime import image
import os
import glob

import dask.array as da
from dask import delayed
import numpy as np
import tifffile
from tqdm.asyncio import tqdm
import zarr
import SimpleITK as sitk

CHUNK_SIZE = (16, 512, 512)



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


def downsample(volume, XY_DOWNSAMPLE):

    return volume[
        ::1,
        ::XY_DOWNSAMPLE,
        ::XY_DOWNSAMPLE
    ]

def shrink_volume(image, XY_DOWNSAMPLE):
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    # Define a scale factor (e.g., shrink by 50%)
    new_size = [original_size[0] // XY_DOWNSAMPLE, original_size[1] // XY_DOWNSAMPLE, original_size[2]]
    new_spacing = [original_spacing[0] * XY_DOWNSAMPLE, original_spacing[1] * XY_DOWNSAMPLE, original_spacing[2]]

    # Resample to new size and spacing
    shrunk_image = sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0.0,
        image.GetPixelID()
    )
    return shrunk_image

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
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)
    print(f"Affine registration with fixed image size {fixed.GetSize()} and moving image size {moving.GetSize()}")
    print(f"Fixed image spacing: {fixed.GetSpacing()}, Moving image spacing: {moving.GetSpacing()}")


    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.AffineTransform(fixed.GetDimension()), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    # ------------------------------------------------------------
    # 4. Rigid+Affine registration (MI metric)
    # ------------------------------------------------------------
    registration = sitk.ImageRegistrationMethod()

    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.2)
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
    registration.SetInitialTransform(initial_transform, inPlace=False)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()


    #registration.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration))
    affine_transform = registration.Execute(fixed, moving)

    print("Affine done. Final metric:", registration.GetMetricValue())
    print("Optimizer's stopping condition: ", registration.GetOptimizerStopConditionDescription())

    return affine_transform


def resample_full_resolution(moving_zarr,
                             fixed_zarr,
                             transform, spacing):

    moving = sitk.GetImageFromArray(moving_zarr.compute())
    fixed = sitk.GetImageFromArray(fixed_zarr.compute())

    moving.SetSpacing(spacing)
    fixed.SetSpacing(spacing)

    print(f"Resampling moving image with shape {moving.GetSize()} and spacing {moving.GetSpacing()}")
    print(f"Resampling fixed image with shape {fixed.GetSize()} and spacing {fixed.GetSpacing()}")

    result = sitk.Resample(
        moving,
        fixed,
        transform,
        sitk.sitkLinear,
        0,
        moving.GetPixelID()
    )

    #arr = sitk.GetArrayFromImage(result)
    return result

    return da.from_array(
        arr,
        chunks=CHUNK_SIZE
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--moving', help='Enter the animal (moving)', required=True, type=str)
    parser.add_argument('--fixed', help='Enter the animal (fixed)', required=True, type=str)
    parser.add_argument('--xy_resolution', help='XY resolution in um', required=True, default=0.325, type=float)
    parser.add_argument('--z_resolution', help='Z resolution in um', required=True, default=20.0, type=float)
    parser.add_argument('--xy_downsample', help='XY downsample factor', required=True, default=1.0, type=float)

    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)

    args = parser.parse_args()
    moving_brain = args.moving
    fixed_brain = args.fixed
    xy_resolution = args.xy_resolution
    z_resolution = args.z_resolution
    xy_downsample = args.xy_downsample

    base_path = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data"
    moving_path = os.path.join(base_path, moving_brain, 'preps', 'C1', 'thumbnail_aligned')
    fixed_path = os.path.join(base_path, fixed_brain, 'preps', 'C1', 'thumbnail_aligned')
    fixed_sitk = create_sitk_volume(fixed_path)
    moving_sitk = create_sitk_volume(moving_path)
    moving_sitk.SetSpacing((xy_resolution, xy_resolution, z_resolution))
    fixed_sitk.SetSpacing((xy_resolution, xy_resolution, z_resolution))
    print(f'moving sitk shape: {moving_sitk.GetSize()}, fixed sitk shape: {fixed_sitk.GetSize()}')
    print(f'moving sitk spacing: {moving_sitk.GetSpacing()}, fixed sitk spacing: {fixed_sitk.GetSpacing()}')
    #fixed_dask = build_dask_array_from_folder(fixed_path)
    #print(f"Loaded fixed volume with shape {fixed_dask.shape} and dtype {fixed_dask.dtype}")


    fixed_zarr_path = os.path.join(base_path, fixed_brain, 'preps', 'C1', 'thumbnail_aligned.zarr')
    #write_zarr(fixed_dask, fixed_zarr_path)

    #moving_dask = build_dask_array_from_folder(moving_path)
    #print(f"Loaded moving volume with shape {moving_dask.shape} and dtype {moving_dask.dtype}")
    moving_zarr_path = os.path.join(base_path, moving_brain, 'preps', 'C1', 'thumbnail_aligned.zarr')
    #write_zarr(moving_dask, moving_zarr_path)

    fixed_zarr = open_zarr(fixed_zarr_path)
    moving_zarr = open_zarr(moving_zarr_path)
    print(f'fixed zarr shape: {fixed_zarr.shape}, moving zarr shape: {moving_zarr.shape}')
    print(f'fixed zarr spacing: {fixed_sitk.GetSpacing()}, moving zarr spacing: {moving_sitk.GetSpacing()}')
    #ds_fixed = downsample(fixed_zarr, xy_downsample)
    #ds_moving = downsample(moving_zarr, xy_downsample)
    ds_fixed = shrink_volume(fixed_sitk, int(xy_downsample))
    ds_moving = shrink_volume(moving_sitk, int(xy_downsample))
    print(f"Downsampled fixed volume to shape {ds_fixed.GetSize()} and moving volume to shape {ds_moving.GetSize()}")
    print(f"Downsampled fixed volume spacing: {ds_fixed.GetSpacing()}, moving volume spacing: {ds_moving.GetSpacing()}")

    #fixed_ds_sitk = dask_to_sitk(ds_fixed, xy_resolution, xy_downsample)
    #moving_ds_sitk = dask_to_sitk(ds_moving, xy_resolution, xy_downsample)
    #print(f"Converted downsampled volumes to SimpleITK images with spacing {fixed_ds_sitk.GetSpacing()} and {moving_ds_sitk.GetSpacing()}")

    #transform = affine_registration(ds_fixed, ds_moving)
    #transform_path = os.path.join(base_path, moving_brain, 'preps', 'affine.tfm')
    transform_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/DK52/DK52_DK55_20.0x10.4x10.4um.tfm"
    transform = sitk.ReadTransform(transform_path)
    #sitk.WriteTransform(transform, transform_path)

    registered = resample_full_resolution(
        fixed_zarr,
        moving_zarr,
        transform,
        (xy_resolution, xy_resolution, z_resolution)
    )
    registered_tif_path = os.path.join(base_path, moving_brain, 'preps', 'C1', 'registered.tif')
    registered = sitk.Cast(registered, sitk.sitkUInt16)
    print(f'Registered moving volume to fixed volume to {registered_tif_path}')
    sitk.WriteImage(registered, registered_tif_path)
    exit(0)

    #registered_dask = sitk_to_dask(registered)
    registered_zarr_path = os.path.join(base_path, moving_brain, 'preps', 'C1', 'thumbnail_aligned_registered.zarr')

    save_zarr(registered, registered_zarr_path)

    registered_tif_path = os.path.join(base_path, moving_brain, 'preps', 'C1', 'thumbnail_aligned_registered_tiffs')

    save_tiffs(registered, registered_tif_path)
    fixed_tifs = os.path.join(base_path, fixed_brain, 'preps', 'C1', 'thumbnail_aligned.ds')
    os.makedirs(fixed_tifs, exist_ok=True)
    save_tiffs(fixed_zarr, fixed_tifs)


