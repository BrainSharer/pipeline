import argparse
import os
import glob

import dask.array as da
from dask import delayed
import numpy as np
import tifffile
import zarr
import SimpleITK as sitk


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
    z = len(files)

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

    stacked = da.stack(da_slices, axis=0)  # shape (Z, Y, X) or (Z, Y, X, C)

    metadata = dict(shape=stacked.shape, dtype=str(dtype), channels=channels)
    return stacked, metadata


def read_tiff_stack(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.tif")))
    arrays = [
        da.from_delayed(
            da.delayed(tifffile.imread)(f),
            shape=tifffile.imread(files[0]).shape,
            dtype=np.uint16,
        )
        for f in files
    ]

    volume = da.stack(arrays, axis=0)

    return volume

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

def read_zarr(path):
    return da.from_zarr(path)

def dask_to_sitk(volume, xy_um, z_um):
    arr = volume.compute()
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((xy_um, xy_um, z_um))

    return img

def affine_registration(fixed, moving):
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    registration = sitk.ImageRegistrationMethod()

    registration.SetMetricAsMattesMutualInformation(50)

    registration.SetInterpolator(sitk.sitkLinear)

    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=300,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )

    registration.SetOptimizerScalesFromPhysicalShift()

    registration.SetShrinkFactorsPerLevel([8,4,2,1])

    registration.SetSmoothingSigmasPerLevel([4,2,1,0])

    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration.SetInitialTransform(transform)

    final_transform = registration.Execute(
        fixed,
        moving,
    )

    return final_transform

def resample_volume(fixed, moving, transform):

    return sitk.Resample(
        moving,
        fixed,
        transform,
        sitk.sitkLinear,
        0,
        moving.GetPixelID(),
    )


def sitk_to_dask(image):

    arr = sitk.GetArrayFromImage(image)

    return da.from_array(
        arr,
        chunks=(32,512,512)
    )

def write_tiffs(volume, outdir):

    os.makedirs(outdir, exist_ok=True)

    arr = volume.compute()

    for z in range(arr.shape[0]):

        tifffile.imwrite(
            os.path.join(outdir, f"{z:03d}.tif"),
            arr[z],
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--moving', help='Enter the animal (moving)', required=True, type=str)
    parser.add_argument('--fixed', help='Enter the animal (fixed)', required=True, type=str)

    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)

    args = parser.parse_args()
    moving_brain = args.moving
    fixed_brain = args.fixed

    base_path = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data"
    moving_path = os.path.join(base_path, moving_brain, 'preps', 'C1', 'thumbnail_aligned')
    fixed_path = os.path.join(base_path, fixed_brain, 'preps', 'C1', 'thumbnail_aligned')
    fixed, _ = build_dask_array_from_folder(fixed_path)
    print(f"Loaded fixed volume with shape {fixed.shape} and dtype {fixed.dtype}")


    fixed_zarr_path = os.path.join(base_path, fixed_brain, 'preps', 'C1', 'thumbnail_aligned.zarr')
    write_zarr(fixed, fixed_zarr_path)

    moving, _ = build_dask_array_from_folder(moving_path)
    print(f"Loaded moving volume with shape {moving.shape} and dtype {moving.dtype}")
    moving_zarr_path = os.path.join(base_path, moving_brain, 'preps', 'C1', 'thumbnail_aligned.zarr')
    write_zarr(moving, moving_zarr_path)

    fixed = read_zarr(fixed_zarr_path)
    moving = read_zarr(moving_zarr_path)

    fixed_img = dask_to_sitk(fixed, 10.4, 20.0)

    moving_img = dask_to_sitk(moving, 10.4, 20.0)

    transform = affine_registration(
        fixed_img,
        moving_img,
    )

    transform_path = os.path.join(base_path, moving_brain, 'preps', 'affine.tfm')

    sitk.WriteTransform(transform, transform_path)

    registered = resample_volume(
        fixed_img,
        moving_img,
        transform,
    )

    registered_dask = sitk_to_dask(registered)
    registered_zarr_path = os.path.join(base_path, moving_brain, 'preps', 'C1', 'thumbnail_aligned_registered.zarr')

    registered_dask.to_zarr(
        registered_zarr_path,
        overwrite=True,
    )

    registered_tif_path = os.path.join(base_path, moving_brain, 'preps', 'C1', 'thumbnail_aligned_registered_tiffs')

    write_tiffs(
        registered_dask,
        registered_tif_path,
    )


