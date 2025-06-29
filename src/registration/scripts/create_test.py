import shutil

import numpy as np
import dask.array as da
import zarr
import SimpleITK as sitk
from dask.diagnostics import ProgressBar
from dask import delayed
import os

import os
import sys
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import affine_transform

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.utilities.utilities_process import write_image



def pad_to_symmetric_shape(moving_path, fixed_path):
    def process_tuple(input_tuple):
        if not all(num % 2 == 0 for num in input_tuple):
            return tuple(num + 1 for num in input_tuple)
        return input_tuple

    # Load the Zarr arrays
    moving_arr = da.from_zarr(moving_path)
    fixed_arr = da.from_zarr(fixed_path)
    print(f'Original oving shape: {moving_arr.shape}')


    # Determine target shape
    shape1 = moving_arr.shape
    shape2 = fixed_arr.shape
    max_shape = tuple(max(s1, s2) for s1, s2 in zip(shape1, shape2))
    #max_shape = process_tuple(max_shape)

    # Compute padding needed for each array
    def compute_padding(shape, target):
        return [(0, t - s) for s, t in zip(shape, target)]

    pad1 = compute_padding(shape1, max_shape)
    pad2 = compute_padding(shape2, max_shape)

    # Pad with zeros
    moving_arr_padded = da.pad(moving_arr, pad_width=pad1, mode='constant', constant_values=0)
    fixed_arr_padded = da.pad(fixed_arr, pad_width=pad2, mode='constant', constant_values=0)
    return moving_arr_padded, fixed_arr_padded

# Define the affine transform (rotation + translation)
def get_affine_transform():
    transform = sitk.Euler3DTransform()
    transform.SetRotation(np.deg2rad(10), np.deg2rad(5), np.deg2rad(15))
    transform.SetTranslation((5, -5, 10))
    return transform

#affine_transform = get_affine_transform()

# Convert numpy block to SimpleITK image, apply transform, return block
def transform_block(block, block_info=None):
    block = np.squeeze(block)
    # Get global position from block_info
    start = tuple(block_info[None]['array-location'][0])
    #global_shape = block_info[None]['array-shape']
    # Pad block to avoid edge issues
    pad_width = [(overlap, overlap)] * 3
    padded = np.pad(block, pad_width, mode='reflect')
    
    # Create SimpleITK image
    image = sitk.GetImageFromArray(padded)
    image.SetSpacing((1.0, 1.0, 1.0))
    print('start', start)
    origin = [-start[i] - overlap for i in range(3)]
    print('origin', origin)
    image.SetOrigin([-start[i] - overlap for i in range(3)])
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetTransform(affine_transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled = resampler.Execute(image)

    result = sitk.GetArrayFromImage(resampled)
    # Crop back to original block size
    cropped = result[overlap:-overlap, overlap:-overlap, overlap:-overlap]
    return np.expand_dims(cropped, 0)


# Example usage:
if __name__ == "__main__":
    reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
    moving_filepath_zarr = os.path.join(reg_path, 'ALLEN771602/ALLEN771602_32.0x28.8x28.8um_sagittal.zarr')
    fixed_filepath_zarr = os.path.join(reg_path, 'Allen/Allen_32.0x28.8x28.8um_sagittal.zarr')
    transform_filepath = os.path.join(reg_path, 'ALLEN771602/ALLEN771602_Allen_32.0x28.8x28.8um_Affine.txt')

    if not os.path.exists(transform_filepath):
        raise FileNotFoundError(f"Transform not found at {transform_filepath}")
    
    if not os.path.exists(moving_filepath_zarr):
        raise FileNotFoundError(f"Moving volume not found at {moving_filepath_zarr}")
    
    #if not os.path.exists(fixed_filepath_zarr):
    #    raise FileNotFoundError(f"Fixed volume not found at {fixed_filepath_zarr}")

    zarr_output_path = os.path.join(reg_path, 'ALLEN771602/registered.zarr')
    if os.path.exists(zarr_output_path):
        print(f"Removing zarr file already exists at {zarr_output_path}")
        shutil.rmtree(zarr_output_path)


    print('Loading images ...')
    moving_dask, reference_dask = pad_to_symmetric_shape(moving_filepath_zarr, fixed_filepath_zarr)
    assert moving_dask.shape == reference_dask.shape, "Source and target must have the same shape"

    print(f'Moving shape: {moving_dask.shape} type: {type(moving_dask)}')
    #print(f'Ref shape: {reference_dask.shape} type: {type(reference_dask)}')


    # Register blockwise
    chunk_size = (moving_dask.shape[0] // 2, moving_dask.shape[1] // 2, moving_dask.shape[2] // 2)  # Adjust chunk size as needed
    #affine_matrix = np.load(transform_filepath)
    print(f'Chunk size: {chunk_size}')

    moving_dask = moving_dask.rechunk(chunk_size)
    reference_dask = reference_dask.rechunk(chunk_size)
    ##### start code 

    overlap = 8
    # Map the transform block-wise with overlap
    transformed = moving_dask.map_overlap(
        transform_block, 
        depth=overlap, 
        boundary='reflect',
        dtype=np.float32
    )


    transformed = transformed.rechunk(chunk_size)  # Rechunk after transformation
    ##### end code

 
    # --- Step 5: Save to Zarr ---
    transformed.to_zarr(zarr_output_path, overwrite=True)

    volume = zarr.open(zarr_output_path, 'r')
    print(volume.info)
    image_stack = []
    for i in tqdm(range(int(volume.shape[0]))): # type: ignore
        section = volume[i, ...]
        if section.ndim > 2: # type: ignore
            section = section.reshape(section.shape[-2], section.shape[-1]) # type: ignore
        image_stack.append(section)

    print('Stacking images ...')
    volume = np.stack(image_stack, axis=0)


    reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
    outpath = os.path.join(reg_path, 'ALLEN771602', 'registered.tif')
    if os.path.exists(outpath):
        print(f"Removing tiff file already exists at {outpath}")
        os.remove(outpath)
    write_image(outpath, volume.astype(np.uint16))

    exit(1)

    output_tifs_path = os.path.join(reg_path, 'ALLEN771602', 'registered_slices')
    if os.path.exists(output_tifs_path):
        print(f"Removing tiff files already exists at {output_tifs_path}")
        shutil.rmtree(output_tifs_path)

    os.makedirs(output_tifs_path, exist_ok=True)

    for i in tqdm(range(volume.shape[0])):
        slice_i = volume[i, ...]
        slice_i = slice_i.astype(np.uint8)
        outpath_slice = os.path.join(output_tifs_path, f'{str(i).zfill(4)}.tif')
        write_image(outpath_slice, slice_i)
