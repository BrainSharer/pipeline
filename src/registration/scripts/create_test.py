import shutil
import dask.array as da
from dask import delayed
from dask.array.overlap import overlap, trim_internal
import scipy.ndimage

import numpy as np
import zarr
import os
import sys
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import affine_transform
import ants

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
    max_shape = process_tuple(max_shape)

    # Compute padding needed for each array
    def compute_padding(shape, target):
        return [(0, t - s) for s, t in zip(shape, target)]

    pad1 = compute_padding(shape1, max_shape)
    pad2 = compute_padding(shape2, max_shape)

    # Pad with zeros
    moving_arr_padded = da.pad(moving_arr, pad_width=pad1, mode='constant', constant_values=0)
    fixed_arr_padded = da.pad(fixed_arr, pad_width=pad2, mode='constant', constant_values=0)
    return moving_arr_padded, fixed_arr_padded

def cosine_window(shape):
    """Create a cosine window for weighting a block."""
    grids = np.meshgrid(*[np.linspace(-np.pi, np.pi, s) for s in shape], indexing='ij')
    window = np.ones(shape)
    for g in grids:
        window *= 0.5 * (1 + np.cos(g))
    return window

def apply_affine_block(moving_block, block_info=None, affine=None, window=None):
    """
    Apply affine transformation to a block.
    `block_info` gives the location of the block in the full array.
    `affine` is a global affine (4x4).
    `window` is a precomputed cosine window.
    """
    # Get block location in global coordinates
    loc = block_info[None]['array-location']
    z0, y0, x0 = loc[0][0], loc[1][0], loc[2][0]
    
    # Compute the coordinate grid in reference space
    shape = moving_block.shape
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]) + z0,
        np.arange(shape[1]) + y0,
        np.arange(shape[2]) + x0,
        indexing='ij'
    )
    
    coords = np.stack([zz, yy, xx, np.ones_like(zz)], axis=0).reshape(4, -1)
    
    # Transform coordinates using inverse affine (from output to input space)
    inv_affine = np.linalg.inv(affine)
    transformed_coords = inv_affine @ coords
    zt, yt, xt = transformed_coords[:3].reshape(3, *shape)

    # Interpolate from moving image using affine-transformed coordinates
    transformed = scipy.ndimage.map_coordinates(moving_block, [zt, yt, xt], order=1, mode='nearest')
    
    # Apply cosine window
    weighted = transformed * window
    
    return weighted

def affine_transform_dask(reference, moving, affine, block_size=(64, 64, 64), overlap=16):
    """
    Apply affine transform to moving image in the space of reference image.
    """
    # Define chunking with overlap
    depth = {0: overlap, 1: overlap, 2: overlap}
    
    # Create a cosine window for blending
    padded_block_shape = tuple(bs + 2 * overlap for bs in block_size)
    window = cosine_window(padded_block_shape)

    # Normalize weights to avoid intensity bias in overlaps
    weight_array = da.map_overlap(
        lambda x: window,
        reference,
        depth=depth,
        boundary=0,
        dtype=reference.dtype
    )

    # Apply affine transformation block-wise with overlapping and weighting
    transformed = da.map_overlap(
        apply_affine_block,
        moving,
        depth=depth,
        boundary='nearest',
        dtype=moving.dtype,
        affine=affine,
        window=window
    )

    # Stitch by normalizing with weights
    stitched = transformed / weight_array

    return stitched



# Example usage:
if __name__ == "__main__":
    reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
    moving_filepath_zarr = os.path.join(reg_path, 'ALLEN771602/ALLEN771602_32.0x28.8x28.8um_sagittal.zarr')
    fixed_filepath_zarr = os.path.join(reg_path, 'Allen/Allen_32.0x28.8x28.8um_sagittal.zarr')
    transform_filepath = os.path.join(reg_path, 'ALLEN771602/ALLEN771602_Allen_32.0x28.8x28.8um_sagittal.npy')

    if not os.path.exists(transform_filepath):
        raise FileNotFoundError(f"Transform not found at {transform_filepath}")
    
    if not os.path.exists(moving_filepath_zarr):
        raise FileNotFoundError(f"Moving volume not found at {moving_filepath_zarr}")
    
    if not os.path.exists(fixed_filepath_zarr):
        raise FileNotFoundError(f"Fixed volume not found at {fixed_filepath_zarr}")

    zarr_output_path = os.path.join(reg_path, 'ALLEN771602/registered.zarr')
    if os.path.exists(zarr_output_path):
        print(f"Removing zarr file already exists at {zarr_output_path}")
        shutil.rmtree(zarr_output_path)


    print('Loading images ...')
    moving_dask, reference_dask = pad_to_symmetric_shape(moving_filepath_zarr, fixed_filepath_zarr)
    assert moving_dask.shape == reference_dask.shape, "Source and target must have the same shape"

    print(f'Moving shape: {moving_dask.shape} type: {type(moving_dask)}')
    print(f'Ref shape: {reference_dask.shape} type: {type(reference_dask)}')


    # Register blockwise
    chunk_size = (moving_dask.shape[0] // 8, moving_dask.shape[1] // 1, moving_dask.shape[2] // 1)  # Adjust chunk size as needed
    #affine_matrix = np.load(transform_filepath)
    print(f'Chunk size: {chunk_size}')

    moving_dask = moving_dask.rechunk(chunk_size)
    reference_dask = reference_dask.rechunk(chunk_size)

    affine_matrix = np.eye(4)  # Identity matrix for testing
    affine = np.load(transform_filepath)
    R = affine[:3,:3]
    rotated = np.rot90(R, k=2, axes=(0,1))
    t = affine[:3,3]
    new_t = np.array([ t[2],t[1],t[0] ])
    new_affine = np.eye(4)
    new_affine[:3,:3] = rotated
    new_affine[:3,3] = new_t
    transformed = affine_transform_dask(reference_dask, moving_dask, new_affine, chunk_size, overlap=4)

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
