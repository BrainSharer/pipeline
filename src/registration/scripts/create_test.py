import shutil
import dask.array as da
from dask import delayed
from dask.array.overlap import overlap, trim_internal

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


def should_skip_block(block, threshold=1e-3):
    if block.shape[0] == 0 or block.shape[1] == 0 or block.shape[2] == 0:
        print(f'Skipping empty block with shape {block.shape}')
        return True
    return np.mean(block) < threshold

def perform_ants_affine(moving_block_np, reference_block_np):
    # Convert to ANTs images
    moving_img = ants.from_numpy(moving_block_np)
    reference_img = ants.from_numpy(reference_block_np)
    transform_filepath = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/ALLEN771602/ALLEN771602_Allen_32.0x28.8x28.8um_Affine.mat'

    warped_img = ants.apply_transforms(fixed=reference_img, moving=moving_img, 
                                        transformlist=transform_filepath, defaultvalue=0,
                                        interpolator='linear')
    warped_img = warped_img.numpy()
    print(f'Warped image shape: {warped_img.shape} moving shape: {moving_block_np.shape} reference shape: {reference_block_np.shape}')
    return warped_img


# Helper to apply affine transform to each block
def process_block(z0, y0, x0, output_shape, moving, affine_matrix, chunk_size):
    z1 = min(z0 + chunk_size[0], output_shape[0])
    y1 = min(y0 + chunk_size[1], output_shape[1])
    x1 = min(x0 + chunk_size[2], output_shape[2])

    block_shape = (z1 - z0, y1 - y0, x1 - x0)
    # Inverse affine to map output to input
    inv_affine = np.linalg.inv(affine_matrix)
    # Apply affine_transform to corresponding region of moving
    transformed_block = affine_transform(
        input=moving,
        matrix=inv_affine[:3, :3],
        offset=inv_affine[:3, 3],
        output_shape=block_shape,
        mode='constant',
        cval=0.0,
        prefilter=True
    )[z0:z1, y0:y1, x0:x1]  # slicing after transform to ensure proper bounds
    
    #transformed_block = moving[z0:z1, y0:y1, x0:x1]  # For testing without affine transform
    # Create weights for blending
    w = np.ones_like(transformed_block, dtype=np.float32)
    print(f'Processing block at ({z0}, {y0}, {x0})')

    return transformed_block, w, (z0, y0, x0)


def affine_transform_large_3d(
    ref: da.core.Array,
    moving: da.core.Array,
    affine_matrix: np.ndarray,
    output_zarr_path: str,
    chunk_size: tuple,
    overlap_size: int,
    order: int = 1,
):
    """
    Applies affine transformation to a large 3D moving image in chunks using dask and zarr.

    Args:
        ref_zarr_path: Path to reference Zarr dataset (used for shape and alignment).
        moving_zarr_path: Path to moving Zarr dataset to be transformed.
        affine_matrix: 4x4 affine transformation matrix.
        output_zarr_path: Path to store the output Zarr dataset.
        chunk_size: Size of chunks to use for processing (z, y, x).
        overlap_size: Number of overlapping voxels for blending.
        order: Interpolation order for affine_transform (0=nearest, 1=linear, etc).
    """

    # Prepare output array shape and chunking
    output_shape = ref.shape
    output_chunks = chunk_size
    transformed = da.zeros(output_shape, dtype=ref.dtype, chunks=output_chunks)
    weights = da.zeros(output_shape, dtype=np.float32, chunks=output_chunks)

    """
    for z in range(0, shape[0], chunk_size[0] - overlap):
        for y in range(0, shape[1], chunk_size[1] + 23):
            for x in range(0, shape[2], chunk_size[2] -15):
    """

    # Create a grid of block coordinates
    z_blocks = range(0, output_shape[0], chunk_size[0] - overlap_size)
    y_blocks = range(0, output_shape[1], chunk_size[1] - overlap_size)
    x_blocks = range(0, output_shape[2], chunk_size[2] - overlap_size)
    print(f'z_blocks: {list(z_blocks)}')
    print(f'y_blocks: {list(y_blocks)}')
    print(f'x_blocks: {list(x_blocks)}')


    # Collect delayed tasks
    for z0 in z_blocks:
        for y0 in y_blocks:
            for x0 in x_blocks:
                transformed_block, w_block, (z0, y0, x0) = process_block(z0, y0, x0, output_shape, moving, affine_matrix, chunk_size)
                z1, y1, x1 = z0 + transformed_block.shape[0], y0 + transformed_block.shape[1], x0 + transformed_block.shape[2]
                transformed[z0:z1, y0:y1, x0:x1] += transformed_block
                weights[z0:z1, y0:y1, x0:x1] += w_block

    # Normalize by weights to resolve overlaps
    with np.errstate(divide='ignore', invalid='ignore'):
        output = da.where(weights > 0, transformed / weights, 0)

    # Save to zarr
    output.to_zarr(output_zarr_path, overwrite=True)


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
    chunk_size = (moving_dask.shape[0] // 2, moving_dask.shape[1] // 2, moving_dask.shape[2] // 2)  # Adjust chunk size as needed
    #affine_matrix = np.load(transform_filepath)
    print(f'Chunk size: {chunk_size}')

    affine_matrix = np.eye(4)  # Identity matrix for testing


    affine_transform_large_3d(
        ref = reference_dask,
        moving = moving_dask,
        affine_matrix=affine_matrix,
        output_zarr_path=zarr_output_path,
        chunk_size=chunk_size,
        overlap_size=0
    )




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
