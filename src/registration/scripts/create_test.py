import shutil
import dask.array as da
import numpy as np
from scipy.ndimage import affine_transform
import zarr
import os
import sys
from skimage.transform import matrix_transform
from tqdm import tqdm
from pathlib import Path
PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.image_manipulation.filelocation_manager import FileLocationManager
from library.utilities.utilities_process import read_image, write_image


def apply_affine_to_chunk(chunk, affine_matrix, output_shape, offset):
    """
    Apply affine transform to a numpy chunk.
    """
    return affine_transform(chunk, 
                            matrix=affine_matrix[:3, :3], 
                            offset=affine_matrix[:3, 3],
                            output_shape=output_shape,
                            order=1,  # linear interpolation
                            mode='constant',
                            cval=0)

def transform_dask_volume(moving, affine_matrix, reference_shape):
    """
    Applies an affine transformation to a dask array.
    """
    def transform_block(block, block_info=None):
        # Get block location
        if block_info is not None:
            chunk_location = block_info[None]['array-location'][0]
            print(f'Processing block at location: {chunk_location}')
            #offset = [s[0] for s in chunk_location]
            offset = chunk_location
            print(f'Offset for block: {offset}')
        else:
            offset = [0, 0, 0]
        return apply_affine_to_chunk(block, affine_matrix, block.shape, offset)

    transformed = moving.map_blocks(transform_block, dtype=moving.dtype)
    transformed = transformed.rechunk(reference_shape)  # match the reference
    return transformed

def weighted_blend(vol1, vol2):
    """
    Weighted blending of two volumes.
    """
    mask1 = vol1 != 0
    mask2 = vol2 != 0

    overlap = mask1 & mask2
    only1 = mask1 & ~mask2
    only2 = mask2 & ~mask1

    result = da.zeros_like(vol1)

    result = result + only1 * vol1
    result = result + only2 * vol2
    result = result + overlap * ((vol1 + vol2) / 2)

    return result

def stitch_volumes(moving, reference, affine_matrix, output_path, chunks):
    """
    Main function to load, transform, and stitch volumes.
    """
    print("Loading volumes...")

    print("Transforming moving volume...")
    transformed_moving = transform_dask_volume(moving, affine_matrix, reference.shape)

    print("Blending volumes...")
    stitched = weighted_blend(reference, transformed_moving)

    print("Saving to Zarr...")
    if os.path.exists(output_path):
        import shutil
        shutil.rmtree(output_path)
    
    stitched.to_zarr(output_path, overwrite=True)

    volume = zarr.open(output_path, 'r')
    print(volume.info)
    image_stack = []
    for i in tqdm(range(int(volume.shape[0]))): # type: ignore
        section = volume[i, ...]
        if section.ndim > 2: # type: ignore
            section = section.reshape(section.shape[-2], section.shape[-1]) # type: ignore
        image_stack.append(section)

    print('Stacking images ...')
    volume = np.stack(image_stack, axis=0)


    outpath = os.path.join('/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/ALLEN771602/registered.tif')
    if os.path.exists(outpath):
        print(f"Removing tiff file already exists at {outpath}")
        os.remove(outpath)
    write_image(outpath, volume)




    print("Done.")

def pad_to_symmetric_shape(moving_path, fixed_path):
    # Load the Zarr arrays
    moving_arr = da.from_zarr(moving_path)
    fixed_arr = da.from_zarr(fixed_path)

    # Determine target shape
    shape1 = moving_arr.shape
    shape2 = fixed_arr.shape
    max_shape = tuple(max(s1, s2) for s1, s2 in zip(shape1, shape2))

    # Compute padding needed for each array
    def compute_padding(shape, target):
        return [(0, t - s) for s, t in zip(shape, target)]

    pad1 = compute_padding(shape1, max_shape)
    pad2 = compute_padding(shape2, max_shape)

    # Pad with zeros
    moving_arr_padded = da.pad(moving_arr, pad_width=pad1, mode='constant', constant_values=0)
    fixed_arr_padded = da.pad(fixed_arr, pad_width=pad2, mode='constant', constant_values=0)
    return moving_arr_padded, fixed_arr_padded


def apply_affine_blockwise(
    zarr_input_path,
    zarr_output_path,
    affine_matrix,
    chunk_size=(64, 64, 64),
    overlap=(16, 16, 16),
    dtype=np.float32
):
    # Open the input zarr as a dask array
    store = zarr.DirectoryStore(zarr_input_path)
    zarr_array = zarr.open(store, mode='r')
    dask_array = da.from_zarr(zarr_array)

    shape = dask_array.shape

    # Calculate padded shape
    padded_shape = tuple(s + 2 * o for s, o in zip(shape, overlap))

    # Create output zarr and weight volumes
    out_store = zarr.DirectoryStore(zarr_output_path)
    output_zarr = zarr.open(
        out_store, mode='w', shape=shape, chunks=chunk_size, dtype=dtype
    )
    weight_zarr = zarr.open(
        zarr_output_path + "_weights", mode='w', shape=shape, chunks=chunk_size, dtype=dtype
    )

    output = da.zeros(shape, chunks=chunk_size, dtype=dtype)
    weights = da.zeros(shape, chunks=chunk_size, dtype=dtype)

    # Define function to apply affine on each chunk
    def transform_chunk(block, block_info=None):
        loc = block_info[None]['array-location']
        z0, z1 = loc[0]
        y0, y1 = loc[1]
        x0, x1 = loc[2]

        # Get global indices with padding
        z0p = max(z0 - overlap[0], 0)
        y0p = max(y0 - overlap[1], 0)
        x0p = max(x0 - overlap[2], 0)

        z1p = min(z1 + overlap[0], shape[0])
        y1p = min(y1 + overlap[1], shape[1])
        x1p = min(x1 + overlap[2], shape[2])

        # Extract padded block
        padded_block = zarr_array[z0p:z1p, y0p:y1p, x0p:x1p]

        # Define inverse affine to apply to padded block
        inv_affine = np.linalg.inv(affine_matrix)

        # Define output shape (same as original block)
        out_shape = (z1 - z0, y1 - y0, x1 - x0)

        # Apply affine transform
        transformed = affine_transform(
            padded_block,
            inv_affine[:3, :3],
            offset=inv_affine[:3, 3],
            output_shape=out_shape,
            order=1,
            mode='nearest'
        )

        return transformed

    # Apply the transformation
    transformed = dask_array.map_blocks(
        transform_chunk,
        dtype=dtype
    )

    # Create weights for averaging (binary mask of ones for each transformed chunk)
    weight_mask = dask_array.map_blocks(lambda x: np.ones_like(x, dtype=dtype), dtype=dtype)

    # Accumulate output and weight arrays
    output += transformed
    weights += weight_mask

    # Compute weighted average and write to Zarr
    final = output / da.maximum(weights, 1e-5)  # prevent divide by zero

    da.to_zarr(final, out_store, overwrite=True)

    print("Affine transformation complete and written to:", zarr_output_path)

# Example usage:
if __name__ == "__main__":
    # Example 4x4 affine matrix
    affine_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/ALLEN771602/ALLEN771602_Allen_32.0x28.8x28.8um_sagittal.npy'
    affine = np.load(affine_path)
    reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
    moving_filepath_zarr = os.path.join(reg_path, 'ALLEN771602/ALLEN771602_32.0x28.8x28.8um_sagittal.zarr')

    if not os.path.exists(moving_filepath_zarr):
        raise FileNotFoundError(f"Moving volume not found at {moving_filepath_zarr}")

    zarr_output_path = os.path.join(reg_path, 'ALLEN771602/registered.zarr')
    if os.path.exists(zarr_output_path):
        print(f"Removing zarr file already exists at {zarr_output_path}")
        shutil.rmtree(zarr_output_path)

    apply_affine_blockwise(
        zarr_input_path=moving_filepath_zarr,
        zarr_output_path= zarr_output_path,
        affine_matrix=affine,
        chunk_size=(64, 64, 64),
        overlap=(2, 2, 2)
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


    outpath = os.path.join('/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/ALLEN771602/registered.tif')
    if os.path.exists(outpath):
        print(f"Removing tiff file already exists at {outpath}")
        os.remove(outpath)
    write_image(outpath, volume)
