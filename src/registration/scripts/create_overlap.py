import shutil
import dask.array as da
import numpy as np
import zarr
import SimpleITK as sitk
from dask.array.overlap import trim_internal, map_overlap
from dask import delayed
from dask.diagnostics import ProgressBar

import os
import sys
from tqdm import tqdm
from pathlib import Path
PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())
from library.utilities.utilities_process import write_image

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


def read_ants_affine_transform():
    """
    Reads an ANTs .mat affine transform file and returns a 4x4 numpy affine matrix.
    This correctly sets up the affine transformation for simpleITK!!!!!!!!!

    Args:
        file_path (str): Path to the .mat file.

    Returns:
        np.ndarray: A 4x4 affine transformation matrix.
    """
    reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
    transform_filepath = os.path.join(reg_path,'ALLEN771602', 'ALLEN771602_Allen_32.0x28.8x28.8um_Affine.txt')

    with open(transform_filepath, 'r') as f:
        lines = f.readlines()

    # Parse parameters
    params_line = next(line for line in lines if line.startswith("Parameters:"))
    fixed_params_line = next(line for line in lines if line.startswith("FixedParameters:"))

    params = list(map(float, params_line.strip().split()[1:]))
    fixed_params = list(map(float, fixed_params_line.strip().split()[1:]))
    matrix = np.array(params[:9]).reshape(3, 3)
    translation = np.array(params[9:12])
    center = np.array(fixed_params)

    # Build the affine transform with rotation center
    affine = np.eye(4)
    affine[:3, :3] = matrix
    affine[:3, 3] = translation + center - matrix @ center

    rotation_matrix = np.rot90(matrix, k=2, axes=(1, 0))  # Adjust axes if needed
    t = affine[:3, 3]
    translation = np.array([t[2], t[1], t[0]])  # Adjust order if needed
    #translation_vector = [0,0,47]  # Assuming no translation for now, can be modified as needed
    #print(f'2 translation:\t{translation}')

    # Create an AffineTransform
    transform = sitk.AffineTransform(3)  # 3D transformation
    transform.SetMatrix(rotation_matrix.flatten())
    #translation = [0,0,0]
    transform.SetTranslation(translation)

    return transform, translation


def apply_affine_to_chunk(moving_chunk, block_info=None, transform=None):
    #print("block info:", block_info)
    original_shape = moving_chunk.shape
    if original_shape[0] == 0 or original_shape[1] == 0 or original_shape[2] == 0:
        return
        return np.zeros(original_shape, dtype=np.float32)
    block = np.squeeze(moving_chunk)
    # Pad block to avoid edge issues
    image = sitk.GetImageFromArray(block.astype(np.float32))
    image.SetSpacing((1.0, 1.0, 1.0))
    #image.SetOrigin([-origin[i] - overlap for i in range(3)])
    image.SetOrigin((0,0,0))
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled = resampler.Execute(image)
    result = sitk.GetArrayFromImage(resampled)
    #print(f"Result shape after resampling: {result.shape}")
    # Crop back to original size
    weights = gaussian_weight(result.shape, sigma=0.5) # this makes the seams smoother
    return result * weights  # Apply weights to blend overlapping blocks

def gaussian_weight(shape, sigma=0.5):
    grids = np.meshgrid(
        *[np.linspace(-1, 1, s) for s in shape],
        indexing='ij'
    )
    distance_squared = sum(g**2 for g in grids)
    weights = np.exp(-distance_squared / (2 * sigma**2))
    return weights.astype(np.float32)

def blend_blocks(data, depth):
    """Weighting function for blending overlapping blocks."""
    weights = np.ones_like(data, dtype=np.float32)

    for axis, d in enumerate(depth):
        if d == 0:
            continue
        length = data.shape[axis]
        ramp = np.ones(length)
        ramp[:d] = np.linspace(0, 1, d)
        ramp[-d:] = np.linspace(1, 0, d)
        shape = [1] * 3
        shape[axis] = length
        weights *= ramp.reshape(shape)

    return weights


def sitk_affine_block(moving_chunk):
    # Convert to SimpleITK image
    original_shape = moving_chunk.shape
    if original_shape[0] == 0 or original_shape[1] == 0 or original_shape[2] == 0:
        return
    transform, translation = read_ants_affine_transform()
    block = np.squeeze(moving_chunk)
    # Pad block to avoid edge issues
    image = sitk.GetImageFromArray(block.astype(np.float32))
    #reference_image = sitk.GetImageFromArray(fixed_chunk.astype(np.float32))
    image.SetSpacing((1.0, 1.0, 1.0))
    #image.SetOrigin([-origin[i] - overlap for i in range(3)])
    image.SetOrigin((0,0,0))
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled = resampler.Execute(image)
    result = sitk.GetArrayFromImage(resampled)
    return result

def register_block(fixed_block, moving_block, depth):
    """Apply registration and blending to a block."""
    registered = sitk_affine_block(moving_block)
    weights = blend_blocks(moving_block, depth)
    return registered * weights, weights

# Create input volume and save to Zarr
# CONFIG
reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
output_zarr_path = os.path.join(reg_path, 'ALLEN771602', 'registered.zarr')
if os.path.exists(output_zarr_path):
    print(f"Removing existing Zarr file at {output_zarr_path}")
    shutil.rmtree(output_zarr_path)
moving_filepath_zarr = os.path.join(reg_path,'ALLEN771602', 'ALLEN771602_32.0x28.8x28.8um_sagittal.zarr')
fixed_filepath_zarr = os.path.join(reg_path,'Allen', 'Allen_32.0x28.8x28.8um_sagittal.zarr')
transform_file_path = os.path.join(reg_path,'ALLEN771602', 'ALLEN771602_Allen_32.0x28.8x28.8um_Affine.txt')
#moving_dask = da.from_zarr(moving_filepath_zarr)
#fixed_dask = da.from_zarr(fixed_filepath_zarr)
moving_dask, fixed_dask = pad_to_symmetric_shape(moving_filepath_zarr, fixed_filepath_zarr)

assert moving_dask.shape == fixed_dask.shape, "Source and target must have the same shape"

#transform, translation = read_ants_affine_transform(transform_file_path)
#print(f"Loaded affine with translation {translation}")
chunk_shape = (moving_dask.shape[0]//1, moving_dask.shape[1]//1, moving_dask.shape[2]//1)  # Full chunks
# Convert to Dask array with overlap
moving_dask = moving_dask.rechunk(chunk_shape) 
fixed_dask = fixed_dask.rechunk(chunk_shape)  # type: ignore
print(f"Moving Dask shape: {moving_dask.shape}, chunks: {moving_dask.chunksize}")
print(f"Fixed Dask shape: {fixed_dask.shape}, chunks: {fixed_dask.chunksize}")
##### done setting up volume and config
boundary = 'nearest'
depth = {0:0, 1:0, 2:0}  # overlap in each dimension
overlapped0 = da.overlap.overlap(moving_dask, depth, boundary=boundary)
overlapped1 = da.overlap.overlap(fixed_dask, depth, boundary=boundary)
print(f"Overlapped moving shape: {overlapped0.shape}, chunksize: {overlapped0.chunksize} chunks: {overlapped0.chunks}")
print(f"Overlapped fixed shape: {overlapped1.shape}, chunksize: {overlapped1.chunksize}")

stacked = da.map_blocks(
        lambda m, f: np.stack([m, f]),
        moving_dask,
        fixed_dask,
        dtype=np.float32,
        chunks=((2,) + chunk_shape)
)

print(f'stacked shape: {stacked.shape} chunksize: {stacked.chunksize}')
"""
exit(1)
transformed = overlapped.map_blocks(
    apply_affine_to_chunk,
    transform=transform,
    dtype=np.float32,
)
"""
def wrapped_register(moving_block, fixed_block):
        registered, weight = register_block(fixed_block, moving_block, depth)
        return np.stack([registered, weight])

transformed = map_overlap(
    lambda x: wrapped_register(x[0], x[1]),
    stacked,
    depth=depth,
    boundary="reflect",
    trim=False,
    dtype=moving_dask.dtype,
)
trimmed = da.overlap.trim_internal(transformed, depth, boundary=boundary)
print(f"Trimmed map blocks shape: {trimmed.shape}, chunksize: {trimmed.chunksize}")

with ProgressBar():
    trimmed.to_zarr(output_zarr_path, overwrite=True, compute=True)

volume = zarr.open(output_zarr_path, 'r')
print(volume.info)

print(f"Result shape: {volume.shape}, dtype: {volume.dtype}")

image_stack = []
for i in tqdm(range(int(volume.shape[0]))): # type: ignore
    section = volume[i, ...]
    if section.ndim > 2: # type: ignore
        section = section.reshape(section.shape[-2], section.shape[-1]) # type: ignore
    image_stack.append(section)

print('Stacking images ...')
volume = np.stack(image_stack, axis=0)

outpath = os.path.join(reg_path, 'ALLEN771602', 'registered.tif')
if os.path.exists(outpath):
    print(f"Removing tif file at {outpath}")
    os.remove(outpath)
write_image(outpath, volume)
print(f'Wrote transformed volume to {outpath}')
