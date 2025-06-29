import shutil
import dask.array as da
import numpy as np
import zarr
import SimpleITK as sitk
from dask.array.overlap import trim_internal, map_overlap
from dask import delayed
import os
import sys
from tqdm import tqdm
from pathlib import Path
PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())
from library.utilities.utilities_process import write_image


def read_ants_affine_transform(transform_filepath):
    """
    Reads an ANTs .mat affine transform file and returns a 4x4 numpy affine matrix.
    This correctly sets up the affine transformation for simpleITK!!!!!!!!!

    Args:
        file_path (str): Path to the .mat file.

    Returns:
        np.ndarray: A 4x4 affine transformation matrix.
    """
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



def apply_affine_to_chunk(chunk, transform):
    """Apply affine transformation using SimpleITK."""
    img = sitk.GetImageFromArray(chunk)
    center = np.array(img.GetSize()) / 2.0
    #transform.SetCenter(center.tolist())

    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(img.GetSpacing())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetSize(img.GetSize())
    resampler.SetOutputDirection(img.GetDirection())

    transformed = resampler.Execute(img)
    return sitk.GetArrayFromImage(transformed).astype(np.float32)


def weighted_blend_mean(chunks):
    """Blend overlapping chunks using mean with weights."""
    total = np.zeros_like(chunks[0], dtype=np.float32)
    weights = np.zeros_like(chunks[0], dtype=np.float32)
    for chunk in chunks:
        valid_mask = (chunk != 0)
        total += chunk * valid_mask
        weights += valid_mask
    return total / np.maximum(weights, 1)


# Create input volume and save to Zarr
# CONFIG
reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/ALLEN771602'
output_zarr_path = os.path.join(reg_path, 'registered.zarr')
if os.path.exists(output_zarr_path):
    print(f"Removing existing Zarr file at {output_zarr_path}")
    shutil.rmtree(output_zarr_path)
moving_file_path = os.path.join(reg_path, 'ALLEN771602_32.0x28.8x28.8um_sagittal.zarr')
transform_file_path = os.path.join(reg_path, 'ALLEN771602_Allen_32.0x28.8x28.8um_Affine.txt')
volume = da.from_zarr(moving_file_path)
transform, translation = read_ants_affine_transform(transform_file_path)
print(f"Loaded affine with translation {translation}")
print(f"Loaded moving volume from {moving_file_path}")
chunk_shape = (volume.shape[0]//1, volume.shape[1]//1, volume.shape[2]//8)  # Full chunks
# Convert to Dask array with overlap
darr = volume.rechunk(chunk_shape) 
print(f"Chunk shape: {chunk_shape}, Dask volume shape: {darr.shape}, dtype: {darr.dtype}")
##### done setting up volume and config


# Delayed wrapper for affine application
def transform_block(block):
    return delayed(apply_affine_to_chunk)(block, transform)

# Apply overlap + affine transform + weighted blending
#z=0,y=0,x=80, # large seams and poor alignment
# x overlap = 16, still big seams
# x_overlap = 32  # smaller than 16
z_overlap = 8
y_overlap = 8
x_overlap = 64
depth = {0:z_overlap,1:y_overlap,2:x_overlap}  # overlap in each dimension
overlapped = da.overlap.overlap(darr, depth, boundary='reflect')

# Apply affine to each overlapped chunk
transformed = overlapped.map_blocks(lambda block: apply_affine_to_chunk(block, transform),
                                    dtype=np.float32)

# Trim and blend overlapping blocks
blended = trim_internal(transformed, depth, boundary='reflect')

# Store the result
blended.to_zarr(output_zarr_path, overwrite=True)


volume = zarr.open(output_zarr_path, 'r')
print(volume.info)
image_stack = []
for i in tqdm(range(int(volume.shape[0]))): # type: ignore
    section = volume[i, ...]
    if section.ndim > 2: # type: ignore
        section = section.reshape(section.shape[-2], section.shape[-1]) # type: ignore
    image_stack.append(section)

print('Stacking images ...')
volume = np.stack(image_stack, axis=0)

outpath = os.path.join(reg_path, 'registered.tif')
if os.path.exists(outpath):
    print(f"Removing tif file at {outpath}")
    os.remove(outpath)
write_image(outpath, volume)
print(f'Wrote transformed volume to {outpath}')

