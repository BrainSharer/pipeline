import shutil
import dask.array as da
import numpy as np
import zarr
import SimpleITK as sitk
from dask.array.overlap import trim_internal, map_overlap
from dask import delayed
import os
import sys
import cv2
from tqdm import tqdm
from pathlib import Path
PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())
from library.utilities.utilities_process import write_image



def make_weight(shape, depth):
    """Creates an N-D weight array that tapers at the edges."""
    ndim = len(shape)
    weight = 1
    for i in range(ndim):
        axis_len = shape[i]
        taper = np.ones(axis_len)
        d = depth[i] if isinstance(depth, (list, tuple)) else depth

        if d > 0:
            ramp = np.linspace(0, 1, d, endpoint=False)
            taper[:d] = ramp
            taper[-d:] = ramp[::-1]
        weight = weight * taper.reshape([-1 if j == i else 1 for j in range(ndim)])
    return weight

def weighted_chunk_function(block):
    weight = make_weight(block.shape, OVERLAP)
    weighted = block * weight
    # Now trim off the depth since the overlap is not needed anymore
    slices = tuple(slice(d, -d if d != 0 else None) for d in OVERLAP)
    return weighted[slices]

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



def apply_affine_to_chunk(chunk, block_info=None, transform=None):
    """Apply affine transformation using SimpleITK."""
    img = sitk.GetImageFromArray(chunk)
    """
    center = np.array(img.GetSize()) / 2.0
    transform.SetCenter(center.tolist())
    matrix = transform.GetMatrix()
    matrix = np.array(matrix).reshape(3, 3)
    translation = transform.GetTranslation()
    new_translation = translation + center - matrix @ center
    print(translation)
    print(new_translation)
    transform.SetTranslation(new_translation.tolist())
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(img.GetSpacing())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetSize(img.GetSize())
    resampler.SetOutputDirection(img.GetDirection())

    transformed = resampler.Execute(img)

    # Crop the transformed result back to original block shape
    z, y, x = offset
    dz, dy, dx = chunk.shape
    return transformed[z:z+dz, y:y+dy, x:x+dx]



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

# Delayed wrapper for affine application
def transform_block(block):
    return delayed(apply_affine_to_chunk)(block, transform)

def gaussian_weight(shape, sigma=0.5):
    grids = np.meshgrid(
        *[np.linspace(-1, 1, s) for s in shape],
        indexing='ij'
    )
    distance_squared = sum(g**2 for g in grids)
    weights = np.exp(-distance_squared / (2 * sigma**2))
    return weights.astype(np.float32)

def transform_and_weight(chunk, transform):
    shape = chunk.shape
    weights = gaussian_weight(shape)
    # Apply the affine transformation
    #transformed_chunk = affine_transform(chunk, matrix, order=1, mode='constant', cval=0.0)
    transformed_chunk = apply_affine_to_chunk(chunk, transform)    
    # Multiply by weight map
    weighted_chunk = transformed_chunk * weights
    return weighted_chunk, weights

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
chunk_shape = (volume.shape[0]//1, volume.shape[1]//4, volume.shape[2]//4)  # Full chunks
# Convert to Dask array with overlap
volume = volume.rechunk(chunk_shape) 
print(f"Chunk shape: {chunk_shape}, Dask volume shape: {volume.shape}, dtype: {volume.dtype} chunksize: {volume.chunksize}")

##### done setting up volume and config
def apply_transform_block(chunk, transform):
    weighted_chunk, weight = transform_and_weight(chunk, transform)
    return np.stack([weighted_chunk, weight], axis=0)
# Apply overlap + affine transform + weighted blending
#z=0,y=0,x=80, # large seams and poor alignment
# 16 overlap still has big seams
x_overlap = 28
y_overlap = 4
OVERLAP = (0, y_overlap, x_overlap)
depth = {0:OVERLAP[0],1:OVERLAP[1],2:OVERLAP[2]}  # overlap in each dimension
overlapped = da.overlap.overlap(volume, depth, boundary='nearest')

# Apply affine to each overlapped chunk
#transformed = da.map_blocks(apply_transform_block(block, transform), chunks=((2,) + volume.chunksize),
#                                    dtype=np.float32)

transformed = overlapped.map_blocks(
    apply_affine_to_chunk,
    transform=transform,
    dtype=np.float32,
)
result = da.overlap.trim_internal(transformed, depth, boundary='nearest')
print(f"Result shape: {volume.shape}, dtype: {volume.dtype}")
del volume
#result.to_zarr(output_zarr_path, overwrite=True, compute=True)
#volume = zarr.open(output_zarr_path, 'r')
#print(volume.info)
volume = result.compute()
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

