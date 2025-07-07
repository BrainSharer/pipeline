import os
import shutil
import sys
from tqdm import tqdm
import numpy as np
import dask.array as da
import zarr
import SimpleITK as sitk
from dask.diagnostics import ProgressBar

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


def create_sphere(shape):
    """Create a simple 3D volume with a sphere in the center."""
    Z, Y, X = np.indices(shape)
    center = np.array(shape) // 2
    radius = min(shape) // 4
    volume = ((X - center[2])**2 + (Y - center[1])**2 + (Z - center[0])**2) < radius**2
    return volume.astype(np.float32)

def create_rectangle(shape):
    """Create a simple 3D volume with a rectangle in the center."""
    volume = np.zeros(shape, dtype=np.float32)
    z_start, y_start, x_start = shape[0] // 4, shape[1] // 4, shape[2] // 4
    z_end, y_end, x_end = z_start + shape[0] // 2, y_start + shape[1] // 2, x_start + shape[2] // 2
    volume[z_start:z_end, y_start:y_end, x_start:x_end] = 1.0
    return volume


def get_affine_transform():
    """Returns a SimpleITK AffineTransform with slight translation and scaling."""
    tx = sitk.AffineTransform(3)
    tx.Scale((1.01, 0.95, 1.1))
    tx.Translate((0, 0, 10))  # translate in z,y,x
    return tx

def apply_affine_to_block(block, transform):
    """Apply affine transform to a chunk (block)."""

    image = sitk.GetImageFromArray(block)
    image.SetSpacing([1.0, 1.0, 1.0])
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetOutputSpacing(image.GetSpacing())
    resampler.SetSize(image.GetSize())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampled = resampler.Execute(image)
    return sitk.GetArrayFromImage(resampled)

def weighted_blend(chunks, overlap):
    """Blend overlapping chunks with linear weighting."""
    shape = chunks.shape[1:]
    print(f"Blending chunks with shape {shape} and overlap {overlap}")
    weight = np.ones(shape, dtype=np.float32)

    # Create weight mask with smooth fade near the borders (in each direction)
    for axis in range(3):
        axis_len = shape[axis]
        ramp = np.ones(axis_len, dtype=np.float32)
        fade = np.linspace(0, 1, overlap, endpoint=False)
        ramp[:overlap] = fade
        ramp[-overlap:] = fade[::-1]
        weight = weight * np.expand_dims(ramp, tuple(i for i in range(3) if i != axis))

    # Weight chunks and accumulate
    weighted_chunks = chunks * weight[None, ...]
    sum_weights = np.zeros_like(chunks[0])
    blended = np.zeros_like(chunks[0])
    for chunk in weighted_chunks:
        blended += chunk
        sum_weights += weight

    return blended / (sum_weights + 1e-8)

# ------------------------------
# CONFIG
#volume_shape = (375, 232, 640)

# Generate and store synthetic volume
#volume = create_rectangle(volume_shape)
reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/ALLEN771602'
moving_file_path = os.path.join(reg_path, 'ALLEN771602_32.0x28.8x28.8um_sagittal.zarr')
transform_file_path = os.path.join(reg_path, 'ALLEN771602_Allen_32.0x28.8x28.8um_Affine.txt')
volume = da.from_zarr(moving_file_path) # type: ignore

print(f"Loaded moving volume from {moving_file_path} with shape {volume.shape} and dtype {volume.dtype}")


chunk_shape = (volume.shape[0]//1, volume.shape[1]//1, volume.shape[2]//1)  # Full chunks
# keep overlap small
overlap = 2

# Convert to Dask array with overlap
#dask_vol = da.from_array(volume, chunks=chunk_shape)
dask_vol = volume.rechunk(chunk_shape)  # type: ignore

# Affine transform (SimpleITK)
#transform = get_affine_transform()
transform, translation = read_ants_affine_transform(transform_file_path)
print(f"Loaded affine transform with translation {translation} from {transform_file_path}")
# Pad for overlap, apply transformation, unpad
transformed = da.map_overlap(
    lambda blk: apply_affine_to_block(blk, transform),
    dask_vol,
    depth=overlap,
    boundary='nearest',
    dtype=np.float32
)

print(f"Transformed shape: {transformed.shape}, dtype: {transformed.dtype}")

# Use weighted averaging at overlaps
# To do this manually, we rechunk into non-overlapping blocks and blend
# This is not directly supported in Dask, so we compute manually
"""
chunks = transformed.to_delayed().flatten()
results = []
for chunk in chunks:
    arr = da.from_delayed(chunk, shape=chunk_shape, dtype=np.float32)
    arr = weighted_blend(arr, overlap)
    results.append(arr)
"""

# Manually average overlapping blocks (simulate blending)
# NOTE: In production, you'd use a more efficient blend mechanism
final = da.overlap.overlap(transformed, depth=overlap, boundary='reflect')
final = final.map_blocks(lambda blk: blk, dtype=np.float32)  # identity, just to force compute

# Create output Save to Zarr

output_zarr_path = os.path.join(reg_path, 'registered.zarr')
if os.path.exists(output_zarr_path):
    print(f"Removing existing Zarr file at {output_zarr_path}")
    shutil.rmtree(output_zarr_path)

with ProgressBar():
    final.to_zarr(output_zarr_path, overwrite=True)

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
