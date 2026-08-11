import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import ball
import dask.array as da
import tifffile as tiff
import os
import zarr
from dask.diagnostics import ProgressBar


def create_brain_mask(vessel_bin, closing_radius=50):
    """
    Create a brain mask from binary vasculature.

    closing_radius: µm radius for morphological closing
    """

    # Dilate vessels
    structure = ball(closing_radius)
    #closed = ndi.binary_closig124
    # ng(vessel_bin, structure=structure)
    closed = vessel_bin.copy()
    # Fill holes
    filled = ndi.binary_fill_holes(closed)

    # Keep only largest connected component
    labels, num = ndi.label(filled)
    sizes = ndi.sum(np.ones_like(labels), labels, range(1, num+1))

    largest_label = np.argmax(sizes) + 1
    brain_mask = (labels == largest_label)

    return brain_mask

def compute_radius_map(vessel_bin):
    """
    Returns vessel radius (in microns).
    """
    dist = ndi.distance_transform_edt(vessel_bin)
    return dist.astype(np.float32)

def compute_depth_map(brain_mask):
    """
    Compute cortical depth map in microns.
    """

    # Distance from brain boundary inward
    depth = ndi.distance_transform_edt(brain_mask)

    return depth.astype(np.float32)

def label_vessels(radius_map, depth_map, vessel_bin):
    """
    Create combined vessel labels based on radius + depth.

    Output labels:
    0 = background
    1–4 = vessel radius classes
    10–40 = radius+depth combined classes
    """

    labels = np.zeros_like(radius_map, dtype=np.uint16)

    # Radius classes
    cap = radius_map < 3
    small = (radius_map >= 3) & (radius_map < 6)
    med = (radius_map >= 6) & (radius_map < 15)
    large = radius_map >= 15

    labels[cap & vessel_bin] = 1
    labels[small & vessel_bin] = 2
    labels[med & vessel_bin] = 3
    labels[large & vessel_bin] = 4

    # Depth classes
    surface = depth_map < 200
    mid = (depth_map >= 200) & (depth_map < 800)
    deep = depth_map >= 800

    # Combined classification (10s=surface, 20s=mid, 30s=deep)
    labels[(cap & surface)] = 11
    labels[(small & surface)] = 12
    labels[(med & surface)] = 13
    labels[(large & surface)] = 14

    labels[(cap & mid)] = 21
    labels[(small & mid)] = 22
    labels[(med & mid)] = 23
    labels[(large & mid)] = 24

    labels[(cap & deep)] = 31
    labels[(small & deep)] = 32
    labels[(med & deep)] = 33
    labels[(large & deep)] = 34

    return labels

def load_tiff_stack(path, chunk_shape=(64, 64, 64)):
    files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tif')])
    sample = tiff.imread(files[0])
    z = len(files)
    y, x = sample.shape
    darr = da.zeros((z, y, x), dtype=sample.dtype, chunks=chunk_shape)
    for i, f in enumerate(files):
        darr[i] = da.from_array(tiff.imread(f), chunks=chunk_shape[1:])

    return darr

def block_distance_transform(block):
    return ndi.distance_transform_edt(block)


def process_neuroglancer(zarr_path):
    vessel = da.from_zarr(zarr_path)
    print(f"Loaded vessel volume with type {type(vessel)} with shape {vessel.shape} and chunks {vessel.chunksize} dtype {vessel.dtype}")

def process_large_volume(zarr_path, out_path, chunk=(256,256,256)):

    vessel = da.from_zarr(zarr_path)
    print(f"Loaded vessel volume with type {type(vessel)} with shape {vessel.shape} and chunks {vessel.chunksize} dtype {vessel.dtype}")

    with ProgressBar():        
        radius = da.map_blocks(
            block_distance_transform,
            vessel,
            dtype=np.float32
        )

    print(f'Radius map computed with type {type(radius)} shape {radius.shape} and dtype {radius.dtype}')

    # Brain mask
    brain_mask = vessel.map_blocks(
        create_brain_mask,
        dtype=bool
    )
    print(f"Loaded brain mask with type {type(brain_mask)} with shape {brain_mask.shape} dtype {brain_mask.dtype}")

    with ProgressBar():        
        depth = da.map_blocks(
            block_distance_transform,
            brain_mask,
            dtype=np.float32
        )

    print(f'Depth map computed with shape {depth.shape} and dtype {depth.dtype}')

    # Labeling
    with ProgressBar():
        labels = da.map_blocks(
            label_vessels,
            radius, depth, vessel,
            dtype=np.uint32
        )
    print(f'Labels computed with shape {labels.shape} and chunks {labels.chunksize} dtype {labels.dtype} type {type(labels)}')
    with ProgressBar():
        labels.to_zarr(out_path, overwrite=True)
    del labels
    labels = zarr.open(out_path, mode='r')
    print(labels.info)
    numpy_array = labels[:]

    # Get the unique values
    unique_values = np.unique(numpy_array)
    print("Unique label values in the saved zarr:")
    print(unique_values)

#input_dir = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/X/preps/C1/downsampled_32"
#sample = tiff.imread(os.path.join(input_dir, os.listdir(input_dir)[0]))
#chunk_shape = (1, sample.shape[1], sample.shape[0])
#volume = load_tiff_stack(input_dir, chunk_shape=chunk_shape)
#volume = volume.rechunk('auto')

zarr_path = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/X/preps/C1/downsampled_10.zarr"

#volume.to_zarr(zarr_path, overwrite=True)
out_path = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/X/www/neuroglancer_data/vessel_labels.zarr"
if os.path.exists(out_path):
    print(f"Output path {out_path} already exists")
    labels = zarr.open(out_path, mode='r')
    print(labels.info)
    numpy_array = labels[:]

    # Get the unique values
    unique_values = np.unique(numpy_array)
    print("Unique label values in the saved zarr:")
    print(unique_values)
    process_neuroglancer(out_path)
else:
    process_large_volume(zarr_path, out_path, chunk=(64, 64, 64))