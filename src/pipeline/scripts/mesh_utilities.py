import zarr
import dask.array as da
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt
from dask.array import map_overlap


# -----------------------------
# CONFIG
# -----------------------------

RADIUS_BINS = [1, 3, 8, 12, 20, 35, 35, 50, 100, 1e9]  # µm
DEPTH_BINS = [100, 200, 400, 600, 800, 1200, 2500, 1e9]  # µm

def create_brain_mask(binary_dask, closing_radius=2):
    """
    Create a brain mask from binary vasculature.

    closing_radius: µm radius for morphological closing
    """

    closed = ndi.gaussian_filter(binary_dask.astype(np.float32), sigma=closing_radius)
    # Fill holes
    filled = ndi.binary_fill_holes(closed)

    # Keep only largest connected component
    labels, num = ndi.label(filled)
    sizes = ndi.sum(np.ones_like(labels), labels, range(1, num+1))

    try:
        largest_label = np.argmax(sizes) + 1
    except Exception as e:
        print(f'Error finding largest component: {e}')
        largest_label = 1
    
    brain_mask = (labels == largest_label)

    return brain_mask


def optimal_chunk_size(shape):
    z, y, x = shape
    return (1, y, x)


# -----------------------------
# DISTANCE TRANSFORM (chunk-safe)
# -----------------------------

def local_radius_transform(chunk):
    """Local EDT-based radius estimate."""
    return ndi.distance_transform_edt(chunk)


def label_chunk(binary_chunk, radius_chunk):
    """
    Labels vessels based on radius + depth
    """

    #radius_um = radius_chunk * res_um
    #depth_um = depth_chunk * res_um
    #radius_chunk *= res_um
    #depth_chunk *= res_um

    labels = np.zeros(binary_chunk.shape, dtype=np.uint8)

    #rcls = np.digitize(radius_um, RADIUS_BINS)
    #dcls = np.digitize(depth_um, DEPTH_BINS)
    #labels[binary_chunk > 0] = rcls[binary_chunk > 0] + 10 * dcls[binary_chunk > 0]
    
    """
    # Capillaries
    labels[(binary_chunk > 0) & (radius_chunk < 4)] = 1
    # Small arterioles
    labels[(binary_chunk > 0) & (radius_chunk >= 4) & (radius_chunk < 8) & (depth_chunk < 600)] = 2
    # Large arterioles
    labels[(binary_chunk > 0) & (radius_chunk >= 8) & (radius_chunk < 15) & (depth_chunk < 1200)] = 3
    # Arteries
    labels[(binary_chunk > 0) & (radius_chunk >= 15)] = 4
    # Veins (deep + large)
    labels[(binary_chunk > 0) & (radius_chunk > 10) & (depth_chunk > 1200)] = 5    
    """
    # Capillaries
    labels[(binary_chunk > 0) & (radius_chunk < 4)] = 1
    # Small arterioles
    labels[(binary_chunk > 0) & (radius_chunk >= 4) & (radius_chunk < 8)] = 2
    # Large arterioles
    labels[(binary_chunk > 0) & (radius_chunk >= 8) & (radius_chunk < 15)] = 3
    # Arteries
    labels[(binary_chunk > 0) & (radius_chunk >= 15)] = 4
    # Veins (deep + large)
    labels[(binary_chunk > 0) & (radius_chunk > 10)] = 5    


    return labels


# -----------------------------
# PIPELINE
# -----------------------------
def smoother(block):
    return ndi.binary_fill_holes(block)

def get_distance_transform(block):
    filled = ndi.binary_fill_holes(block)
    return ndi.distance_transform_edt(filled)


def label_vessels_zarr(binary_zarr_path, resolution_um):

    # Load zarr lazily
    zin = zarr.open(binary_zarr_path, mode='r')
    binary = da.from_zarr(zin)
    chunks = optimal_chunk_size(binary.shape)

    # Force chunking
    binary = binary.rechunk(chunks)

    print(f"Computing global depth transform and binary chunksize = {binary.chunksize}")

    # Compute brain mask
    #brain_mask = binary > 0
    #brain_mask = create_brain_mask(binary, closing_radius=2)


    # Depth = distance from surface inward
    #depth = da.map_blocks(ndi.distance_transform_edt,brain_mask,dtype=np.float32)
    print("Computing local radius transform...")
    binary = da.map_blocks(
        smoother,
        binary,
        dtype=np.float32
    )

    radius = da.map_blocks(
        local_radius_transform,
        binary,
        dtype=np.float32
    )



    print("Computing final labels...")

    labeled = da.map_blocks(
        label_chunk,
        binary,
        radius,
        dtype=np.uint32
    )
    return labeled

