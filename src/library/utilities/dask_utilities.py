"""Read nested directories of images into an n-dimensional array."""
import math
import os
from pathlib import Path
import dask.array as da
import numpy as np
import tifffile
import toolz as tz
from skimage.io import imread
from typing import List, Optional, Tuple
import zarr
from dask.delayed import delayed

@tz.curry
def _load_block(files_array, block_id=None, *,  n_leading_dim, load_func=imread):
    image = np.asarray(load_func(files_array[block_id[:n_leading_dim]]))
    return image[(np.newaxis,) * n_leading_dim]


def _find_shape(file_sequence):
    n_total = len(file_sequence)
    parents = {p.parent for p in file_sequence}
    n_parents = len(parents)
    if n_parents == 1:
        return (n_total,)
    else:
        return _find_shape(parents) + (n_total // n_parents,)


def load_stack(INPUT):
    lazyimread = delayed(imread, pure=True)  # Lazy version of imread
    files = sorted(os.listdir(INPUT))
    if len(files) == 0:
        raise ValueError(f'no files found at path {INPUT}.')
    
    len_files = len(files)
    midpoint = len_files // 2
    infile = os.path.join(INPUT, files[midpoint])
    midfile = imread(infile)

    lazy_values = []
    for file in files:
        filepath = os.path.join(INPUT, file)
        lazy_values.append(lazyimread(filepath))

    arrays = [da.from_delayed(lazy_value,           # Construct a small Dask array
                          dtype=midfile.dtype,   # for every lazy value
                          shape=midfile.shape)
          for lazy_value in lazy_values]
    
    print('Finished created dask stack')
    return da.stack(arrays, axis=0)  


def imreads(root, pattern='*.tif'):
    """Read tif images from the downsampled or full folder.
    This function is specifically for Neuroglancer that wants x,y,z
    Not z,y,z !!!!!

    Parameters
    ----------
    root : str | pathlib.Path
        The root folder containing the hierarchy of image files.
    pattern : str
        A glob pattern with zero or more levels of subdirectories. Each level
        will be counted as a dimension in the output array. Directories *must*
        be specified with a forward slash ("/").

    Returns
    -------
    stacked : dask.array.Array
        The stacked dask array. The array will have the number of dimensions of
        each image plus one per directory level.
    """
    root = Path(root)
    files = sorted(root.glob(pattern))
    if len(files) == 0:
        raise ValueError(f'no files found at path {root} with pattern {pattern}.')
    leading_shape = _find_shape(files)
    n_leading_dim = len(leading_shape)
    first_file = imread(files[0])
    ndim = first_file.ndim
    dtype = first_file.dtype
    lagging_shape = first_file.shape
    if ndim == 3:
        divisor = 1
        lagging_shape = (first_file.shape[0] // divisor, first_file.shape[1] // divisor, 3)
    lagging_shape = first_file.shape
    #print(f'leading_shape={leading_shape}, n_leading_dim={n_leading_dim}, dtype={dtype}, lagging_shape={lagging_shape}')
    files_array = np.array(list(files)).reshape(leading_shape)
    chunks = tuple((1,) * shp for shp in leading_shape) + lagging_shape
    #print(f'chunks={chunks}')
    stacked = da.map_blocks(
        _load_block(n_leading_dim=n_leading_dim, load_func=imread),
        files_array,
        chunks=chunks,
        dtype=dtype,
    )
    return stacked

    
def mean_dtype(arr, **kwargs):
    return np.mean(arr, **kwargs).astype(arr.dtype)


def get_transformations(axes, n_levels) -> tuple[dict,dict]:
    '''
    GENERATES META-INFO FOR PYRAMID

    :param axis_names:
    :param resolution:
    :param n_levels:
    :return: list[dict,dict] [{'scale': [10.4, 10.4, 20.0], 'type': 'scale'}]
    '''

    transformations = []
    #n_levels = 1
    for scale_level in range(n_levels):
        scales = []
        for axis_dict in axes:
            resolution = axis_dict['resolution']
            coarsen = axis_dict['coarsen']
            scales.append(resolution * coarsen**scale_level)

        #while (scales[1] * 0.8) > scales[0]:
        #    scales[0] *= 2
        transformations.append({"scale": scales, "type": "scale"})
    return transformations




def aligned_coarse_chunks(chunks: List[int], multiple: int) -> List[int]:
    """ Returns a new chunking aligned with the coarsening multiple"""
    
    def round_down(num, divisor):
        return num - (num%divisor)

    z = chunks[0]
    y = round_down(chunks[1], multiple)
    x = round_down(chunks[2], multiple)
    return [z, y, x]


def get_xy_chunk() -> int:
    '''
    CALCULATES OPTIMAL CHUNK SIZE FOR IMAGE STACK (TARGET IS ~25MB EACH)
    N.B. CHUNK DIMENSION ORDER (XYZ) SHOULD CORRESPOND TO DASK DIMENSION ORDER (XYZ)
    
    ref: https://forum.image.sc/t/deciding-on-optimal-chunk-size/63023/7

    :return: int: xy chunk
    '''

    z_section_chunk = 20
    byte_per_pixel = 2
    target_chunk_size_mb = 25
    xy_chunk = (target_chunk_size_mb*10**6 / byte_per_pixel / z_section_chunk)**(1/2) #1MB / BYTES PER PIXEL / kui_constant, SPLIT (SQUARE ROOT) BETWEEN LAST 2 DIMENSIONS

    return int(xy_chunk)

def get_optimum_chunks(image_shape, leading_chunk):
    z, y, x = image_shape
    if leading_chunk > z:
        leading_chunk = z

    return (leading_chunk, y, x)

def compute_optimal_chunks(shape: Tuple[int, ...],
                           dtype: np.dtype,
                           channels: int,
                           total_mem_bytes: int,
                           n_workers: int,
                           target_chunk_bytes: Optional[int] = None,
                           max_chunk_bytes: int = 256 * 1024**2,
                           xy_align: int = 256,
                           tile_boundary: Optional[int] = None,
                           prefer_z_chunks: int = 1) -> Tuple[int, ...]:
    """
    Heuristic to compute chunk sizes.

    shape: global array shape (Z, Y, X[, C])
    dtype: numpy dtype
    channels: number of channels
    total_mem_bytes: usable memory in bytes (system-level or per-worker? pass per-worker*something)
    n_workers: number of dask workers
    target_chunk_bytes: desired bytes per chunk (optional)
    max_chunk_bytes: hard cap on chunk size in bytes
    xy_align: make X,Y divisible by this (e.g., 256 for cloud)
    tile_boundary: if tif tile grid is known (e.g., 512), align XY to tile boundary
    prefer_z_chunks: number of z slices per chunk (try to keep small)
    """
    itemsize = np.dtype(dtype).itemsize
    # default target: smaller of max_chunk_bytes and per-worker_mem/4
    if target_chunk_bytes is None:
        per_worker_mem = max(256 * 1024**2, int(total_mem_bytes / max(1, n_workers)))
        t = max(16 * 1024**2, int(per_worker_mem / 4))
        target_chunk_bytes = min(t, max_chunk_bytes)

    # shape unpack
    # Handle shapes: (Z, Y, X) or (Z, Y, X, C)
    Z = shape[0]
    Y = shape[1]
    X = shape[2]
    # channels param used separately

    # Start with Z-chunk small (prefer 1 or a few slices)
    z_chunk = min(max(1, prefer_z_chunks), Z)

    # compute XY area that results in target bytes:
    # bytes_per_xy = target_chunk_bytes / (z_chunk * itemsize * channels)
    bytes_per_xy = max(1, int(target_chunk_bytes / (z_chunk * itemsize * max(1, channels))))
    # aim for square-ish XY chunk:
    approx_side = int(math.sqrt(bytes_per_xy))
    # ensure at least 16 px
    approx_side = max(16, approx_side)

    # align to xy_align and optionally tile_boundary
    def align(n, align_to):
        return int(math.ceil(n / align_to) * align_to)

    y_chunk = align(min(Y, approx_side), xy_align)
    x_chunk = align(min(X, approx_side), xy_align)

    # if aligning pushed bytes too large, reduce z_chunk or xy sizes
    def chunk_bytes(zc, yc, xc):
        return zc * yc * xc * itemsize * max(1, channels)

    # iterative adjust if chunk too big
    current_bytes = chunk_bytes(z_chunk, y_chunk, x_chunk)
    while current_bytes > target_chunk_bytes and (y_chunk > xy_align or x_chunk > xy_align or z_chunk > 1):
        # prefer shrinking XY then Z
        if y_chunk > xy_align:
            y_chunk = max(xy_align, int(y_chunk / 2))
            y_chunk = align(y_chunk, xy_align)
        elif x_chunk > xy_align:
            x_chunk = max(xy_align, int(x_chunk / 2))
            x_chunk = align(x_chunk, xy_align)
        elif z_chunk > 1:
            z_chunk = max(1, int(z_chunk / 2))
        else:
            break
        current_bytes = chunk_bytes(z_chunk, y_chunk, x_chunk)

    # ensure divisibility of X,Y by 2 (not strictly necessary)
    y_chunk = min(Y, y_chunk)
    x_chunk = min(X, x_chunk)

    # final fallback: if resulting chunk too small, increase a bit
    if current_bytes < (4 * 1024**2):  # < 4MB - inefficient
        # increase XY until reasonable or hit image size
        while current_bytes < (16 * 1024**2) and (y_chunk * 2 <= Y or x_chunk * 2 <= X):
            if y_chunk * 2 <= Y:
                y_chunk = align(min(Y, y_chunk * 2), xy_align)
            if x_chunk * 2 <= X:
                x_chunk = align(min(X, x_chunk * 2), xy_align)
            current_bytes = chunk_bytes(z_chunk, y_chunk, x_chunk)

    # Return a chunk tuple depending on channels presence
    if channels > 1:
        # shape: (Z, Y, X, C) -> keep channel axis small (full channel)
        return (z_chunk, y_chunk, x_chunk, channels)
    else:
        return (z_chunk, y_chunk, x_chunk)



def get_tiff_zarr_array(filepaths):
    with tifffile.imread(filepaths, aszarr=True) as store:
        return zarr.open(store)

def open_store(storepath, mip, mode="a"):
    try:
        return zarr.open(get_store(storepath, mip, mode=mode))
    except Exception as ex:
        print('Exception opening zarr store')
        print(ex)

def get_store(storepath, mip, mode="a"):
    return get_store_from_path(os.path.join(storepath, str(mip)), mode=mode)

def get_store_from_path(path, mode="a"):
    store = zarr.storage.NestedDirectoryStore(path)
    return store


def get_size_GB(shape,dtype):
    
    current_size = math.prod(shape)/1024**3
    if dtype == np.dtype('uint8'):
        pass
    elif dtype == np.dtype('uint16'):
        current_size *=2
    elif dtype == np.dtype('float32'):
        current_size *=4
    elif dtype == float:
        current_size *=8
    
    return current_size


def closest_divisors_to_target(number, target):
    def find_divisors(n):
        divisors = set()
        for i in range(1, int(n**0.5)+1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        return divisors

    divisors = find_divisors(number)
    closest = min(divisors, key=lambda x: abs(x - target))

    return closest
