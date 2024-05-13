"""Read nested directories of images into an n-dimensional array."""
import math
import os
from pathlib import Path
import dask.array as da
import numpy as np
import psutil
import tifffile
import toolz as tz
from skimage.io import imread
from typing import List
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




def get_tiff_zarr_array(filepaths):
    with tifffile.imread(filepaths, aszarr=True) as store:
        return zarr.open(store)

def open_store(storepath, res, mode="a"):
    try:
        return zarr.open(get_store(storepath, res, mode=mode))
    except Exception as ex:
        print('Exception opening zarr store')
        print(ex)

def get_store(storepath, res, mode="a"):
    return get_store_from_path(os.path.join(storepath, f'scale{res}'), mode=mode)

def get_store_from_path(path, mode="a"):
    store = zarr.storage.NestedDirectoryStore(path)
    return store


# def optimize_chunk_shape_3d_2(image_shape, origional_chunks, output_chunks, dtype, chunk_limit_GB):


def optimize_chunk_shape(image_shape, existing_chunks, output_chunks):
    '''
    original chunks is actually chunks size from test image
    Grows chunks shape by axis y and x by the original chunk shape until a
    defined size in GB is reached

    return tuple of new chunk shape
    '''
    dtype = np.uint16
    mem = int((psutil.virtual_memory().free/1024**3)*.8)
    cpu_cores = os.cpu_count()
    chunk_limit_GB = mem / cpu_cores / 8

    print(f'existing_chunks={existing_chunks}')

    y = existing_chunks[1] if existing_chunks[1] > output_chunks[1] else output_chunks[1]
    x = existing_chunks[2] if existing_chunks[2] > output_chunks[2] else output_chunks[2]

    existing_chunks = (existing_chunks[0], y, x)
    
    current_chunks = existing_chunks
    current_size = get_size_GB(current_chunks, dtype)

    print('current_chunks', current_chunks)
    print('current_size', current_size)

    if current_size > chunk_limit_GB:
        return current_chunks

    idx = 0
    chunk_bigger_than_z = True if current_chunks[0] >= image_shape[0] else False
    chunk_bigger_than_y = True if current_chunks[1] >= image_shape[1] else False
    chunk_bigger_than_x = True if current_chunks[2] >= image_shape[2] else False

    while current_size <= chunk_limit_GB:

        # last_size = get_size_GB(current_chunks,dtype)
        last_shape = current_chunks

        # chunk_iter_idx = idx % 2
        # if chunk_iter_idx == 0 and chunk_bigger_than_y == False:
        #     current_chunks = (existing_chunks[0], current_chunks[1] + output_chunks[1], current_chunks[2])
        # elif chunk_iter_idx == 1 and chunk_bigger_than_x == False:
        #     current_chunks = (existing_chunks[0], current_chunks[1], current_chunks[2] + output_chunks[2])

        # Iterate over y first then x
        if chunk_bigger_than_y == False:
            current_chunks = (existing_chunks[0],current_chunks[1]+output_chunks[1],current_chunks[2])
        elif chunk_bigger_than_x == False:
            current_chunks = (existing_chunks[0],current_chunks[1],current_chunks[2]+output_chunks[2])
        elif chunk_bigger_than_z == False:
            current_chunks = (existing_chunks[0] + output_chunks[0], current_chunks[1], current_chunks[2])

        current_size = get_size_GB(current_chunks, dtype)

        #print(f'current_chunks={current_chunks}')
        #print(f'current_size={current_size}')

        if current_size > chunk_limit_GB:
            return last_shape

        if current_chunks[0] > image_shape[0]:
            chunk_bigger_than_z = True

        if current_chunks[1] > image_shape[1]:
            chunk_bigger_than_y = True

        if current_chunks[2] > image_shape[2]:
            chunk_bigger_than_x = True

        if all([chunk_bigger_than_z, chunk_bigger_than_y, chunk_bigger_than_x]):
            return last_shape

        idx += 1


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

def organize_by_groups(a_list, group_len):
    """
    Organizes a list into groups of a specified length.

    Args:
        a_list (list): The list to be organized into groups.
        group_len (int): The length of each group.

    Returns:
        list: A new list containing groups of elements from the original list.

    Example:
        >>> a_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> group_len = 3
        >>> organize_by_groups(a_list, group_len)
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
    """
    new = []
    working = []
    idx = 0
    for aa in a_list:
        working.append(aa)
        idx += 1

        if idx == group_len:
            new.append(working)
            idx = 0
            working = []

    if working != []:
        new.append(working)
    return new
