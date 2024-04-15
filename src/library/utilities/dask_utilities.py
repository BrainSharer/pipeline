"""Read nested directories of images into an n-dimensional array."""
import math
import os
from pathlib import Path
import dask.array as da
import numpy as np
import tifffile
import toolz as tz
from skimage.io import imread
from typing import List
#import dask_image
import zarr
from dask.delayed import delayed
import dask.array as da
from distributed import Client, progress, performance_report

from library.omezarr.tiff_manager import tiff_manager_3d

#from library.utilities.tiff_manager import tiff_manager_3d

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
    """Read images from root (heh) folder.

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
    dtype = first_file.dtype
    lagging_shape = first_file.shape
    files_array = np.array(list(files)).reshape(leading_shape)
    chunks = tuple((1,) * shp for shp in leading_shape) + lagging_shape
    stacked = da.map_blocks(
        _load_block(n_leading_dim=n_leading_dim, load_func=imread),
        files_array,
        chunks=chunks,
        dtype=dtype,
    )
    #stacked = np.swapaxes(stacked, 0, 2)
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
    for scale_level in range(n_levels):
        scales = []
        for axis_dict in axes:
            resolution = axis_dict['resolution']
            coarsen = axis_dict['coarsen'] 
            scales.append(resolution * coarsen**scale_level)
        transformations.append([{"scale": scales, "type": "scale"}])
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

    
def write_mip_series(INPUT, store):
    '''
    Make downsampled versions of dataset based on pyramidMap
    Requies that a dask.distribuited client be passed for parallel processing
    '''
    with Client(n_workers=8,threads_per_worker=1) as client:
        write_first_mip(INPUT, store, client)

def get_tiff_zarr_array(filepaths):
    with tifffile.imread(filepaths, aszarr=True) as store:
        return zarr.open(store)

def open_store(res, mode="a"):
    try:
        return zarr.open(get_store(res, mode=mode))
    except Exception as ex:
        print('Exception opening zarr store')
        print(ex)

def get_store(storepath, res, mode="a"):
    return get_store_from_path(os.path.join(storepath, f'scale{res}'), mode=mode)

def get_store_from_path(path, mode="a"):
    store = zarr.storage.NestedDirectoryStore(path)
    return store


def write_first_mip(INPUT, storepath, client):

    print('Building Virtual Stack')
    filepaths = []
    files = sorted(os.listdir(INPUT))
    for file in files:
        filepath = os.path.join(INPUT, file)
        filepaths.append(filepath)

    s = get_tiff_zarr_array(filepaths)



    # s = [test_image.clone_manager_new_file_list(x) for x in filepaths]
    print(f'Length of file list is {len(s)}')
    #s = [da.from_array(x,chunks=x.chunks,name=False,asarray=False) for x in s]
    s = [da.from_array(x) for x in s]
    #s = da.concatenate(s)
    tiff_stack = da.stack(s)
    #stack.append(s)
    #stack = da.stack(s)
    #stack = stack[None,...]
    old_shape = tiff_stack.shape
    trimto = 8
    new_shape = aligned_coarse_chunks(old_shape, trimto)
    tiff_stack = tiff_stack[:, 0:new_shape[1], 0:new_shape[2]]
    tiff_stack = tiff_stack.rechunk('auto')




    print(f'stack shape  {tiff_stack.shape} type(tiff_stack)={type(tiff_stack)} chunks={tiff_stack.chunksize}')
    chunks = [64,64,64]
    store = get_store(storepath, 0)
    z = zarr.zeros(tiff_stack.shape, chunks=chunks, store=store, overwrite=True, dtype=tiff_stack.dtype)

    # print(client.run(lambda: os.environ["HDF5_USE_FILE_LOCKING"]))
    to_store = da.store(tiff_stack, z, lock=False, compute=False)
    to_store = client.compute(to_store)
    to_store = client.gather(to_store)
    client.shutdown()
