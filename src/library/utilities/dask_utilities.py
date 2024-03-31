"""Read nested directories of images into an n-dimensional array."""
from pathlib import Path
import dask.array as da
import numpy as np
import toolz as tz
from skimage.io import imread
from typing import List

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

    def choose_new_size(multiple, q, left):
        """ 
        See if multiple * q is a good choice when 'left' elements are remaining.
        Else return multiple * (q-1)
        """
        possible = multiple * q
        if (left - possible) > 0:
            return possible
        else:
            return multiple * (q - 1)

    # print(chunks)
    # print(sum(chunks))
    newchunks = []
    left = sum(chunks) - sum(newchunks)
    chunkgen = (c for c in chunks)
    while left > 0:
        if left < multiple:
            newchunks.append(left)
            break

        chunk_size = next(chunkgen, 0)
        if chunk_size == 0:
            chunk_size = multiple

        q, r = divmod(chunk_size, multiple)
        # print(c0, left, q, r)
        if q == 0:
            continue
        elif r == 0:
            newchunks.append(chunk_size)
        elif r >= 5:
            newchunks.append(choose_new_size(multiple, q + 1, left))
        else:
            newchunks.append(choose_new_size(multiple, q, left))

        left = sum(chunks) - sum(newchunks)
        # print(newchunks, left)

    print(f"{chunks} → {newchunks}")

    # checks
    assert sum(chunks) == sum(newchunks)
    if sum(chunks) % multiple == 0:
        lastind = None
    else:
        lastind = -1
    assert all(c % multiple == 0 for c in newchunks[slice(lastind)])

    return tuple(newchunks)