"""Read nested directories of images into an n-dimensional array."""
from pathlib import Path
import dask.array as da
import numpy as np
import toolz as tz
from skimage.io import imread

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
    shape = first_file.shape
    lagging_shape = shape
    files_array = np.array(list(files)).reshape(leading_shape)
    chunks = tuple((1,) * shp for shp in leading_shape) + lagging_shape
    stacked = da.map_blocks(
            _load_block(n_leading_dim=n_leading_dim, load_func=imread),
            files_array,
            chunks=chunks,
            dtype=dtype,
            )
    stacked = np.swapaxes(stacked, 0,2)
    return stacked
