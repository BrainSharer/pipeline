# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 21:11:13 2022

@author: awatson
"""

import zarr
import os
import numpy as np
import tifffile
import skimage
import io
import math

# import imagecodecs
from copy import deepcopy

"""
Tiff managers can take a tiff file (tiff_manager) or list of files (tiff_manager_3d)
and present them as sliceable arrays.  

'desired_chunk_depth parameters' allow non-chunked tiff files to appear to be 
chunked by a desireable number of lines for interacting with downstream libraries
like dask array.

Files are only opened during __init__ and when data is requested via slicing

Cloning options are available to maintain the properties of a previously instantiated
manager but with a different file.  This allows for rapid creation of 1000s of 
managers without having to access the file system during __init__ to integrigate
each file.
"""


class tiff_manager:
    def __init__(self, file, desired_chunk_depth=64):
        self.file = file

        self.ext = os.path.splitext(file)[-1]
        if self.ext == ".tiff" or self.ext == ".tif":
            img = self._get_tiff_zarr_array()
            self.shape = img.shape
            self.nbytes = img.nbytes
            self.ndim = img.ndim
            self.chunks = img.chunks
            self.dtype = img.dtype

        elif self.ext == ".jp2":
            img = self._read_jp2(slice(None))
            self.shape = img.shape
            self.nbytes = img.nbytes
            self.ndim = img.ndim
            self.chunks = (1, self.shape[1])
            self.dtype = img.dtype
        del img

        self._desired_chunk_depth = desired_chunk_depth
        self._adjust_chunk_depth()

    def __getitem__(self, key):
        # if key == (np.s_[0:0],)*self.ndim:
        if key == (slice(0, 0, None),) * self.ndim:
            # Hack to speed up dask array conversions
            return np.asarray([], dtype=self.dtype)
        return self._read_img(key)

    # def _read_tiff(self,key):
    #     with tifffile.imread(self.file,aszarr=True) as store:
    #         return zarr.open(store)[key]

    def _change_file(self, file):
        self.file = file

    def _read_img(self, key):
        if self.ext == ".tiff" or self.ext == ".tif":
            return self._read_tiff(key)
        elif self.ext == ".jp2":
            return self._read_jp2(key)

    def _get_tiff_zarr_array(self):
        with tifffile.imread(self.file, aszarr=True) as store:
            return zarr.open(store)

    def _read_tiff(self, key):
        print("Read {}".format(self.file))
        return self._get_tiff_zarr_array()[key]

    def _read_jp2(self, key):
        print("Read {}".format(self.file))
        with open(self.file, "rb") as f:
            img = io.BytesIO(f.read())
        return skimage.io.imread(img)[key]

    def clone_manager_new_file(self, file):
        """
        Changes only the file associated with the class
        Assumes that the new file shares all other properties
        No attempt is made to verify this

        This method is designed for speed.
        It is to be used when 1000s of tiff files must be referenced and
        it avoids opening each file to inspect metadata

        Returns: a new instance of the class with a different filename
        """
        new = deepcopy(self)
        new._change_file(file)
        return new

    def _adjust_chunk_depth(self):
        if self._desired_chunk_depth >= self.shape[0]:
            self.chunks = (self.shape[0], *self.chunks[1:])
        elif self._desired_chunk_depth % self.chunks[0] == 0:
            self.chunks = (self._desired_chunk_depth, *self.chunks[1:])


# Simple 3d tiff_manager without any chunk_depth options
class tiff_manager_3d:
    def __init__(self, fileList):
        assert isinstance(fileList, (list, tuple))
        self.fileList = fileList
        self.ext = os.path.splitext(fileList[0])[-1]

        img = self._get_tiff_zarr_array(0)
        self.shape = img.shape
        self.nbytes = img.nbytes
        self.ndim = img.ndim
        self.chunks = img.chunks
        self.dtype = img.dtype

        del img

        self._conv_3d()

    def _conv_3d(self):
        z_depth = len(self.fileList)
        self.shape = (z_depth, *self.shape)
        self.nbytes = int(self.nbytes * z_depth)
        self.ndim = 3
        self.chunks = (z_depth, *self.chunks)

    def __getitem__(self, key):
        # Hack to speed up dask array conversions
        if key == (slice(0, 0),) * self.ndim:
            return np.asarray([], dtype=self.dtype)

        return self._get_3d(key)

    @staticmethod
    def _format_slice(key):
        # print('In Slice {}'.format(key))
        if isinstance(key, slice):
            return (key,)

        if isinstance(key, int):
            return (slice(key, key + 1, None),)

        if isinstance(key, tuple):
            out_slice = []
            for ii in key:
                if isinstance(ii, slice):
                    out_slice.append(ii)
                elif isinstance(ii, int):
                    out_slice.append(slice(ii, ii + 1, None))
                else:
                    out_slice.append(ii)

        print('Out Slice {}'.format(out_slice))
        return tuple(out_slice)

    def _slice_out_shape(self, key):
        key = self._format_slice(key)
        key = list(key)
        if isinstance(key, int):
            key = [slice(key, None, None)]
            # print(key)
        out_shape = []
        for idx, _ in enumerate(self.shape):
            if idx < len(key):
                if isinstance(key[idx], int):
                    key[idx] = slice(key[idx], None, None)
                # print(key)
                test_array = np.asarray((1,) * self.shape[idx], dtype=bool)
                # print(test_array)
                # print(key[idx])
                test_array = test_array[key[idx]].shape[0]
                # print(test_array)
                out_shape.append(test_array)
            else:
                out_shape.append(self.shape[idx])
        out_shape = tuple(out_shape)
        # print(out_shape)
        return out_shape

    def _read_img(self, key, idx):
        if self.ext == ".tiff" or self.ext == ".tif":
            return self._read_tiff(key, idx)
        elif self.ext == ".jp2":
            return self._read_jp2(key, idx)

    def _get_tiff_zarr_array(self, idx):
        with tifffile.imread(self.fileList[idx], aszarr=True) as store:
            return zarr.open(store)

    def _read_tiff(self, key, idx):
        print("Read {}".format(self.fileList[idx]))
        return self._get_tiff_zarr_array(idx)[key]

    def _read_jp2(self, key, idx):
        print("Read {}".format(self.fileList[idx]))
        with open(self.fileList[idx], "rb") as f:
            img = io.BytesIO(f.read())
        return skimage.io.imread(img)[key]
        # return skimage.io.imread(self.fileList[idx])[key]

    def _get_3d(self, key):
        print('top of _get_3d and key=', key)

        key = self._format_slice(key)
        shape_of_output = self._slice_out_shape(key)
        canvas = np.zeros(shape_of_output, dtype=self.dtype)
        print(f'_get_3d canvas shape={canvas.shape} key={key}')

        # if len(key) == 1:
        #     key = key[slice(None)]

        for idx in range(canvas.shape[0]):
            two_d = key[1:]
            # print(two_d)
            if len(two_d) == 1:
                two_d = two_d[0]
            # print(two_d)
            canvas[idx] = self._read_img(two_d, idx)
        return canvas

    def _change_file_list(self, fileList):
        old_zdepth = self.shape[0]

        self.fileList = fileList

        new_zdepth = len(self.fileList)
        self.shape = (new_zdepth, *self.shape[1:])
        self.nbytes = int(self.nbytes / old_zdepth * new_zdepth)
        self.chunks = (new_zdepth, *self.chunks[1:])

    def clone_manager_new_file_list(self, fileList):
        """
        Changes only the file associated with the class
        Assumes that the new file shares all other properties
        No attempt is made to verify this

        This method is designed for speed.
        It is to be used when 1000s of tiff files must be referenced and
        it avoids opening each file to inspect metadata

        Returns: a new instance of the class with a different filename
        """
        new = deepcopy(self)
        #####new._change_file_list(fileList)
        return new


class tiff_manager_3d_to_5d:
    """
    Given a list of 3d tiff_managers that each represent a color channel, create a 5 dimensional manager where time
    is always assumed to be a shape of 1

    Enable the creation of a virtual array that has some of the elements of a dask array or zarr array but none of
    the computational overhead
    """

    def __init__(self, list_of_tiff_manager_3d):

        assert isinstance(list_of_tiff_manager_3d, (list, tuple, tiff_manager_3d))
        if isinstance(list_of_tiff_manager_3d, (list, tuple)):
            for ii in list_of_tiff_manager_3d:
                assert isinstance(ii, tiff_manager_3d)

        if isinstance(list_of_tiff_manager_3d, (list, tuple)):
            self.stack = tuple(list_of_tiff_manager_3d)
        else:
            self.stack = (list_of_tiff_manager_3d,)

        self.shape = (1, len(self.stack), *self.stack[0].shape)
        self.nbytes = self.stack[0].nbytes * len(self.stack)
        self.ndim = 5
        self.chunks = (1, 1, 1, *self.stack[0].chunks)
        self.chunks = (1, *self.stack[0].chunks)
        self.dtype = self.stack[0].dtype

    def __getitem__(self, key):
        # REQUIRES A 5-DIM tuple of slices

        # Hack to speed up dask array conversions (not tested to work with dask)
        if key == (slice(0, 0, None),) * self.ndim:
            return np.asarray([], dtype=self.dtype)

        key = [slice(x, x + 1) if isinstance(x, int) else x for x in key]
        # key = [slice(None) if x is None else x for x in key]
        print(key)

        new_key = []
        for idx in range(5):
            if key[idx] == slice(None):
                new_key.append(slice(0, self.shape[idx]))
            elif isinstance(key[idx], slice):
                new_key.append(key[idx])
            else:
                new_key.append(key[idx])

        key = new_key
        print(key)

        t = 1
        c = key[1].stop - key[1].start if key[1] is not None else self.shape[1]
        z = key[2].stop - key[2].start if key[2] is not None else self.shape[2]
        y = key[3].stop - key[3].start if key[3] is not None else self.shape[3]
        x = key[4].stop - key[4].start if key[4] is not None else self.shape[4]

        canvas = np.zeros((t, c, z, y, x), dtype=self.dtype)
        print(canvas.shape)
        for cc in range(key[1].start, key[1].stop):
            print((slice(0, 1), slice(cc, cc + 1), *key[2:]))

            canvas[
                (slice(0, 1), slice(cc, cc + 1), slice(0, z), slice(0, y), slice(0, x))
            ] = self.stack[cc][(*key[2:],)]
        return canvas


def get_size_GB(shape, dtype):

    current_size = math.prod(shape) / 1024**3
    if dtype == np.dtype("uint8"):
        pass
    elif dtype == np.dtype("uint16"):
        current_size *= 2
    elif dtype == np.dtype("float32"):
        current_size *= 4
    elif dtype == float:
        current_size *= 8

    return current_size


def optimize_chunk_shape_3d(image_shape, origional_chunks, dtype, chunk_limit_GB):

    current_chunks = origional_chunks
    current_size = get_size_GB(current_chunks, dtype)

    print(current_chunks)
    print(current_size)

    if current_size > chunk_limit_GB:
        return current_size

    idx = 0
    while current_size <= chunk_limit_GB:

        last_size = get_size_GB(current_chunks, dtype)
        last_shape = current_chunks

        chunk_iter_idx = idx % 2
        if chunk_iter_idx == 0:
            current_chunks = (
                origional_chunks[0],
                current_chunks[1] + origional_chunks[1],
                current_chunks[2],
            )
        elif chunk_iter_idx == 1:
            current_chunks = (
                origional_chunks[0],
                current_chunks[1],
                current_chunks[2] + origional_chunks[2],
            )

        current_size = get_size_GB(current_chunks, dtype)

        if current_size > chunk_limit_GB:
            return last_shape
        if any([x > y for x, y in zip(current_chunks, image_shape)]):
            return last_shape

        idx += 1
