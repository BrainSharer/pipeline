# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 21:11:13 2022

@author: awatson
"""

import zarr
import os
import numpy as np
import tifffile
import math
from copy import deepcopy
import sys


MINFILES = 5
# Simple 3d tiff_manager without any chunk_depth options
class TiffManager3d:

    def __init__(self, files, channel):
        assert isinstance(files,(list,tuple))
        assert len(files) > MINFILES, 'files must be a list of at least 5 tiff files'
        self.files = files
        self.channel = channel
        self.num_channels = 1
        self.ext = os.path.splitext(files[0][0])[1]
        img = self._get_tiff_zarr_array(0, self.channel)
        self.shape = img.shape
        print(f'tiff mananger individual tiff self.shape={self.shape}')
        self.nbytes = img.nbytes
        self.ndim = img.ndim
        self.chunks = img.chunks
        self.dtype = img.dtype
        del img
        self._conv_3d()

    def _conv_3d(self):
        z_depth = len(self.files)
        self.shape = (z_depth, *self.shape)
        print(f'tiff mananger stack self.shape={self.shape}')
        self.nbytes = int(self.nbytes * z_depth)
        self.ndim = 3
        self.chunks = (z_depth, *self.chunks)

    def __getitem__(self, key):
        # Hack to speed up dask array conversions
        if key == (slice(0, 0, None),) * self.ndim:
            return np.asarray([], dtype=self.dtype)

        return self._get_3d(key)

    @staticmethod
    def _format_slice(key):
        # print('In Slice {}'.format(key))
        if isinstance(key,slice):
            return (key,)

        if isinstance(key,int):
            return (slice(key,key+1,None),)

        if isinstance(key,tuple):
            out_slice = []
            for ii in key:
                if isinstance(ii,slice):
                    out_slice.append(ii)
                elif isinstance(ii,int):
                    out_slice.append(slice(ii,ii+1,None))
                else:
                    out_slice.append(ii)

        # print('Out Slice {}'.format(out_slice))
        return tuple(out_slice)

    def _slice_out_shape(self,key):
        key = self._format_slice(key)
        key = list(key)
        if isinstance(key,int):
            key = [slice(key,None,None)]
            # print(key)
        out_shape = []
        for idx,_ in enumerate(self.shape):
            if idx < len(key):
                if isinstance(key[idx],int):
                    key[idx] = slice(key[idx],None,None)
                # print(key)
                test_array = np.asarray((1,)*self.shape[idx],dtype=bool)
                # print(test_array)
                # print(key[idx])
                test_array = test_array[key[idx]].shape[0]
                # print(test_array)
                out_shape.append(
                        test_array
                    )
            else:
                out_shape.append(self.shape[idx])
        out_shape = tuple(out_shape)
        # print(out_shape)
        return out_shape

    def _get_tiff_zarr_array(self, idx, channel=0):
        infile = self.files[idx]
        assert isinstance(idx, int) , 'idx must be an integer'
        assert isinstance(channel, int) , 'channel must be an integer'
        assert len(self.files) > idx, 'idx out of range'
        if str(infile).endswith('.tif') == False:
            infile = infile[0]
        if str(infile).endswith('.tif') == False:
            infile = infile[0]
            sys.exit()

        print(f'Read channel={channel} {infile}')
        with tifffile.imread(infile, aszarr=True) as store:
            z = zarr.open(store, mode='r')
            self.num_channels = z.ndim
            if self.num_channels == 2:
                return z
            z = z[... ,channel]
            z = zarr.array(z)            
            return z

    def _read_tiff(self, key, idx):
        # print('Read {}'.format(self.files[idx]))
        tif = self._get_tiff_zarr_array(idx, self.channel)
        return tif[key]

    def _get_3d(self,key):
        key = self._format_slice(key)
        shape_of_output = self._slice_out_shape(key)
        canvas = np.zeros(shape_of_output,dtype=self.dtype)

        for idx in range(canvas.shape[0]):
            two_d = key[1:]
            if len(two_d) == 1:
                two_d = two_d[0]
            canvas[idx] = self._read_tiff(two_d, idx)
        return canvas

    def _change_file_list(self, files):
        old_zdepth = self.shape[0]

        self.files = files

        new_zdepth = len(self.files)
        #self.shape = (new_zdepth,*self.shape[1:])
        self.shape = (1,*self.shape[1:])
        #print(f'_change_file_list stack self.shape={self.shape} old_zdepth={old_zdepth} new_zdepth={new_zdepth}')
        self.nbytes = int(self.nbytes / old_zdepth * new_zdepth)
        self.chunks = (new_zdepth,*self.chunks[1:])

    def clone_manager_new_file_list(self, files):
        '''
        Changes only the file associated with the class
        Assumes that the new file shares all other properties
        No attempt is made to verify this
        
        This method is designed for speed.
        It is to be used when 1000s of tiff files must be referenced and 
        it avoids opening each file to inspect metadata
        
        Returns: a new instance of the class with a different filename
        '''
        new = deepcopy(self)
        new._change_file_list(files)
        return new
    

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

def optimize_chunk_shape_3d(image_shape,origional_chunks,dtype,chunk_limit_GB):
    
    current_chunks = origional_chunks
    current_size = get_size_GB(current_chunks,dtype)
    
    print(current_chunks)
    print(current_size)
    
    if current_size > chunk_limit_GB:
        return current_size
    
    idx = 0
    while current_size <= chunk_limit_GB:
        
        #####last_size = get_size_GB(current_chunks,dtype)
        last_shape = current_chunks
        
        chunk_iter_idx = idx%2
        if chunk_iter_idx == 0:
            current_chunks = (origional_chunks[0],current_chunks[1]+origional_chunks[1],current_chunks[2])
        elif chunk_iter_idx == 1:
            current_chunks = (origional_chunks[0],current_chunks[1],current_chunks[2]+origional_chunks[2])
            
        current_size = get_size_GB(current_chunks,dtype)
        
        print(current_chunks)
        print(current_size)
        print('next step chunk limit {}'.format(current_size))
        
        
        if current_size > chunk_limit_GB:
            return last_shape
        if any([x>y for x,y in zip(current_chunks,image_shape)]):
            return last_shape
        
        idx += 1
