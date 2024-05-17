# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 21:11:13 2022

@author: awatson
"""

import zarr
import os
import psutil
from numcodecs import Blosc


## Import mix-in classes
from library.omezarr.builder_img_processing import _builder_downsample
from library.omezarr.builder_utils import _builder_utils
from library.omezarr.builder_ome_zarr_utils import _builder_ome_zarr_utils
from library.omezarr.builder_image_utils import _builder_image_utils
from library.omezarr.builder_multiscale_generator import _builder_multiscale_generator
from library.omezarr.tiff_manager import tiff_manager
from library.utilities.dask_utilities import get_pyramid
# from stack_to_multiscale_ngff._builder_colors import _builder_colors

class builder(_builder_downsample,
            _builder_utils,
            _builder_ome_zarr_utils,
            _builder_image_utils,
            _builder_multiscale_generator):
    '''
    A mix-in class for builder.py
    '''
    def __init__(
        self,
        in_location,
        out_location,
        filesList,
        geometry=(1, 1, 1),
        originalChunkSize=(1, 1, 1, 1024, 1024),
        finalChunkSize=(1, 1, 64, 64, 64),
        tmp_dir="/tmp",
        debug=False,
        omero_dict={},
        mips=4,
    ):

        self.input = in_location
        self.output = out_location
        self.filesList = filesList
        self.geometry = tuple(geometry)
        self.originalChunkSize = tuple(originalChunkSize)
        self.finalChunkSize = tuple(finalChunkSize)
        self.cpu_cores = os.cpu_count()
        self.sim_jobs = 2
        self.workers = int(self.cpu_cores / self.sim_jobs) // 2
        self.mem = int((psutil.virtual_memory().free / 1024**3) * 0.9)
        self.compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
        self.zarr_store_type = zarr.storage.NestedDirectoryStore
        self.tmp_dir = tmp_dir
        self.debug = debug
        self.omero_dict = omero_dict
        self.downSampType = "mean"
        self.mips = mips
        self.res0_chunk_limit_GB = self.mem / (self.cpu_cores - 2) / 8 #Fudge factor for maximizing data being processed with available memory during res0 conversion phase
        self.res_chunk_limit_GB = self.mem / (self.cpu_cores - 2) / 24 #Fudge factor for maximizing data being processed with available memory during downsample phase

        # Makes store location and initial group
        # do not make a class attribute because it may not pickle when computing over dask

        #####store = self.get_store_from_path(self.output) # location: _builder_utils

        self.Channels = len(self.filesList)
        self.TimePoints = 1
        # print(self.Channels)
        # print(self.filesList)

        testImage = tiff_manager(self.filesList[0][0])
        self.dtype = testImage.dtype
        self.ndim = testImage.ndim
        self.shape_3d = (len(self.filesList[0]),*testImage.shape)
        self.shape = (self.TimePoints, self.Channels, *self.shape_3d)
        out_shape = self.shape_3d
        initial_chunk = self.originalChunkSize[2:]
        final_chunk_size = self.finalChunkSize[2:]
        resolution = self.geometry[2:]

        self.pyramidMap = get_pyramid(out_shape, initial_chunk, final_chunk_size, resolution,  self.mips)
        for k, v in self.pyramidMap.items():
            print(k,v)
        

        #import sys
        #sys.exit()
        self.build_zattrs()
