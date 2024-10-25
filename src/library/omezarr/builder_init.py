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
from library.image_manipulation.image_manager import ImageManager
from library.omezarr.builder_img_processing import BuilderDownsample
from library.omezarr.builder_utils import BuilderUtils
from library.omezarr.builder_ome_zarr_utils import BuilderOmeZarrUtils
from library.omezarr.builder_multiscale_generator import BuilderMultiscaleGenerator
from library.omezarr.tiff_manager import TiffManager
from library.utilities.dask_utilities import get_pyramid

class builder(BuilderDownsample,
            BuilderUtils,
            BuilderOmeZarrUtils,
     #####       BuilderImageUtils,
            BuilderMultiscaleGenerator):
    '''
    A mix-in class for builder.py
    '''
    def __init__(
        self,
        in_location,
        out_location,
        filesList,
        geometry=(1, 1, 1),
        originalChunkSize=(1, 1, 1, 2048, 2048),
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
        self.sim_jobs = self.cpu_cores - 2 
        self.workers = 1
        self.mem = int((psutil.virtual_memory().free / 1024**3) * 0.8)
        self.compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
        self.zarr_store_type = zarr.storage.NestedDirectoryStore
        self.tmp_dir = tmp_dir
        self.debug = debug
        self.omero_dict = omero_dict
        self.downSampType = "mean"
        self.mips = mips
        self.res0_chunk_limit_GB = self.mem / self.cpu_cores / 8 #Fudge factor for maximizing data being processed with available memory during res0 conversion phase
        self.res_chunk_limit_GB = self.mem / self.cpu_cores / 24 #Fudge factor for maximizing data being processed with available memory during downsample phase
        # workers = cpu_count, sim=1 died right away
        # workers = 1 sim=1 worker uses about 20%ram
        # workers = 1 sim=2 complains about ram
        # workers = 2, seems each worker uses about 20%ram
        # workers = 8, complains
        # Makes store location and initial group
        # do not make a class attribute because it may not pickle when computing over dask

        #####store = self.get_store_from_path(self.output) # location: _builder_utils

        self.TimePoints = 1

        image_manager = ImageManager(self.input)
        self.dtype = image_manager.dtype
        self.ndim = image_manager.ndim
        self.channels = image_manager.num_channels
        self.shape_3d = (len(self.filesList),*image_manager.shape)
        self.shape = (self.TimePoints, self.channels, *self.shape_3d)
        out_shape = self.shape_3d
        #initial_chunk = self.originalChunkSize[2:]
        #final_chunk_size = self.finalChunkSize[2:]
        #resolution = self.geometry[2:]

        initial_chunk = self.originalChunkSize[1:]
        final_chunk_size = self.finalChunkSize[1:]
        resolution = self.geometry[1:]

        print(f'initial chunk = {initial_chunk}')
        print(f'final chunk size = {final_chunk_size}')
        print(f'resolution = {resolution}')

        self.pyramidMap = get_pyramid(out_shape, initial_chunk, final_chunk_size, resolution,  self.mips)
        for k, v in self.pyramidMap.items():
            print(k,v)
        

        self.build_zattrs()
