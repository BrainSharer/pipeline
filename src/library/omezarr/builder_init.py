# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 21:11:13 2022

@author: awatson
"""

import zarr
import os
from numcodecs import Blosc


## Import mix-in classes
from library.image_manipulation.image_manager import ImageManager
from library.omezarr.builder_img_processing import BuilderDownsample
from library.omezarr.builder_utils import BuilderUtils
from library.omezarr.builder_ome_zarr_utils import BuilderOmeZarrUtils
from library.omezarr.builder_multiscale_generator import BuilderMultiscaleGenerator
from library.utilities.dask_utilities import closest_divisors_to_target

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
        files,
        resolution,
        originalChunkSize,
        tmp_dir,
        debug,
        omero_dict,
        mips,
        available_memory
    ):

        self.input = in_location
        self.output = out_location
        self.files = sorted(files)
        self.resolution = resolution
        self.originalChunkSize = tuple(originalChunkSize)
        self.cpu_cores = os.cpu_count()
        self.sim_jobs = self.cpu_cores // 2 
        self.workers = 1
        self.compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
        self.zarr_store_type = zarr.storage.NestedDirectoryStore
        self.tmp_dir = tmp_dir
        self.debug = debug
        self.omero_dict = omero_dict
        self.downSampType = "mean"
        self.mips = mips
        self.available_memory = available_memory
        self.res0_chunk_limit_GB = self.available_memory / self.cpu_cores / 8 #Fudge factor for maximizing data being processed with available memory during res0 conversion phase
        self.res_chunk_limit_GB = self.available_memory / self.cpu_cores / 24 #Fudge factor for maximizing data being processed with available memory during downsample phase
        # workers = cpu_count, sim=1 died right away
        # workers = 1 sim=1 worker uses about 20%ram
        # workers = 1 sim=2 complains about ram
        # workers = 2, seems each worker uses about 20%ram
        # workers = 8, complains
        # workers = 2, sim=14 dies on high res

        #####store = self.get_store_from_path(self.output) # location: _builder_utils


        image_manager = ImageManager(self.input)
        self.dtype = image_manager.dtype
        self.ndim = image_manager.ndim
        self.channels = image_manager.num_channels
        self.img_shape = image_manager.shape
        self.shape_3d = (len(self.files),*image_manager.shape)

        self.pyramidMap = {}
        self.pyramidMap[0] = {'chunk': self.originalChunkSize, 'resolution': resolution, 'downsample': (1, 1, 1)}
        z_chunk = closest_divisors_to_target(image_manager.len_files, 64)
        for mip in range(1, mips + 1):
            previous_chunks = self.pyramidMap[mip-1]['chunk']
            previous_resolution = self.pyramidMap[mip-1]['resolution']

            if mip < 3:
                x_chunk = closest_divisors_to_target(previous_chunks[-1], previous_chunks[-1] // 8)
                y_chunk = closest_divisors_to_target(previous_chunks[-2], previous_chunks[-2] // 8)
                chunks = (1, self.channels, z_chunk, y_chunk, x_chunk)
            else:
                chunks = (1, self.channels, 64, 64, 64)

            resolution = (resolution[0], previous_resolution[1] * 2, previous_resolution[2] * 2)
            self.pyramidMap[mip] = {'chunk': chunks, 'resolution': resolution, 'downsample': (1, 2, 2)}



        for k, v in self.pyramidMap.items():
            print(k,v)
        self.build_zattrs()
