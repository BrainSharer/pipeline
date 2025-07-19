# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 21:11:13 2022

@author: awatson
"""

import zarr
import os
from numcodecs import Blosc


from library.image_manipulation.image_manager import ImageManager
from library.omezarr.builder_ome_zarr_utils import BuilderOmeZarrUtils
from library.omezarr.builder_multiscale_generator import BuilderMultiscaleGenerator
from library.utilities.dask_utilities import closest_divisors_to_target

class builder(BuilderOmeZarrUtils, BuilderMultiscaleGenerator):

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
        downsample
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

        image_manager = ImageManager(self.input)
        self.dtype = image_manager.dtype
        self.ndim = image_manager.ndim
        self.channels = image_manager.num_channels
        self.img_shape = image_manager.shape
        self.shape_3d = (len(self.files),*image_manager.shape)
        self.initial_resolution = "rechunkme"

        self.pyramidMap = {}
        self.pyramidMap[self.initial_resolution] = {'chunk': self.originalChunkSize, 'resolution': resolution, 'downsample': (1, 1, 1)}
        z_chunk = closest_divisors_to_target(image_manager.len_files, 64)
        for mip in range(0, mips + 1):
            if mip == 0:
                chunks = (1, self.channels, z_chunk, 512, 512)
                previous_resolution = self.pyramidMap[self.initial_resolution]['resolution']
            elif mip == 1:
                chunks = (1, self.channels, z_chunk, 256, 256)
                previous_resolution = self.pyramidMap[mip - 1]['resolution']
            else:
                chunks = (1, self.channels, z_chunk, 64, 64)
                previous_resolution = self.pyramidMap[mip - 1]['resolution']

            if downsample:
                chunks = (1, self.channels, 64, 64, 64)

            if mip == 0:
                resolution = (previous_resolution[0], previous_resolution[1], previous_resolution[2])
                downsample = (1, 1, 1)
            else:
                resolution = (resolution[0], previous_resolution[1] * 2, previous_resolution[2] * 2)
                downsample = (1, 2, 2)
            self.pyramidMap[mip] = {'chunk': chunks, 'resolution': resolution, 'downsample': downsample}

        for k, v in self.pyramidMap.items():
            print(k,v)
        if self.debug:
            exit(1)
        self.build_zattrs()
