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
        scratch_space,
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
        self.scratch_space = scratch_space
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
        scratch_parent = os.path.dirname(self.scratch_space)
        self.initial_resolution = "rechunkme"
        self.transfer_path = os.path.join(scratch_parent, self.initial_resolution)


        self.pyramidMap = {}
        z_chunk = closest_divisors_to_target(image_manager.len_files, 64)
        x_chunk = closest_divisors_to_target(image_manager.width, 2056)
        y_chunk = closest_divisors_to_target(image_manager.height, 2056)
        self.pyramidMap[0] = {'chunk': (1, self.channels, z_chunk, y_chunk, x_chunk), 'resolution': resolution, 'downsample': (1, 1, 1)}
        for mip in range(1, mips):
            previous_resolution = self.pyramidMap[mip-1]['resolution']            

            if mip < 3:
                chunks = (1, self.channels, z_chunk, 128, 128)
            else:
                chunks = (1, self.channels, 64, 64, 64)

            if downsample:
                chunks = (1, self.channels, 64, 64, 64)

            resolution = (resolution[0], previous_resolution[1] * 2, previous_resolution[2] * 2)
            self.pyramidMap[mip] = {'chunk': chunks, 'resolution': resolution, 'downsample': (1, 2, 2)}

        for k, v in self.pyramidMap.items():
            print(k,v)
        if self.debug:
            exit(1)
        self.build_zattrs()
