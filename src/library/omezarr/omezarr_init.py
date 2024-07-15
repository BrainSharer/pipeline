# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 21:11:13 2022

@author: awatson
"""

import zarr
import os
import psutil
import glob
from numcodecs import Blosc
from natsort import natsorted

# Import local functions
from library.omezarr.builder_image_utils import tiff_manager
from library.omezarr.builder_image_processing import _builder_downsample
from library.omezarr.builder_utils import _builder_utils
from library.omezarr.builder_ome_zarr_utils import _builder_ome_zarr_utils
from library.omezarr.builder_multiscale_generator import _builder_multiscale_generator

class OmeZarrBuilder(_builder_downsample,
            _builder_utils,
            _builder_ome_zarr_utils,
            _builder_multiscale_generator):
    '''
    A mix-in class for builder.py
    '''
    def __init__(
        self,
        in_location,
        out_location,
        scales=(1, 1, 20.0, 10.4, 10.4),
        originalChunkSize=(1, 1, 1, 64, 64),
        finalChunkSize=(1, 1, 32, 32, 32),
        cpu_cores=os.cpu_count(),
        mem=int((psutil.virtual_memory().free / 1024**3) * 0.8),
        zarr_store_type=zarr.storage.NestedDirectoryStore,
        tmp_dir="/tmp",
        debug=False,
        omero_dict={},
        downSampType="mean",
    ):

        self.in_location = in_location
        self.out_location = out_location
        self.scales = tuple(scales)
        self.originalChunkSize = tuple(originalChunkSize)
        self.finalChunkSize = tuple(finalChunkSize)
        self.cpu_cores = cpu_cores
        self.sim_jobs = 1
        self.workers = int(self.cpu_cores / self.sim_jobs)
        self.mem = mem
        self.zarr_store_type = zarr_store_type
        self.tmp_dir = tmp_dir
        self.debug = debug
        self.omero_dict = omero_dict
        self.downSampType = downSampType

        # Hack to build zarr in tmp location then copy to finalLocation (finalLocation is the original out_location)
        self.finalLocation = self.out_location
        self.res0_chunk_limit_GB = self.mem / self.cpu_cores / 8 #Fudge factor for maximizing data being processed with available memory during res0 conversion phase
        self.res_chunk_limit_GB = self.mem / self.cpu_cores / 24 #Fudge factor for maximizing data being processed with available memory during downsample phase

        # Makes store location and initial group
        # do not make a class attribute because it may not pickle when computing over dask

        store = self.get_store_from_path(self.out_location) # location: _builder_utils

        # Sanity check that we can open the store
        store = zarr.open(store)
        del store
        filesList = []

        ##  LIST ALL FILES TO BE CONVERTED  ##
        filesList.append(natsorted(glob.glob(os.path.join(self.in_location, '*.tif') ) ) )
        

        self.filesList = filesList
        self.channels = len(self.filesList)
        self.TimePoints = 1

        testImage = tiff_manager(self.filesList[0][0])
        self.dtype = testImage.dtype
        self.ndim = testImage.ndim
        self.shape_3d = (len(self.filesList[0]),*testImage.shape)

        self.shape = (self.TimePoints, self.channels, *self.shape_3d)

        # self.pyramidMap = self.imagePyramidNum()
        self.pyramidMap = self.imagePyramidNum_converge_isotropic()

        self.build_zattrs()