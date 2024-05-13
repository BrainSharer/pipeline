"""
Place the yaml below in: ~/.config/dask/distributed.yaml

distributed:
  worker:
    # Fractions of worker memory at which we take action to avoid memory blowup
    # Set any of the lower three values to False to turn off the behavior entirely
    memory:
      target: 0.50  # target fraction to stay below
      spill: 0.60  # fraction at which we spill to disk
      pause: 0.70  # fraction at which we pause worker threads
      terminate: False  # fraction at which we terminate the worker
"""
import glob
import math
import os
import sys
import shutil
import time
import psutil
import zarr
import dask
import numpy as np
from itertools import product

from dask.delayed import delayed
import dask.array as da

from skimage import io, img_as_uint, img_as_ubyte, img_as_float32, img_as_float64

from timeit import default_timer as timer
from dask.distributed import Client, progress
from library.image_manipulation.image_manager import ImageManager
from library.omezarr.tiff_manager import tiff_manager, tiff_manager_3d
from library.omezarr.builder_init import builder
from library.omezarr.utils import get_size_GB, optimize_chunk_shape_3d_2
from library.utilities.dask_utilities import aligned_coarse_chunks, imreads, mean_dtype
from library.utilities.utilities_process import SCALING_FACTOR, get_cpus, get_scratch_dir

class OmeZarrManager():
    """
    Manages the creation of OmeZarr files by performing necessary setup, generating transformations, and writing the data to the file.
    Chunk size is very important. On the first mip, set it so the store has 1,2048,2048 chunks.
    This works best. The chunks in later mips then get set to a lower size for better viewing in neuroglancer.

    Attributes:
        tmp_dir (str): The temporary directory for storing intermediate files.
        xy_resolution (float): The resolution of the xy plane.
        z_resolution (float): The resolution of the z axis.
        trimto (int): The trimto value based on the downsample flag.
        storefile (str): The name of the store file.
        scaling_factor (int): The scaling factor based on the downsample flag.
        input (str): The input file location.
        chunks (list): The list of chunk sizes for each MIP level.
        initial_chunks (list): The initial chunk size.
        mips (int): The number of MIP levels.
        ndims (int): The number of dimensions of the input image.
        storepath (str): The path to the store file.
        axes (list): The list of axis configurations.
        axis_scales (list): The list of coarsen values for each axis.
    """

    def omezarr_setup(self):
        tmp_dir = get_scratch_dir()
        self.tmp_dir = os.path.join(tmp_dir, f'{self.animal}')
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.xy_resolution = self.sqlController.scan_run.resolution
        self.z_resolution = self.sqlController.scan_run.zresolution
        if self.downsample:
            # Main mip took 10.05 seconds with chunks=[1, 256, 256] trimto=8
            # Main mip took 4.97 seconds with chunks=[1, 256, 256] trimto=8
            # Main mip took 4.87 seconds with chunks=[1, 256, 256] trimto=1
            self.trimto = 8
            self.storefile = 'C1T.zarr'
            self.rechunkmefile = 'C1T_rechunk.zarr'
            self.scaling_factor = SCALING_FACTOR
            self.input = os.path.join(self.fileLocationManager.prep, 'C1', 'thumbnail_aligned')
            self.chunks = [
                    [64, 64, 64],
                    [32, 32, 32],
                    [32, 32, 32],
                    [32, 32, 32],
                ]
            self.mips = len(self.chunks)
            self.mips = 1
        else:
            #
            self.trimto = 64
            self.storefile = 'C1.zarr'
            self.rechunkmefile = 'C1_rechunk.zarr'
            self.scaling_factor = 1
            self.input = os.path.join(self.fileLocationManager.prep, 'C1', 'full_aligned')
            self.chunks = [
                    [64, 128, 128],
                    [64, 64, 64],
                    [64, 64, 64],
                    [32, 32, 32],
                    [32, 32, 32],
                    [32, 32, 32],
                ]
            self.mips = len(self.chunks)
        # vars from stack to multi
        self.originalChunkSize = [1, 1, 1, 128, 128]
        self.finalChunkSize=(1, 1, 64, 64, 64)
        self.cpu_cores = os.cpu_count()
        self.mem = (psutil.virtual_memory().free // 1024**3) * 0.8
        self.res0_chunk_limit_GB = self.mem / self.cpu_cores / 8 #Fudge factor for maximizing data being processed with available memory during res0 conversion phase
        self.res_chunk_limit_GB = self.mem / self.cpu_cores / 24 #Fudge factor for maximizing data being processed with available memory during downsample phase
        filesList = []
        for file in sorted(os.listdir(self.input)):
            filepath = os.path.join(self.input, file)
            filesList.append(filepath)
        self.filesList = [filesList]
        self.Channels = len(self.filesList)
        self.TimePoints = 1
        testImage = tiff_manager(self.filesList[0][0])
        self.dtype = testImage.dtype
        self.ndim = testImage.ndim
        self.shape_3d = (len(self.filesList[0]),*testImage.shape)        
        self.shape = (self.TimePoints, self.Channels, *self.shape_3d)
        self.directToFinalChunks = True # Use final chunks for all multiscales except full resolution
        omero = {}
        omero['channels'] = {}
        omero['channels']['color'] = None
        omero['channels']['label'] = None
        omero['channels']['window'] = None
        omero['name'] = self.animal
        omero['rdefs'] = {}
        omero['rdefs']['defaultZ'] = len(filesList) // 2
        self.omero_dict = omero

        image_manager = ImageManager(self.input)
        self.ndims = image_manager.ndim
        self.dtype = image_manager.dtype

        self.storepath = os.path.join(
            self.fileLocationManager.www, "neuroglancer_data", self.storefile
        )
        self.axes = [
            {
                "name": "z",
                "type": "space",
                "unit": "micrometer",
                "coarsen": 1,
                "resolution": self.z_resolution,
            },
            {
                "name": "y",
                "type": "space",
                "unit": "micrometer",
                "coarsen": 2,
                "resolution": self.xy_resolution * self.scaling_factor,
            },
            {
                "name": "x",
                "type": "space",
                "unit": "micrometer",
                "coarsen": 2,
                "resolution": self.xy_resolution * self.scaling_factor,
            },
        ]
        self.axis_scales = [a["coarsen"] for a in self.axes]
        xy = self.xy_resolution * self.scaling_factor
        self.geometry = (1, 1, self.z_resolution, xy, xy)

    def create_omezarr(self):
        """
            Creates an omezarr file by performing the necessary setup, generating transformations,
            and writing the data to the file.

            Returns:
                None
            """
        # self.transformations = get_transformations(self.axes, self.mips + 1)
        # for transformation in self.transformations:
        #    print(transformation)
        self.workers = self.cpu_cores
        self.sim_jobs = 4
        #self.workers = 1
        #self.sim_jobs = 1
        GB = (psutil.virtual_memory().free // 1024**3) * 0.8
        # workers, _ = get_cpus()

        omezarr = builder(
            self.input,
            self.storepath,
            geometry=self.geometry,
            originalChunkSize=self.originalChunkSize,
            finalChunkSize=self.finalChunkSize,
            cpu_cores=self.cpu_cores,
            sim_jobs=self.sim_jobs,
            mem=self.mem,
            tmp_dir=self.tmp_dir,
            debug=self.debug,
            omero_dict=self.omero_dict,
            directToFinalChunks=self.directToFinalChunks
        )


        try:
            with dask.config.set({'temporary_directory': self.tmp_dir, 
                                    'logging.distributed': 'error'}):

                os.environ["DISTRIBUTED__COMM__TIMEOUTS__CONNECT"] = "160s"
                os.environ["DISTRIBUTED__COMM__TIMEOUTS__TCP"] = "160s"
                os.environ["DISTRIBUTED__DEPLOY__LOST_WORKER"] = "160s"
                # https://docs.dask.org/en/stable/array-best-practices.html#orient-your-chunks
                os.environ["OMP_NUM_THREADS"] = "1"
                os.environ["MKL_NUM_THREADS"] = "1"
                os.environ["OPENBLAS_NUM_THREADS"] = "1"

                print(f'Starting distributed dask with {self.workers} workers and {self.sim_jobs} sim_jobs in tmp dir={self.tmp_dir} with free memory={GB}GB')
                print('With Dask memory config:')
                print(dask.config.get("distributed.worker.memory"))
                print()
                omezarr.write_resolution_series()

        except Exception as ex:
            print('Exception in running builder in omezarr_manager')
            print(ex)
