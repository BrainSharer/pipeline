import os
import sys
from packaging.version import Version
import cloudvolume
from cloudvolume import CloudVolume
from taskqueue.taskqueue import LocalTaskQueue
import igneous.task_creation as tc
import numpy as np

from library.image_manipulation.neuroglancer_manager import NumpyToNeuroglancer
from library.utilities.utilities_process import test_dir
from library.image_manipulation.image_manager import ImageManager
XY_CHUNK = 64
Z_CHUNK = 64


class NgPrecomputedMaker:
    """Class to convert a tiff image stack to the precomputed
    neuroglancer format code from Seung lab
    """

    def __init__(self, sqlController, *args, **kwargs):
        self.sqlController = sqlController
        # Initialize other attributes as needed
        self.input = kwargs.get('input', None)
        self.output = kwargs.get('output', None)
        self.animal = kwargs.get('animal', None)
        self.section_count = kwargs.get('section_count', None)
        self.downsample = kwargs.get('downsample', False)
        self.scaling_factor = kwargs.get('scaling_factor', 1.0)
        self.rechunkme_path = kwargs.get('rechunkme_path', None)
        self.progress_dir = kwargs.get('progress_dir', None)
        self.fileLogger = kwargs.get('fileLogger', None)
        self.debug = kwargs.get('debug', False)

    def get_scales(self):
        """returns the scanning resolution for a given animal.  
        The scan resolution and sectioning thickness are retrived from the database.
        The resolution in the database is stored as micrometers (microns -um). 

        :returns: list of converstion factors from pixel to micron for x,y and z
        """
        db_resolution = self.sqlController.scan_run.resolution
        zresolution = self.sqlController.scan_run.zresolution
        resolution = db_resolution
        if self.downsample:
          if zresolution < 20:
                zresolution *=  self.scaling_factor

          resolution *=  self.scaling_factor         

        resolution = resolution
        zresolution = zresolution
        scales = (resolution, resolution, zresolution)
        return scales


    def create_neuroglancer(self):
        """create the Seung lab cloud volume format from the image stack
        For a large isotropic data set, Allen uses chunks = [128,128,128]
        self.input and self.output are defined in the pipeline_process
        """
        image_manager = ImageManager(self.input)

        if self.downsample:
            chunks = [image_manager.height, image_manager.width, 1]
        else:
            chunks = [image_manager.height//16, image_manager.width//16, 1] # 1796x984


        #test_dir(self.animal, self.input, self.section_count, self.downsample, same_size=True)
        scales = self.get_scales()
        scales = tuple(int(s*1000) for s in scales) # convert from microns to nanometers for neuroglancer
        print(f'scales={scales} scaling_factor={self.scaling_factor} downsample={self.downsample}')
        num_channels = image_manager.num_channels
        # neuroglancer does not support boolean dtype
        dtype = image_manager.dtype
        if dtype == 'bool':
            dtype = 'uint8'
        print(f'volume_size={image_manager.volume_size} ndim={image_manager.ndim} dtype={dtype} num_channels={num_channels}')
        print(f'Creating initial pretransfer data with chunks={chunks}')
        
        ng = NumpyToNeuroglancer(
            self.animal,
            None,
            scales,
            "image",
            dtype,
            num_channels=num_channels,
            chunk_size=chunks,
        )
        
        ng.init_precomputed(self.rechunkme_path, image_manager.volume_size)
        file_keys = []
        for i, f in enumerate(image_manager.files):
            filepath = os.path.join(self.input, f)
            file_keys.append([i, filepath, self.progress_dir, False, 0, 0]) #added is_blank, height, width

        workers = self.get_nworkers()
        if self.debug:
            for file_key in file_keys:
                ng.process_image(file_key=file_key)
        else:
            self.run_commands_concurrently(ng.process_image, file_keys, workers)
        ng.precomputed_vol.cache.flush()


    def create_downsamples(self):
        """Downsamples the neuroglancer cloudvolume this step is needed to make the files viewable in neuroglancer
        """

        image_manager = ImageManager(self.input)

        if image_manager.len_files < Z_CHUNK:
            z_chunk = image_manager.len_files
        else:
            z_chunk = Z_CHUNK
            
        chunks = [XY_CHUNK, XY_CHUNK, z_chunk]

        scales, resolutions = self.compute_mipmaps(self.get_scales())
        mips = len(scales) - 1  # number of downsampled levels to create (excluding the original)
        outpath = f"file://{self.output}"
        if not os.path.exists(self.rechunkme_path):
            print(f"DIR {self.rechunkme_path} does not exist, exiting.")
            sys.exit()
        cloud_volume_version = cloudvolume.__version__
        # hard coding to sharded = false
        if image_manager.num_channels > 2 or Version(cloud_volume_version) >= Version('9.0.0'):
            sharded = False
        else:
            sharded = True
        
        cloudpath = f"file://{self.rechunkme_path}"
        self.fileLogger.logevent(f"Input DIR: {self.rechunkme_path}")
        self.fileLogger.logevent(f"Output DIR: {self.output}")
        workers =self.get_nworkers()
        tq = LocalTaskQueue(parallel=workers)

        print(f'Writing transfer task with {workers} workers')
        # I have been having trouble with newer versions of cloud volume and the sharded transfer tasks.  

        if sharded:
            task = tc.create_image_shard_transfer_tasks(cloudpath, outpath, mip=0, chunk_size=chunks)
        else:
            task = tc.create_transfer_tasks(cloudpath, dest_layer_path=outpath, max_mips=mips, chunk_size=chunks, mip=0, skip_downsamples=True)

        print(f'Creating transfer task with chunks={chunks} to layer {outpath} sharded={sharded}')

        tq.insert(task)
        tq.execute()
        print('Finished transfer task')

        for mip in range(0, mips):
            factor = scales[mip]
            resolution = resolutions[mip]
            cv = CloudVolume(outpath, mip)
            print(f'Creating downsample task at mip={mip} factor={factor} with chunks={chunks} resolution = {resolution} sharded={sharded}')
            
            if sharded:
                task = tc.create_image_shard_downsample_tasks(cv.layer_cloudpath, mip=mip, chunk_size=chunks, factor=factor)
            else:
                task = tc.create_downsampling_tasks(cv.layer_cloudpath, mip=mip, num_mips=1, compress=True, factor=factor)
            
            tq.insert(task)            
            tq.execute()

    @staticmethod
    def compute_mipmaps(base_resolution, max_voxel_size=512.0):
        num_mips=100
        base_resolution = np.array(base_resolution, dtype=float)

        scales = [(2,2,1)]

        current_res = base_resolution.copy()

        for mip in range(1, num_mips):
            # --- Continuous anisotropy correction ---
            # Normalize by smallest voxel dimension
            min_res = np.min(current_res)
            ratios = current_res / min_res

            # Smooth scaling: more aggressive for finer axes
            # Use inverse ratio to bias scaling
            inv_ratios = 1.0 / ratios

            # Normalize to [1, 2] range
            scale = 1.0 + inv_ratios
            scale = np.clip(scale, 1.0, 2.0)

            # Round to nearest integer (Neuroglancer prefers ints)
            scale = np.round(scale).astype(int)

            # Ensure at least 1x scaling
            scale = np.maximum(scale, 1)

            # --- Apply scaling ---
            new_res = current_res * scale

            # --- Stop if exceeding max voxel size ---
            if np.any(new_res > max_voxel_size):
                print(f"Stopping at mip {mip}: exceeded max voxel size")
                break

            # Store results
            scales.append(scale)
            current_res = new_res

        scales = [tuple(int(x) for x in s) for s in scales]
        resolutions = []
        x,y,z = base_resolution
        for mip, scale in enumerate(zip(scales)):
            if mip == 0:
                x = x * scales[mip][0]
                y = y * scales[mip][1]
                z = z * scales[mip][2]
                resolution = [x, y, z]
            else:
                x = resolutions[mip-1][0] * scales[mip][0]
                y = resolutions[mip-1][1] * scales[mip][1]
                z = resolutions[mip-1][2] * scales[mip][2]

            resolution = [float(x), float(y), float(z)]
            resolutions.append(resolution)

        return scales, resolutions
