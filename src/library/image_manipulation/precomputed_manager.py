import os
import sys
from cloudvolume import CloudVolume
from taskqueue.taskqueue import LocalTaskQueue
import igneous.task_creation as tc


from library.image_manipulation.neuroglancer_manager import NumpyToNeuroglancer
from library.utilities.utilities_process import test_dir
from library.image_manipulation.image_manager import ImageManager
XY_CHUNK = 128
Z_CHUNK = 64


class NgPrecomputedMaker:
    """Class to convert a tiff image stack to the precomputed
    neuroglancer format code from Seung lab
    """

    def __init__(self, sqlController, *args, **kwargs):
        self.sqlController = sqlController
        # Initialize other attributes as needed
        # Example:
        # self.input = kwargs.get('input', None)
        # self.output = kwargs.get('output', None)
        # self.animal = kwargs.get('animal', None)
        # self.section_count = kwargs.get('section_count', None)
        self.downsample = kwargs.get('downsample', False)
        self.scaling_factor = kwargs.get('scaling_factor', 1.0)
        # self.rechunkme_path = kwargs.get('rechunkme_path', None)
        # self.progress_dir = kwargs.get('progress_dir', None)
        # self.fileLogger = kwargs.get('fileLogger', None)
        # self.debug = kwargs.get('debug', False)

    def get_scales(self):
        """returns the scanning resolution for a given animal.  
        The scan resolution and sectioning thickness are retrived from the database.
        The resolution in the database is stored as micrometers (microns -um). But
        neuroglancer wants nanometers so we multipy by 1000

        :returns: list of converstion factors from pixel to micron for x,y and z
        """
        db_resolution = self.sqlController.scan_run.resolution
        zresolution = self.sqlController.scan_run.zresolution
        resolution = int(db_resolution * 1000) 
        if self.downsample:
          if zresolution < 20:
                zresolution = int(zresolution * 1000 * self.scaling_factor)
          resolution = int(db_resolution * 1000 * self.scaling_factor)          
 
        scales = (resolution, resolution, int(zresolution * 1000))
        return scales


    def create_neuroglancer(self):
        """create the Seung lab cloud volume format from the image stack
        For a large isotropic data set, Allen uses chunks = [128,128,128]
        self.input and self.output are defined in the pipeline_process
        """
        image_manager = ImageManager(self.input)

        if self.downsample:
            self.xy_chunk = int(XY_CHUNK//2)
            chunks = [self.xy_chunk, self.xy_chunk, 1]
        else:
            chunks = [image_manager.height//16, image_manager.width//16, 1] # 1796x984

        test_dir(self.animal, self.input, self.section_count, self.downsample, same_size=True)
        scales = self.get_scales()
        print(f'scales={scales} scaling_factor={self.scaling_factor} downsample={self.downsample}')
        num_channels = image_manager.num_channels
        print(image_manager.width, image_manager.height, image_manager.len_files, image_manager.shape)
        print(f'volume_size={image_manager.volume_size} ndim={image_manager.ndim} dtype={image_manager.dtype} num_channels={num_channels} and size={image_manager.size}')
        ng = NumpyToNeuroglancer(
            self.animal,
            None,
            scales,
            "image",
            image_manager.dtype,
            num_channels=num_channels,
            chunk_size=chunks,
        )
        
        ng.init_precomputed(self.rechunkme_path, image_manager.volume_size)
        file_keys = []
        orientation = self.sqlController.histology.orientation
        for i, f in enumerate(image_manager.files):
            filepath = os.path.join(self.input, f)
            file_keys.append([i, filepath, orientation, self.progress_dir])

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

        chunks = [XY_CHUNK, XY_CHUNK, Z_CHUNK]
        if self.downsample:
            chunks = [self.xy_chunk, self.xy_chunk, self.xy_chunk]

        if not self.downsample and self.section_count < 100:
            z_chunk = int(XY_CHUNK)//2
            chunks = [XY_CHUNK, XY_CHUNK, z_chunk]

        if self.downsample or image_manager.size < 100000000:
            mips = 4
        else:
            mips = 7

        if os.path.exists(self.output):
            print(f"DIR {self.output} already exists. Downsampling has already been performed.")
            return
        outpath = f"file://{self.output}"
        if not os.path.exists(self.rechunkme_path):
            print(f"DIR {self.rechunkme_path} does not exist, exiting.")
            sys.exit()
        cloudpath = f"file://{self.rechunkme_path}"
        self.fileLogger.logevent(f"Input DIR: {self.rechunkme_path}")
        self.fileLogger.logevent(f"Output DIR: {self.output}")
        workers =self.get_nworkers()

        tq = LocalTaskQueue(parallel=workers)

        if image_manager.num_channels > 2:
            print(f'Creating non-sharded transfer tasks with chunks={chunks} and section count={self.section_count}')
            tasks = tc.create_transfer_tasks(cloudpath, dest_layer_path=outpath, max_mips=mips, chunk_size=chunks, mip=0, skip_downsamples=True)
        else:
            print(f'Creating sharded transfer tasks with chunks={chunks} and section count={self.section_count}')
            tasks = tc.create_image_shard_transfer_tasks(cloudpath, outpath, mip=0, chunk_size=chunks)

        tq.insert(tasks)
        tq.execute()
        print('Finished transfer tasks')

        for mip in range(0, mips):
            cv = CloudVolume(outpath, mip)
            if image_manager.num_channels > 2:
                print(f'Creating downsample tasks at mip={mip}')
                tasks = tc.create_downsampling_tasks(cv.layer_cloudpath, mip=mip, num_mips=1, compress=True)
            else:
                print(f'Creating sharded downsample tasks at mip={mip}')
                tasks = tc.create_image_shard_downsample_tasks(cv.layer_cloudpath, mip=mip, chunk_size=chunks)
            tq.insert(tasks)
            tq.execute()
