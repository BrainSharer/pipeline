import os
import sys
from cloudvolume import CloudVolume
from taskqueue import LocalTaskQueue
import igneous.task_creation as tc


from library.image_manipulation.neuroglancer_manager import NumpyToNeuroglancer
from library.utilities.utilities_process import SCALING_FACTOR, test_dir
from library.image_manipulation.image_manager import ImageManager
XY_CHUNK = 128
Z_CHUNK = 64


class NgPrecomputedMaker:
    """Class to convert a tiff image stack to the precomputed
    neuroglancer format code from Seung lab
    """

    def get_scales(self):
        """returns the scanning resolution for a given animal.  
        The scan resolution and sectioning thickness are retrived from the database.
        The resolution in the database is stored as micrometers (microns -um). But
        neuroglancer wants nanometers so we multipy by 1000

        :returns: list of converstion factors from pixel to micron for x,y and z
        """
        db_resolution = self.sqlController.scan_run.resolution
        zresolution = self.sqlController.scan_run.zresolution
        #28-AUG-2024 mod [self.scaling_factor was not defined]
        # if self.scaling_factor < SCALING_FACTOR: 
        #     zresolution *= self.scaling_factor
        resolution = int(db_resolution * 1000) 
        if self.downsample:
        #   resolution = int(db_resolution * 1000 * self.scaling_factor)
          resolution = int(db_resolution * 1000 * SCALING_FACTOR) #28-AUG-2024 mod [self.scaling_factor was not defined]
          
          self.mips = 4
        else:
            self.mips = 8
 
        scales = (resolution, resolution, int(zresolution * 1000))
        return scales


    def create_neuroglancer(self):
        """create the Seung lab cloud volume format from the image stack
        For a large isotropic data set, Allen uses chunks = [128,128,128]
        self.input and self.output are defined in the pipeline_process
        """

        if self.downsample:
            xy_chunk = int(XY_CHUNK//2)
            chunks = [xy_chunk, xy_chunk, 1]
        else:
            chunks = [XY_CHUNK, XY_CHUNK, 1]


        files, nfiles, *_ = test_dir(self.animal, self.input, self.section_count, self.downsample, same_size=True)

        self.fileLogger.logevent(f"self.input FOLDER: {self.input}")
        self.fileLogger.logevent(f"CURRENT FILE COUNT: {nfiles}")
        self.fileLogger.logevent(f"Output FOLDER: {self.output}")
        
        image_manager = ImageManager(self.input)
        scales = self.get_scales()
        self.fileLogger.logevent(f"CHUNK SIZE: {chunks}; SCALES: {scales}")
        print(f'volume_size={image_manager.volume_size} ndim={image_manager.ndim} dtype={image_manager.dtype}')
            
        ng = NumpyToNeuroglancer(
            self.animal,
            None,
            scales,
            "image",
            image_manager.dtype,
            num_channels=image_manager.num_channels,
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
                print(file_key)
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
            xy_chunk = int(XY_CHUNK//2)
            chunks = [xy_chunk, xy_chunk, xy_chunk]
        if not self.downsample and self.section_count < 100:
            z_chunk = int(XY_CHUNK)//2
            chunks = [XY_CHUNK, XY_CHUNK, z_chunk]

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

        if image_manager.num_channels == 3:
            print(f'Creating non-sharded transfer tasks with chunks={chunks} and section count={self.section_count}')
            tasks = tc.create_transfer_tasks(cloudpath, dest_layer_path=outpath, max_mips=self.mips, chunk_size=chunks, mip=0, skip_downsamples=True)
        else:
            print(f'Creating sharded transfer tasks with chunks={chunks} and section count={self.section_count}')
            tasks = tc.create_image_shard_transfer_tasks(cloudpath, outpath, mip=0, chunk_size=chunks)

        tq.insert(tasks)
        tq.execute()
        print('Finished transfer tasks')

        for mip in range(0, self.mips):
            cv = CloudVolume(outpath, mip)
            print(f'Creating downsample tasks at mip={mip}')
            if image_manager.num_channels == 3:
                tasks = tc.create_downsampling_tasks(cv.layer_cloudpath, mip=mip, num_mips=1, compress=True)
            else:
                tasks = tc.create_image_shard_downsample_tasks(cv.layer_cloudpath, mip=mip, chunk_size=chunks)
            tq.insert(tasks)
            tq.execute()

    def create_neuroglancer_stack(self, input_path, rechunkme_path, output, progress_dir):
        """
        Downsamples the neuroglancer cloudvolume this step is needed to make the files viewable in neuroglancer
        """

        if self.debug:
            print(f"DEBUG: START NgPrecomputedMaker::create_neuroglancer_stack")

        if self.downsample:
            xy_chunk = int(XY_CHUNK // 2)
            chunks = [xy_chunk, xy_chunk, 1]
        else:
            chunks = [XY_CHUNK, XY_CHUNK, 1]

        os.makedirs(rechunkme_path, exist_ok=True)
        os.makedirs(progress_dir, exist_ok=True) #PROGRESS MAY STAY ON NFS

        starting_files, *_ = test_dir(self.animal, input_path, self.section_count, self.downsample, same_size=True)
        self.fileLogger.logevent(f"input_path FOLDER: {input_path}")
        self.fileLogger.logevent(f"CURRENT FILE COUNT: {len(starting_files)}")
        self.fileLogger.logevent(f"Output/staging FOLDER: {output}")

        image_manager = ImageManager(input_path)
        scales = self.get_scales()
        self.fileLogger.logevent(f"CHUNK SIZE: {chunks}; SCALES: {scales}")
        print(f'volume_size={image_manager.volume_size} ndim={image_manager.ndim} dtype={image_manager.dtype}')

        ####################################################### 'RECHUNK' (NON-CHUNKED: FULL RESOLUTION: MIP=0) START
        ng = NumpyToNeuroglancer(
            self.animal,
            None,
            scales,
            "image",
            image_manager.dtype,
            num_channels=image_manager.num_channels,
            chunk_size=chunks,
        )
        
        ng.init_precomputed(rechunkme_path, image_manager.volume_size)
        file_keys = []
        orientation = self.sqlController.histology.orientation
        for i, f in enumerate(image_manager.files):
            filepath = os.path.join(input_path, f)
            file_keys.append([i, filepath, orientation, progress_dir])

        workers = self.get_nworkers()
        if self.debug:
            for file_key in file_keys:
                print(file_key)
                ng.process_image(file_key=file_key)
        else:
            self.run_commands_concurrently(ng.process_image, file_keys, workers)
        ng.precomputed_vol.cache.flush()
        ####################################################### 'RECHUNK' END
        chunks = [XY_CHUNK, XY_CHUNK, Z_CHUNK]
        if self.downsample:
            xy_chunk = int(XY_CHUNK // 2)
            chunks = [xy_chunk, xy_chunk, xy_chunk]
        if not self.downsample and self.section_count < 100:
            z_chunk = int(XY_CHUNK) // 2
            chunks = [XY_CHUNK, XY_CHUNK, z_chunk]

        if os.path.exists(output):
            print(f"DIR {output} already exists. Downsampling has already been performed.")
            return
        else:
            os.makedirs(output, exist_ok=True)
        outpath = f"file://{output}"
        if not os.path.exists(rechunkme_path):
            print(f"DIR {rechunkme_path} does not exist, exiting.")
            sys.exit()
        cloudpath = f"file://{rechunkme_path}"
        self.fileLogger.logevent(f"Input DIR: {rechunkme_path}")
        self.fileLogger.logevent(f"Output DIR: {output}")
        workers = self.get_nworkers()
        
        tq = LocalTaskQueue(parallel=workers)
        if image_manager.num_channels == 3:
            print(f'Creating non-sharded transfer tasks with chunks={chunks} and section count={self.section_count}')
            tasks = tc.create_transfer_tasks(cloudpath, dest_layer_path=outpath, max_mips=self.mips, chunk_size=chunks, mip=0, skip_downsamples=True)
        else:
            print(f'Creating sharded transfer tasks with chunks={chunks} and section count={self.section_count}')
            tasks = tc.create_image_shard_transfer_tasks(cloudpath, outpath, mip=0, chunk_size=chunks)
        tq.insert(tasks)
        tq.execute()
        
        print('Finished transfer tasks')
        
        for mip in range(0, self.mips):
            cv = CloudVolume(outpath, mip)
            print(f'Creating downsample tasks at mip={mip}')
            if image_manager.num_channels == 3:
                tasks = tc.create_downsampling_tasks(cv.layer_cloudpath, mip=mip, num_mips=1, compress=True)
            else:
                tasks = tc.create_image_shard_downsample_tasks(cv.layer_cloudpath, mip=mip, chunk_size=chunks)
            tq.insert(tasks)
            tq.execute()


