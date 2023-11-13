import os
from skimage import io
import sys
from cloudvolume import CloudVolume
from taskqueue import LocalTaskQueue
import igneous.task_creation as tc


from library.image_manipulation.neuroglancer_manager import NumpyToNeuroglancer, calculate_chunks, \
    calculate_factors
from library.utilities.utilities_mask import normalize16
from library.utilities.utilities_process import SCALING_FACTOR, test_dir
XY_CHUNK = 128


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
        resolution = int(db_resolution * 1000) 
        if self.downsample:
          resolution = int(db_resolution * 1000 * SCALING_FACTOR)
 
        scales = (resolution, resolution, int(zresolution * 1000))
        return scales

    def get_file_information(self, INPUT, PROGRESS_DIR):
        """getting the information of files in the directory

        Args:
            INPUT (str): path to input directory

        Returns:
            str: name of the tif images corresponding to the section in the middle of the stack
            list: list of id and filename tuples for the files in the directory
            tuple: tuple of integers for the width,height and number of sections in the stack
            int: number of channels present in each tif files
        """
        files = sorted(os.listdir(INPUT))
        midpoint = len(files) // 2
        midfilepath = os.path.join(INPUT, files[midpoint])
        midfile = io.imread(midfilepath, img_num=0)
        height = midfile.shape[0]
        width = midfile.shape[1]
        num_channels = midfile.shape[2] if len(midfile.shape) > 2 else 1
        file_keys = []
        volume_size = (width, height, len(files))
        orientation = self.sqlController.histology.orientation
        for i, f in enumerate(files):
            filepath = os.path.join(INPUT, f)
            file_keys.append([i, filepath, orientation, PROGRESS_DIR])
        return midfile, file_keys, volume_size, num_channels

    def create_neuroglancer(self):
        """create the Seung lab cloud volume format from the image stack
        For a large isotropic data set, Allen uses chunks = [128,128,128]
        """

        if self.downsample:
            xy_chunk = int(XY_CHUNK//2)
            chunks = [xy_chunk, xy_chunk, 1]
            INPUT = self.fileLocationManager.get_thumbnail_aligned(channel=self.channel)
        else:
            chunks = [XY_CHUNK, XY_CHUNK, 1]
            INPUT = self.fileLocationManager.get_full_aligned(channel=self.channel)

        OUTPUT_DIR = self.fileLocationManager.get_neuroglancer(self.downsample, self.channel)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        PROGRESS_DIR = self.fileLocationManager.get_neuroglancer_progress(self.downsample, self.channel)
        os.makedirs(PROGRESS_DIR, exist_ok=True)

        """
        starting_files = test_dir(self.animal, INPUT, self.section_count, self.downsample, same_size=True)
        self.logevent(f"INPUT FOLDER: {INPUT}")
        self.logevent(f"CURRENT FILE COUNT: {starting_files}")
        self.logevent(f"OUTPUT FOLDER: {OUTPUT_DIR}")
        """
        midfile, file_keys, volume_size, num_channels = self.get_file_information(INPUT, PROGRESS_DIR)
        scales = self.get_scales()
        self.logevent(f"CHUNK SIZE: {chunks}; SCALES: {scales}")
        ng = NumpyToNeuroglancer(
            self.animal,
            None,
            scales,
            "image",
            midfile.dtype,
            num_channels=num_channels,
            chunk_size=chunks,
        )
        
        ng.init_precomputed(OUTPUT_DIR, volume_size)
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

        chunks = [XY_CHUNK, XY_CHUNK, 1]
        mips = 8
        if self.downsample:
            xy_chunk = int(XY_CHUNK//2)
            chunks = [xy_chunk, xy_chunk, 1]
            mips = 4
        
        OUTPUT_DIR = self.fileLocationManager.get_neuroglancer(self.downsample, self.channel, rechunk=True)
        if os.path.exists(OUTPUT_DIR):
            print(f"DIR {OUTPUT_DIR} already exists and not performing any downsampling.")
            return
        outpath = f"file://{OUTPUT_DIR}"
        INPUT_DIR = self.fileLocationManager.get_neuroglancer( self.downsample, self.channel )
        if not os.path.exists(INPUT_DIR):
            print(f"DIR {INPUT_DIR} does not exist, exiting.")
            sys.exit()
        cloudpath = f"file://{INPUT_DIR}"
        self.logevent(f"INPUT_DIR: {INPUT_DIR}")
        self.logevent(f"OUTPUT_DIR: {OUTPUT_DIR}")
        workers =self.get_nworkers()
        if self.downsample:
            chunks[2] = xy_chunk
        else:
            chunks[2] = XY_CHUNK


        tq = LocalTaskQueue(parallel=workers)
        #if self.section_count < 100:
        #    chunks = []

        shard = True
        if shard:
            print(f'Creating sharded transfer transfer tasks with chunks={chunks}')
            tasks = tc.create_image_shard_transfer_tasks(cloudpath, dst_layer_path=outpath, chunk_size=chunks, mip=0, fill_missing=True)
            tq.insert(tasks)
            tq.execute()

            cv = CloudVolume(outpath)
            for mip in range(0, mips):
                print(f'Creating downsampled shards at mip={mip}')
                tasks = tc.create_image_shard_downsample_tasks(cloudpath=cv.layer_cloudpath, mip=mip, fill_missing=True)
                tq.insert(tasks)
                tq.execute()
        else:
            print(f'Creating transfer tasks with chunks={chunks} and section count={self.section_count}')
            tasks = tc.create_transfer_tasks(cloudpath, dest_layer_path=outpath, chunk_size=chunks, mip=0, skip_downsamples=True)
            tq.insert(tasks)
            tq.execute()
            print('Finished transfer tasks')
            print(f'Creating downsample tasks with mips={mips} and chunks={chunks}')
            cv = CloudVolume(outpath)
            tasks = tc.create_downsampling_tasks(
                cv.layer_cloudpath,
                num_mips=mips,
                compress=True,
            )
            tq.insert(tasks)
            tq.execute()
        

    def create_neuroglancer_normalization(self):
        """Downsamples the neuroglancer cloudvolume this step is needed to make the files viewable in neuroglancer"""
        workers =self.get_nworkers()
        mips = [0, 1, 2, 3, 4, 5, 6, 7]
        if self.downsample:
            mips = [0, 1, 2]
        OUTPUT_DIR = self.fileLocationManager.get_neuroglancer(self.downsample, self.channel, rechunck=True)
        outpath = f"file://{OUTPUT_DIR}"

        tq = LocalTaskQueue(parallel=workers)
        for mip in mips:
            # first pass: create per z-slice histogram
            cv = CloudVolume(outpath, mip)
            tasks = tc.create_luminance_levels_tasks(cv.layer_cloudpath, coverage_factor=0.01, mip=mip) 
            tq.insert(tasks)    
            tq.execute()
            # second pass: apply histogram equalization
            tasks = tc.create_contrast_normalization_tasks(cv.layer_cloudpath, cv.layer_cloudpath, shape=None, mip=mip, clip_fraction=0.05, fill_missing=False, translate=(0,0,0))
            tq.insert(tasks)    
            tq.execute()


