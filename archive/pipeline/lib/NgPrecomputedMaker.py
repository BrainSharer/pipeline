import os
from skimage import io
from utilities.utilities_cvat_neuroglancer import NumpyToNeuroglancer, calculate_chunks
from utilities.utilities_process import SCALING_FACTOR, test_dir


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
        """create the Seung lab cloud volume format from the image stack"""
        progress_id = self.sqlController.get_progress_id(self.downsample, self.channel, "NEUROGLANCER")

        if self.downsample:
            INPUT = self.fileLocationManager.get_thumbnail_aligned(channel=self.channel)
        if not self.downsample:
            INPUT = self.fileLocationManager.get_full_aligned(channel=self.channel)
            self.sqlController.set_task(self.animal, progress_id)

        OUTPUT_DIR = self.fileLocationManager.get_neuroglancer(self.downsample, self.channel)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        PROGRESS_DIR = self.fileLocationManager.get_neuroglancer_progress(self.downsample, self.channel)
        os.makedirs(PROGRESS_DIR, exist_ok=True)

        starting_files = test_dir(self.animal, INPUT, self.section_count, self.downsample, same_size=True)
        self.fileLogger.logevent(f"INPUT FOLDER: {INPUT}")
        self.fileLogger.logevent(f"CURRENT FILE COUNT: {starting_files}")
        self.fileLogger.logevent(f"OUTPUT FOLDER: {OUTPUT_DIR}")

        midfile, file_keys, volume_size, num_channels = self.get_file_information(INPUT, PROGRESS_DIR)
        chunks = calculate_chunks(self.downsample, -1)
        scales = self.get_scales()
        self.fileLogger.logevent(f"CHUNK SIZE: {chunks}; SCALES: {scales}")
        ng = NumpyToNeuroglancer(
            self.animal,
            None,
            scales,
            "image",
            midfile.dtype,
            num_channels=num_channels,
            chunk_size=chunks,
        )
        
        ng.init_precomputed(OUTPUT_DIR, volume_size, progress_id=progress_id)
        workers = self.get_nworkers()
        self.run_commands_concurrently(ng.process_image, file_keys, workers)
        ng.precomputed_vol.cache.flush()
