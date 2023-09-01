import os
import glob
import sys

from library.image_manipulation.parallel_manager import ParallelManager
from library.image_manipulation.czi_manager import extract_tiff_from_czi, extract_png_from_czi
from library.utilities.utilities_process import DOWNSCALING_FACTOR


class TiffExtractor(ParallelManager):
    """Includes methods to extract tiff images from czi source files and generate png files for quick viewing of
    downsampled images in stack

    """

    def extract_tiffs_from_czi(self):
        """
        This method will:
            1. Fetch the meta information of each slide and czi files from the database
            2. Extract the images from the czi file and store them as tiff format.
            3. Then updates the database with meta information about the sections in each slide
        
        :param animal: the prep id of the animal
        :param channel: the channel of the stack image to process
        :param compression: Compression used to store the tiff files default is LZW compression
        """

        if self.downsample:
            OUTPUT = self.fileLocationManager.thumbnail_original
            scale_factor = DOWNSCALING_FACTOR
        else:
            OUTPUT = self.fileLocationManager.tif
            scale_factor = 1

        INPUT = self.fileLocationManager.get_czi(self.rescan_number)
        os.makedirs(OUTPUT, exist_ok=True)
        starting_files = glob.glob(
            os.path.join(OUTPUT, "*_C" + str(self.channel) + ".tif")
        )
        total_files = os.listdir(OUTPUT)
        self.logevent(f"TIFF EXTRACTION FOR CHANNEL: {self.channel}")
        self.logevent(f"OUTPUT FOLDER: {OUTPUT}")
        self.logevent(f"FILE COUNT [FOR CHANNEL {self.channel}]: {len(starting_files)}")
        self.logevent(f"TOTAL FILE COUNT [FOR DIRECTORY]: {len(total_files)}")

        sections = self.sqlController.get_sections(self.animal, self.channel, self.rescan_number)
        if len(sections) == 0:
            print('\nError, no sections found, exiting.')
            sys.exit()

        file_keys = [] # czi_file, output_path, scenei, channel=1, scale=1
        for section in sections:
            czi_file = os.path.join(INPUT, section.czi_file)
            tif_file = os.path.basename(section.file_name)
            output_path = os.path.join(OUTPUT, tif_file)
            if self.debug:
                print(f'creating thumbnail={output_path}')
            if not os.path.exists(czi_file):
                continue
            if os.path.exists(output_path):
                continue
            scene = section.scene_index
            file_keys.append([czi_file, output_path, scene, self.channel, scale_factor])
        if self.debug:
            print(f'Extracting a total of {len(file_keys)} thumbnails')
        workers = self.get_nworkers()
        self.run_commands_with_threads(extract_tiff_from_czi, file_keys, workers)


    def create_web_friendly_image(self):
        """Create downsampled version of full size tiff images that can be 
        viewed on the Django admin portal.
        These images are used for Quality Control.
        """

        INPUT = self.fileLocationManager.get_czi(self.rescan_number)
        OUTPUT = self.fileLocationManager.thumbnail_web
        channel = 1
        os.makedirs(OUTPUT, exist_ok=True)

        sections = self.sqlController.get_sections(self.animal, channel, self.rescan_number)
        self.logevent(f"SINGLE (FIRST) CHANNEL ONLY - SECTIONS: {len(sections)}")
        self.logevent(f"OUTPUT FOLDER: {OUTPUT}")

        file_keys = []
        files_skipped = 0
        for i, section in enumerate(sections):
            infile = os.path.join(INPUT, section.czi_file)
            outfile = os.path.basename(section.file_name)
            output_path = os.path.join(OUTPUT, outfile)
            outfile = output_path[:-5] + "1.png"  # force "C1" in filename
            if os.path.exists(outfile):
                files_skipped += 1
                continue
            scene = section.scene_index

            scale = 0.01
            file_keys.append([i, infile, outfile, scene, scale])

        if files_skipped > 0:
            print(f" SKIPPED [PRE-EXISTING] FILES: {files_skipped}", end=" ")
            self.logevent(f"SKIPPED [PRE-EXISTING] FILES: {files_skipped}")
        n_processing_elements = len(file_keys)
        self.logevent(f"PROCESSING [NOT PRE-EXISTING] FILES: {n_processing_elements}")

        workers = self.get_nworkers()
        self.run_commands_concurrently(extract_png_from_czi, file_keys, workers)

