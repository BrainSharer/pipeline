import os
import glob
import sys
from pathlib import Path, PurePath

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

        if self.debug:
            print(f"DEBUG: START TiffExtractor::extract_tiffs_from_czi")

        if self.downsample:
            self.output = self.fileLocationManager.thumbnail_original
            self.checksum = os.path.join(self.fileLocationManager.www, 'checksums', 'thumbnail_original')
            scale_factor = DOWNSCALING_FACTOR
        else:
            self.output = self.fileLocationManager.tif
            self.checksum = os.path.join(self.fileLocationManager.www, 'checksums', 'full')
            scale_factor = 1

        self.input = self.fileLocationManager.get_czi()
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.checksum, exist_ok=True)
        starting_files = glob.glob(
            os.path.join(self.output, "*_C" + str(self.channel) + ".tif")
        )
        total_files = os.listdir(self.output)
        self.logevent(f"TIFF EXTRACTION FOR CHANNEL: {self.channel}")
        self.logevent(f"Output FOLDER: {self.output}")
        self.logevent(f"FILE COUNT [FOR CHANNEL {self.channel}]: {len(starting_files)}")
        self.logevent(f"TOTAL FILE COUNT [FOR DIRECTORY]: {len(total_files)}")

        sections = self.sqlController.get_sections(self.animal, self.channel)
        if len(sections) == 0:
            print('\nError, no sections found, exiting.')
            print("Were the CZI file names correct on birdstore?")
            print("File names should be in the format: DK123_slideXXX_anything.czi")
            sys.exit()

        file_keys = [] # czi_file, output_path, scenei, channel=1, scale=1
        for section in sections:
            czi_file = os.path.join(self.input, section.czi_file)
            tif_file = os.path.basename(section.file_name)
            output_path = os.path.join(self.output, tif_file)

            # CREATE .sha256 CHECKSUM FILENAME
            sha256_filename = Path(section.file_name).with_suffix('.sha256').name
            checksum_filepath = Path(self.checksum, sha256_filename)

            if not os.path.exists(czi_file):
                continue
            if os.path.exists(output_path):
                continue
            if self.debug:
                print(f'creating image={output_path}')
            scene = section.scene_index
            
            file_keys.append([czi_file, output_path, checksum_filepath, scene, self.channel, scale_factor, self.debug])
        if self.debug:
            print(f'Extracting a total of {len(file_keys)} files.')
            workers = 1
        else:
            workers = self.get_nworkers()
        self.run_commands_with_threads(extract_tiff_from_czi, file_keys, workers)


    def create_web_friendly_image(self):
        """Create downsampled version of full size tiff images that can be 
        viewed on the Django admin portal.
        These images are used for Quality Control.
        """
        if self.debug:
            print(f"DEBUG: START TiffExtractor::create_web_friendly_image")

        self.checksum = os.path.join(self.fileLocationManager.www, 'checksums', 'scene')

        self.input = self.fileLocationManager.get_czi()
        self.output = self.fileLocationManager.thumbnail_web
        channel = 1
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.checksum, exist_ok=True)

        sections = self.sqlController.get_sections(self.animal, channel)
        self.logevent(f"SINGLE (FIRST) CHANNEL ONLY - SECTIONS: {len(sections)}")
        self.logevent(f"Output FOLDER: {self.output}")

        file_keys = []
        files_skipped = 0
        for i, section in enumerate(sections):
            infile = os.path.join(self.input, section.czi_file)
            outfile = os.path.basename(section.file_name)
            output_path = os.path.join(self.output, outfile)
            outfile = output_path[:-5] + "1.png"  # force "C1" in filename
            if os.path.exists(outfile):
                files_skipped += 1
                continue
            scene = section.scene_index

            scale = 0.01

            # CREATE .sha256 CHECKSUM FILENAME
            sha256_filename = Path(outfile).with_suffix('.sha256').name
            checksum_filepath = Path(self.checksum, sha256_filename)
            
            file_keys.append([i, infile, outfile, checksum_filepath, scene, scale])
            
        if files_skipped > 0:
            self.logevent(f"SKIPPED [PRE-EXISTING] FILES: {files_skipped}")

        n_processing_elements = len(file_keys)
        self.logevent(f"PROCESSING [NOT PRE-EXISTING] FILES: {n_processing_elements}")

        workers = self.get_nworkers()
        self.run_commands_concurrently(extract_png_from_czi, file_keys, workers)