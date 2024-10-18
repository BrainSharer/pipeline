import os, glob
import hashlib
from lib.ParallelManager import ParallelManager
from lib.CZIManager import extract_tiff_from_czi, extract_png_from_czi
from utilities.utilities_process import DOWNSCALING_FACTOR


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
        Args:
            animal: the prep id of the animal
            channel: the channel of the stack image to process
            compression: Compression used to store the tiff files default is LZW compression

        Returns:
            nothing
        """

        if self.downsample:
            OUTPUT = self.fileLocationManager.thumbnail_original
            scale_factor = DOWNSCALING_FACTOR
        else:
            OUTPUT = self.fileLocationManager.tif
            scale_factor = 1

        INPUT = self.fileLocationManager.czi
        os.makedirs(OUTPUT, exist_ok=True)
        starting_files = glob.glob(
            os.path.join(OUTPUT, "*_C" + str(self.channel) + ".tif")
        )
        total_files = os.listdir(OUTPUT)
        self.fileLogger.logevent(f"TIFF EXTRACTION FOR CHANNEL: {self.channel}")
        self.fileLogger.logevent(f"OUTPUT FOLDER: {OUTPUT}")
        self.fileLogger.logevent(f"FILE COUNT [FOR CHANNEL {self.channel}]: {len(starting_files)}")
        self.fileLogger.logevent(f"TOTAL FILE COUNT [FOR DIRECTORY]: {len(total_files)}")

        sections = self.sqlController.get_distinct_section_filenames(
            self.animal, self.channel
        )

        file_keys = [] # czi_file, output_path, scenei, channel=1, scale=1
        for section in sections:
            czi_file = os.path.join(INPUT, section.czi_file)
            tif_file = os.path.basename(section.file_name)
            output_path = os.path.join(OUTPUT, tif_file)
            if not os.path.exists(czi_file):
                continue
            if os.path.exists(output_path):
                continue
            scenei = section.scene_number - 1
            file_keys.append([czi_file, output_path, scenei, self.channel, scale_factor])

        workers = self.get_nworkers()
        self.run_commands_concurrently(extract_tiff_from_czi, file_keys, workers)

        self.update_database()

    def update_database(self):
        """Updating the file log table in the database about the completion of the QC preparation steps"""
        self.sqlController.set_task(
            self.animal, self.progress_lookup.QC_IS_DONE_ON_SLIDES_IN_WEB_ADMIN
        )
        self.sqlController.set_task(
            self.animal,
            self.progress_lookup.CZI_FILES_ARE_CONVERTED_INTO_NUMBERED_TIFS_FOR_CHANNEL_1,
        )

    def calc_filechecksum(self, filename, channel):
        sha256_hash = hashlib.sha256()

        if filename.find("_") != -1:
            path_to_full_resolution_tiff = os.path.join(
                self.fileLocationManager.tif, filename
            )
        else:
            path_to_full_resolution_tiff = os.path.join(
                self.fileLocationManager.prep, "CH" + str(channel), filename
            )
        with open(path_to_full_resolution_tiff, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
            checksum = str(sha256_hash.hexdigest())
        print(
            sha256_hash.hexdigest(),
            "\t",
            os.path.basename(path_to_full_resolution_tiff),
            "\n",
        )
        return checksum

    def create_web_friendly_image(self):
        """Create downsampled version of full size tiff images that can be viewed on the Django admin portal
        These images are used for Quality Control
        ONLY 1 CHANNEL CREATED FOR .png FILES
        """
        INPUT = self.fileLocationManager.czi
        OUTPUT = self.fileLocationManager.thumbnail_web
        channel = 1
        os.makedirs(OUTPUT, exist_ok=True)

        sections = self.sqlController.get_distinct_section_filenames(
            self.animal, channel
        )
        self.fileLogger.logevent(f"SINGLE (FIRST) CHANNEL ONLY - SECTIONS: {len(sections)}")
        self.fileLogger.logevent(f"OUTPUT FOLDER: {OUTPUT}")

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
            scene_index = section.scene_number - 1
            scale = 0.01
            file_keys.append([i, infile, outfile, scene_index, scale])

        if files_skipped > 0:
            print(f" SKIPPED [PRE-EXISTING] FILES: {files_skipped}", end=" ")
            self.fileLogger.logevent(f"SKIPPED [PRE-EXISTING] FILES: {files_skipped}")
        n_processing_elements = len(file_keys)
        self.fileLogger.logevent(f"PROCESSING [NOT PRE-EXISTING] FILES: {n_processing_elements}")

        workers = self.get_nworkers()
        self.run_commands_concurrently(extract_png_from_czi, file_keys, workers)

