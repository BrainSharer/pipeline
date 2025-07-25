import os
import inspect
import glob
import sys
from pathlib import Path
import hashlib
from tqdm import tqdm

from library.image_manipulation.czi_manager import extract_tiff_from_czi, extract_png_from_czi
from library.utilities.utilities_process import DOWNSCALING_FACTOR


class TiffExtractor():
    """Includes methods to extract tiff images from czi source files and generate png files for quick viewing of
    downsampled images in stack

    """

    def extract_tiffs_from_czi(self):
        """
        Extracts TIFF images from CZI files for a specified channel.
        This method performs the following steps:
        1. Determines the output directory and scale factor based on the downsample flag.
        2. Creates the output directory if it does not exist.
        3. Logs the initial state including the number of existing TIFF files in the output directory.
        4. Retrieves sections from the database for the specified animal and channel.
        5. For each section, extracts the TIFF image from the corresponding CZI file if it does not already exist.
        6. Logs and prints errors if CZI files are missing or if no sections are found.
        7. Checks for duplicate files in the output directory and logs and prints any duplicates found.
        :raise: SystemExit: If no sections are found in the database or if duplicate files are found.

        Note, it cannot be run with threads.
        """

        if self.debug:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} START")

        if self.downsample:
            self.output = self.fileLocationManager.thumbnail_original
            scale_factor = DOWNSCALING_FACTOR
        else:
            # if str(self.channel) == '1':
            #     self.output = self.fileLocationManager.tif
            # else:

            scale_factor = 1

        self.input = self.fileLocationManager.get_czi()
        os.makedirs(self.output, exist_ok=True)
        starting_files = glob.glob(
            os.path.join(self.output, "*_C" + str(self.channel) + ".tif")
        )
        total_files = os.listdir(self.output)
        self.fileLogger.logevent(f"TIFF EXTRACTION FOR CHANNEL: {self.channel}")
        self.fileLogger.logevent(f"Output FOLDER: {self.output}")
        self.fileLogger.logevent(f"FILE COUNT [FOR CHANNEL {self.channel}]: {len(starting_files)}")
        self.fileLogger.logevent(f"TOTAL FILE COUNT [FOR DIRECTORY]: {len(total_files)}")

        sections = self.sqlController.get_sections(self.animal, self.channel, self.debug)
        if self.debug:
            print(f"DEBUG: DB SECTION COUNT: {len(sections)}")
            print(f"OUTPUT FILES DESTINATION: {self.output}")

        if len(sections) == 0:
            print('\nError, no sections found, exiting.')
            print("Were the CZI file names correct on birdstore?")
            print("File names should be in the format: DK123_slideXXX_anything.czi")
            print("Are there slides in the database but no tifs? Check the database for existing slides and missing tifs")
            sys.exit()

        for section in tqdm(sections, desc="Extracting TIFFs", disable=self.debug):
            czi_file = os.path.join(self.input, section.czi_file)
            tif_file = os.path.basename(section.file_name)
            outfile = os.path.join(self.output, tif_file)

            if not os.path.exists(czi_file):
                print(f'Error: {czi_file} does not exist.')
                continue
            if os.path.exists(outfile):
                continue
            scene = section.scene_index
            if self.debug:
                print(f"extracting from {os.path.basename(czi_file)}, {scene=}, to {outfile}")
            extract_tiff_from_czi([czi_file, outfile, scene, self.channel, scale_factor])
        
        # Check for duplicates
        duplicates = self.find_duplicates(self.fileLocationManager.thumbnail_original)
        if duplicates:
            self.fileLogger.logevent(f"DUPLICATE FILES FOUND: {duplicates}")
            print("\nDUPLICATE FILES FOUND:")
            for duplicate in duplicates:
                print()
                for file in duplicate:
                    print(f"{os.path.basename(file)}", end=" ")
            print("\n\nDuplicate files found, please fix. Exiting.")                                
            sys.exit()
                
            print()


    def create_web_friendly_image(self):
        """Create downsampled version of full size tiff images that can be 
        viewed on the Django admin portal.
        These images are used for Quality Control.
        """
        workers = self.get_nworkers()

        if self.debug:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} START")
            workers = 1

        self.checksum = os.path.join(self.fileLocationManager.www, 'checksums', 'scene')

        self.input = self.fileLocationManager.get_czi()
        self.output = self.fileLocationManager.thumbnail_web
        channel = 1
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.checksum, exist_ok=True)

        sections = self.sqlController.get_sections(self.animal, channel, self.debug)
        self.fileLogger.logevent(f"SINGLE (FIRST) CHANNEL ONLY - SECTIONS: {len(sections)}")
        self.fileLogger.logevent(f"Output FOLDER: {self.output}")

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
            self.fileLogger.logevent(f"SKIPPED [PRE-EXISTING] FILES: {files_skipped}")

        n_processing_elements = len(file_keys)
        self.fileLogger.logevent(f"PROCESSING [NOT PRE-EXISTING] FILES: {n_processing_elements}")

        self.run_commands_concurrently(extract_png_from_czi, file_keys, workers)

    def create_previews(self):
        workers = self.get_nworkers()
        if self.debug:
            workers = 1
        self.input = self.fileLocationManager.get_czi()
        czifiles = sorted(os.listdir(self.input))
        file_keys = []
        for czifile in czifiles:
            infile = os.path.join(self.input, czifile)
            infile = infile.replace(" ","_").strip()
            file_keys.append(infile)
        
        self.run_commands_with_threads(self.extract_slide_scene_data, file_keys, workers) #SLIDE PREVIEW

    def create_checksums(self):

        def create_checksum_dir(input_dir, output_dir):
            files = sorted(os.listdir(input_dir))
            os.makedirs(output_dir, exist_ok=True)
            for file in files:
                infile = os.path.join(input_dir, file)
                outfile = os.path.join(output_dir, file)
                outfile = Path(outfile).with_suffix('.sha256')
                if os.path.exists(outfile):
                    continue
                if self.debug:
                    print(f"Creating checksum at {outfile}")
                readable_hash = calculate_hash(infile)
                with open(outfile, 'w') as f:
                    f.write(readable_hash)

        def update_slide_checksum(input_dir, checksum_dir):
            files = sorted(os.listdir(input_dir))
            for file in files:
                infile = os.path.join(input_dir, file)
                checksumfile = os.path.join(checksum_dir, file)
                checksumfile = Path(checksumfile).with_suffix('.sha256')
                if not os.path.exists(infile):
                    print(f'Error: CZI {infile} does not exist.')
                    continue
                if not os.path.exists(checksumfile):
                    print(f'Error: checksum {checksumfile} does not exist.')
                    continue
                czifile = os.path.basename(infile)
                with open(checksumfile, 'r') as f:
                    update_dict = {'checksum': f.read()}

                self.sqlController.update_slide(czifile, update_dict)



        os.makedirs(self.checksum, exist_ok=True)
        
        thumbnail_inpath = self.fileLocationManager.thumbnail_original
        thumbnail_outpath = os.path.join(self.fileLocationManager.www, 'checksums', 'thumbnail_original')
        
        scene_inpath = self.fileLocationManager.thumbnail_web
        scene_outpath = os.path.join(self.fileLocationManager.www, 'checksums', 'scene')

        preview_inpath = os.path.join(self.fileLocationManager.www, 'slides_preview')
        preview_outpath = os.path.join(self.fileLocationManager.www, 'checksums', 'slides_preview')

        create_checksum_dir(thumbnail_inpath, thumbnail_outpath)
        create_checksum_dir(scene_inpath, scene_outpath)
        create_checksum_dir(preview_inpath, preview_outpath)

        update_slide_checksum(self.fileLocationManager.get_czi(), preview_outpath)
        
    @staticmethod
    def find_duplicates(directory):
        """Finds duplicate files in a directory."""
        files_by_hash = {}
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_hash = calculate_hash(file_path)
                files_by_hash.setdefault(file_hash, []).append(file_path)

        duplicates = [files for files in files_by_hash.values() if len(files) > 1]
        return duplicates
    
def calculate_hash(file_path):
    """Calculates the MD5 hash of a file."""
    
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
