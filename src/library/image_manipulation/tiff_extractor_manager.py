import os, sys, glob, json
import inspect
from pathlib import Path
import hashlib
from tqdm import tqdm
import shutil
import traceback
from collections import defaultdict
import asyncio
import aiofiles

from library.image_manipulation.czi_manager import extract_tiff_from_czi, extract_png_from_czi
from library.utilities.utilities_process import DOWNSCALING_FACTOR, get_scratch_dir, use_scratch_dir
from library.utilities.cell_utilities import (
    copy_with_rclone
)


class TiffExtractor():
    """Includes methods to extract tiff images from czi source files and generate png files for quick viewing of
    downsampled images in stack

    """

    def extract_tiffs_from_czi(self):
        """
        Extracts TIFF images from CZI files for a specified channel. (or all remaining channels if channel=all)
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
            self.output = self.fileLocationManager.tif
            scale_factor = 1

        self.input = self.fileLocationManager.get_czi()
        Path(self.output).mkdir(parents=True, exist_ok=True)
        self.checksum = Path(self.fileLocationManager.www, 'checksums', 'czi')

        if not (self.channel == 'all' and self.downsample == False):
            #SINGLE CHANNEL & DOWNSAMPLED PROCESSING
            self.input = self.fileLocationManager.get_czi()
            os.makedirs(self.output, exist_ok=True)
            
            sections = self.sqlController.get_sections(self.animal, self.channel, self.debug)
            starting_files = glob.glob(
                os.path.join(self.output, "*_C" + str(self.channel) + ".tif")
            )
            total_files = os.listdir(self.output)

            self.fileLogger.logevent(f"TIFF EXTRACTION FOR CHANNEL: {self.channel}")
            self.fileLogger.logevent(f"DB SECTIONS QTY FOR CHANNEL: {len(sections)} (A.K.A. TOTAL FILES TO EXTRACT)")
            self.fileLogger.logevent(f"EXTRACTED FILE COUNT: {len(starting_files)}")
            
            if len(sections) == 0:
                print('\nError, no sections found, exiting.')
                print("Were the CZI file names correct on birdstore?")
                print("File names should be in the format: DK123_slideXXX_anything.czi")
                print("Are there slides in the database but no tifs? Check the database for existing slides and missing tifs")
                sys.exit()
            # elif len(sections) > len(starting_files):
            

            #REVERTED PRIOR PROCEDURES @ 12-SEP-2025
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
                extract_tiff_from_czi([czi_file, outfile, scene, int(self.channel), scale_factor])
            

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
                        
            # self.extract_by_section(self.channel, sections, scale_factor)
            
                # print('count off')
                # print(len(sections), len(starting_files))
                # expected_files = {section.file_name for section in sections} # Get all expected filenames from sections
                
                # actual_files = {os.path.basename(path) for path in starting_files}
                # missing_files = expected_files - actual_files # Find missing files by set difference

                # print(len(actual_files), len(expected_files))

                # if self.debug:
                #     missing_files, extra_files = self.compare_files(sections, self.output, self.channel)

                #     if len(extra_files) > 0 and self.debug:
                #         print(f"Found {len(extra_files)} unexpected files:")
                #         for file in extra_files:
                #             print(f"  - {file}")
                
                # sys.exit()
                # if len(missing_files) > 0:
                #     if self.debug:
                #         print(f"Missing files: {len(missing_files)}")
                #         for file in sorted(missing_files):
                #             print(f"  - {file}")
                #     self.extract_by_section(self.channel, sections, scale_factor)
        else:
            #PROCESS ALL CHANNELS (>1) IN FULL RESOLUTION
            #ASSUMES COMPLETE PIPELINE ALREADY RUN ON DOWNSAMPLED CHANNEL 1
            #GETS CHANNEL LIST FROM meta-data.json
            meta_data_file = 'meta-data.json'
            meta_store = os.path.join(self.fileLocationManager.prep, meta_data_file)

            try:
                meta_data_info = {}
                if os.path.isfile(meta_store):
                    print(f'Found neuroanatomical tracing info; reading from {meta_store}')

                    # verify you have >=2 required channels
                    with open(meta_store) as fp:
                        info = json.load(fp)
                    self.meta_channel_mapping = info['Neuroanatomical_tracing']
                    meta_data_info['Neuroanatomical_tracing'] = self.meta_channel_mapping
                
                    channel_info = [{'number': channel_number, 'name': channel_data['channel_name']} 
                                    for channel_number, channel_data in meta_data_info['Neuroanatomical_tracing'].items()]

                    print(f'PROCESSING {len(channel_info)} CHANNELS FROM meta-data.json')

            except Exception as e:
                print(f'Error reading {meta_store}, {e=}')
                print('Exiting')
                sys.exit()

            finally:
                assert (meta_data_info), (
                    f"MISSING META-DATA INFO; CANNOT AUTO-PROCESS ALL CHANNELS"
                )

                sections = {}
                starting_files_dict = {}  # Dictionary per channel
                all_starting_files = []   # Flat list of all files

                extraction_tasks = []
                for channel_data in channel_info:
                    channel_number = channel_data['number']
                    channel_name = channel_data['name']

                    if self.debug:
                        print(f"PROCESSING CHANNEL {channel_number} ({channel_name})")
                        print(f"==========================================")

                    # Get sections for THIS specific channel
                    channel_sections = self.sqlController.get_sections(self.animal, channel_number, self.debug)
                    sections[channel_number] = channel_sections

                    if self.debug:
                        print(f"Sections found: {len(channel_sections)}")
                        # Check if any files are missing for this channel
                        expected_files = {section.file_name for section in channel_sections}
                        actual_files = {os.path.basename(f) for f in glob.glob(os.path.join(self.output, f"*_C{channel_number}.tif"))}
                        missing_files = expected_files - actual_files
                        print(f"Missing files for channel {channel_number}: {len(missing_files)}")
                        if missing_files and len(missing_files) <= 5:
                            for f in missing_files:
                                print(f"  - {f}")

                    # Get starting files for this specific channel
                    channel_files = glob.glob(os.path.join(self.output, f"*_{channel_name}.tif"))
                    starting_files_dict[channel_name] = channel_files
                    all_starting_files.extend(channel_files)

                    self.fileLogger.logevent(f'Channel {channel_number} ({channel_name}): {len(channel_sections)} sections, {len(channel_files)} existing files')

                    if len(channel_sections) == 0:
                        print('\nError, no sections found, exiting.')
                        print("Were the CZI file names correct on birdstore?")
                        print("File names should be in the format: DK123_slideXXX_anything.czi")
                        print("Are there slides in the database but no tifs? Check the database for existing slides and missing tifs")
                        sys.exit()
                    elif len(channel_sections) > len(channel_files):  # Fixed: compare with channel_files instead of starting_files
        
                        expected_files = {section.file_name for section in channel_sections}  # Fixed: use channel_sections
                        actual_files = {os.path.basename(path) for path in channel_files}     # Fixed: use channel_files
                        missing_files = expected_files - actual_files  # Find missing files by set difference

                        if self.debug:
                            print(f"Missing files for channel {channel_number}: {len(missing_files)}")
                            if missing_files and len(missing_files) <= 5:
                                for f in missing_files:
                                    print(f"  - {f}")

                        if len(missing_files) > 0:
                            if self.debug:
                                print(f"Missing files for channel {channel_number}: {len(missing_files)}")
                                if len(missing_files) <= 20:
                                    for file in sorted(missing_files):
                                        print(f"  - {file}")
                                else:
                                    print(f"  (Too many to list individually - {len(missing_files)} files missing)")
                            
                            extraction_tasks.append((channel_number, channel_sections))

                # After collecting extraction_tasks, organize them by CZI file
                czi_extraction_plan = {}
                
                for channel_number, channel_sections in extraction_tasks:
                    for section in channel_sections:
                        czi_file = os.path.join(self.input, section.czi_file)
                        scene_index = section.scene_index
                        
                        # Create entry for this CZI file if it doesn't exist
                        if czi_file not in czi_extraction_plan:
                            czi_extraction_plan[czi_file] = {}
                        
                        # Create entry for this scene if it doesn't exist
                        if scene_index not in czi_extraction_plan[czi_file]:
                            czi_extraction_plan[czi_file][scene_index] = set()
                        
                        # Add this channel to the scene
                        czi_extraction_plan[czi_file][scene_index].add(channel_number)

                if self.debug:
                    print(f"\nEXTRACTION PLAN: Processing {len(czi_extraction_plan)} CZI files")
                    print("*" * 60)

                for czi_file, scenes in czi_extraction_plan.items():
                    czi_basename = os.path.basename(czi_file)
                    total_channels = sum(len(channels) for channels in scenes.values())
                    
                    if self.debug:
                        print(f"\n{czi_basename}:")
                        print(f"  Scenes: {len(scenes)}, Total extractions: {total_channels}")
                    
                    for scene_index, channels in sorted(scenes.items()):
                        sorted_channels = sorted(channels)
                        if self.debug:
                            print(f"    Scene {scene_index}: Channels {sorted_channels}")

                # Prepare the actual extraction tasks in a structured format
                structured_extraction_tasks = []
                missing_sections = []

                for czi_file, scenes in czi_extraction_plan.items():
                    for scene_index, channels in scenes.items():
                        for channel_number in channels:
                            # channel_number should already be an integer, no need to convert
                            channel_key = channel_number
                            
                            # Find the section for this specific combination
                            matching_sections = [s for s in sections.get(channel_key, []) 
                                            if s.czi_file == os.path.basename(czi_file) and s.scene_index == scene_index]
                            
                            if matching_sections:
                                section = matching_sections[0]
                                base_name = Path(czi_file).stem
                                outfile = os.path.join(self.output, section.file_name)
                                structured_extraction_tasks.append({
                                    'czi_file': czi_file,
                                    'outfile': outfile,
                                    'scene_index': scene_index,
                                    'channel_number': channel_key,
                                    'scale_factor': scale_factor,
                                    'section': section
                                })
                            else:
                                missing_sections.append({
                                    'czi_file': os.path.basename(czi_file),
                                    'scene_index': scene_index,
                                    'channel_number': channel_key,
                                    'available_sections': len(sections.get(channel_key, [])),
                                    'available_matching': [s for s in sections.get(channel_key, []) if s.czi_file == os.path.basename(czi_file)]
                                })

                if self.debug:
                    print(f"\nTOTAL EXTRACTION TASKS: {len(structured_extraction_tasks)}")
                    print("*" * 40)

                # Group by CZI file for efficient processing
                czi_grouped_tasks = {}
                for task in structured_extraction_tasks:
                    czi_file = task['czi_file']
                    if czi_file not in czi_grouped_tasks:
                        czi_grouped_tasks[czi_file] = []
                    czi_grouped_tasks[czi_file].append(task)

                #VALIDATE CHECKSUMS BEFORE PROCESSING
                czi_files_to_process = list(czi_grouped_tasks.keys())
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    verified_files, corrupted_files = loop.run_until_complete(
                        self.verify_czi_files_async(czi_files_to_process)
                    )
                finally:
                    loop.close()

                if corrupted_files:
                    print(f"\nWARNING: {len(corrupted_files)} corrupted files found:")
                    for file in corrupted_files:
                        print(f"  - {os.path.basename(file)}")
                    
                    # Only process verified files
                    czi_grouped_tasks = {k: v for k, v in czi_grouped_tasks.items() if k in verified_files}
                    

                for czi_file, tasks in czi_grouped_tasks.items():
                    czi_basename = os.path.basename(czi_file)

                    if self.debug:
                            print(f"\nPROCESSING: {czi_basename} with {len(tasks)} extraction tasks")
                            print("*" * 50)

                    
                print('NOT YET READY @ 12-SEP-2025')


                sys.exit()
                #     self.extract_by_section(int(channel_number), channel_sections, scale_factor, 'multi')
        
            


    def extract_by_section(self, channel, sections, scale_factor):
        for section in tqdm(sections, desc="Extracting TIFFs", disable=self.debug):
            czi_file = os.path.join(self.input, section.czi_file)
            if not os.path.exists(czi_file):
                print(f'Error: {czi_file} does not exist.')
                continue

            tif_file = os.path.basename(section.file_name)
            outfile = os.path.join(self.output, tif_file)
            
            if os.path.exists(outfile):
                continue
            scene = section.scene_index
            if self.debug:
                print(f"extracting from {os.path.basename(czi_file)}, {scene=}, to {outfile}")
            print(f'run: {outfile}')
            extract_tiff_from_czi([czi_file, outfile, scene, int(channel), scale_factor])


    

            # for section in tqdm(sections, desc="Extracting TIFFs", disable=self.debug):
            #     if mode == 'multi':
            #         remote_czi = os.path.join(self.input, section.czi_file)
            #         local_czi = Path(tmp_path, section.czi_file)
            #         if not os.path.exists(local_czi):
            #             if not os.path.exists(remote_czi):
            #                 print(f'Error: {remote_czi} does not exist.')
            #                 continue
            #             if self.debug:
            #                 print(f'Copying {remote_czi} to {local_czi}')
            #             os.system(f'cp "{remote_czi}" "{local_czi}"')
            #             czi_file = str(local_czi)
            #     else:
            #         czi_file = os.path.join(self.input, section.czi_file)
            #         if not os.path.exists(czi_file):
            #             print(f'Error: {czi_file} does not exist.')
            #             continue

            #     tif_file = os.path.basename(section.file_name)
            #     outfile = os.path.join(self.output, tif_file)
                
            #     if os.path.exists(outfile):
            #         continue
            #     scene = section.scene_index
            #     if self.debug:
            #         print(f"extracting from {os.path.basename(czi_file)}, {scene=}, to {outfile}")
            #     extract_tiff_from_czi([czi_file, outfile, scene, channel, scale_factor])
            
        

        # Check for duplicates
        duplicates = self.find_duplicates(self.fileLocationManager.thumbnail_original)
        if duplicates:
            self.fileLogger.logevent(f"DUPLICATE FILES FOUND: {duplicates}")
            print("Duplicate scenes found:")
            for duplicate in duplicates:
                for file in duplicate:
                    print(f"{os.path.basename(file)}", end=" ")
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
        

    def compare_files(self, sections, output_dir, channel):
        """Compare expected vs actual files and return missing files"""
        
        # Get expected files from sections
        expected_files = {section.file_name for section in sections}
        
        # Get actual files from output directory
        pattern = os.path.join(output_dir, f"*_C{channel}.tif")
        actual_files = {os.path.basename(path) for path in glob.glob(pattern)}
        
        # Find missing files
        missing_files = expected_files - actual_files
        
        return sorted(missing_files), sorted(actual_files - expected_files)


    async def verify_czi_files_async(self, czi_files):
        """Verify checksums for a list of CZI files asynchronously with progress"""
        verified_files = []
        corrupted_files = []
        missing_checksum_files = []
        
        total_files = len(czi_files)
        
        for i, czi_file in enumerate(czi_files, 1):
            czi_basename = os.path.basename(czi_file)
            
            if self.debug and i % 10 == 0:  # Print progress every 10 files
                print(f"Verifying {i}/{total_files}: {czi_basename}")
            
            checksum_filename = Path(czi_file).stem + '.sha256'
            checksum_filepath = Path(self.checksum, checksum_filename)
            
            if not checksum_filepath.exists():
                if self.debug:
                    print(f"WARNING: No checksum for {czi_basename}")
                missing_checksum_files.append(czi_file)
                verified_files.append(czi_file)
                continue
            
            try:
                async with aiofiles.open(checksum_filepath, 'r') as f:
                    stored_checksum = (await f.read()).strip()
            except IOError as e:
                print(f"ERROR reading checksum for {czi_basename}: {e}")
                corrupted_files.append(czi_file)
                continue
            
            current_checksum = await self.calculate_single_hash_async(czi_file)
            
            if current_checksum == stored_checksum:
                verified_files.append(czi_file)
                if self.debug:
                    print(f"✓ Verified: {czi_basename}")
            else:
                print(f"✗ Corrupted: {czi_basename}")
                corrupted_files.append(czi_file)
        
        # Summary report
        print(f"\nCHECKSUM VERIFICATION SUMMARY:")
        print(f"Verified: {len(verified_files)} files")
        print(f"Corrupted: {len(corrupted_files)} files")
        print(f"Missing checksums: {len(missing_checksum_files)} files")
        
        return verified_files, corrupted_files


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