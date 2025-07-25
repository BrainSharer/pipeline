"""This module is responsible for extracting metadata from the CZI files.
"""

import os, sys, time, re, io
import inspect
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image

from library.database_model.slide import Slide, SlideCziTif
from library.image_manipulation.czi_manager import CZIManager
from library.utilities.utilities_mask import scaled
from library.utilities.utilities_process import DOWNSCALING_FACTOR


class MetaUtilities:
    """Collection of methods used to extract meta-data from czi files and insert 
    into database. Also includes methods for validating information in 
    database and/or files [double-check]
    """

    def extract_slide_meta_data_and_insert_to_database(self):
        """
        -Scans the czi dir to extract the meta information for each tif file
        -ALSO CREATES SLIDE PREVIEW IMAGE
        """

        workers = self.get_nworkers()
        if self.debug:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} START")
            workers = 1

        #START VERIFICATION OF PROGRESS & VALIDATION OF FILES
        self.input = self.fileLocationManager.get_czi()
        czi_files = self.check_czi_file_exists()
        self.scan_id = self.sqlController.scan_run.id
        self.czi_directory_validation(czi_files) #CHECK FOR existing files and DUPLICATE SLIDES
        db_validation_status, unprocessed_czifiles, processed_czifiles = self.all_slide_meta_data_exists_in_database(czi_files) #CHECK FOR DB SECTION ENTRIES
        if self.debug:
            print(f'DEBUG: unprocessed czi files:  {sorted(unprocessed_czifiles)}')
        if db_validation_status:
            self.fileLogger.logevent("ERROR IN CZI FILES OR DB COUNTS")
            print("ERROR IN CZI FILES OR DB COUNTS")
            sys.exit()

        #FOR CZI FILES ALREADY PROCESSED; CHECK FOR SLIDE PREVIEW
        file_keys = []
        for czi_file in processed_czifiles:
            infile = os.path.join(self.input, czi_file)
            infile = infile.replace(" ","_").strip()
            file_keys.append([infile, self.scan_id])
        
        #PROCESS OUTSTANDING EXTRACTIONS (SCENES FROM SLIDE FILES)
        if len(unprocessed_czifiles) > 0:
            file_keys = []
            for unprocessed_czifile in unprocessed_czifiles:
                infile = os.path.join(self.input, unprocessed_czifile)
                infile = infile.replace(" ","_").strip()
                file_keys.append([infile, self.scan_id])
            
            self.run_commands_with_threads(self.parallel_extract_slide_meta_data_and_insert_to_database, file_keys, workers)
            
        else:
            self.fileLogger.logevent("NOTHING TO PROCESS - SKIPPING")


    def czi_directory_validation(self, czi_files):
        """CHECK IF DUPLICATE SLIDE NUMBERS EXIST IN FILENAMES. If there are duplicates, record the ID.
        ALSO CHECKS CZI FORMAT
        CHECK DB COUNT FOR SLIDE TABLE

        :param czi_files: list of CZI files
        :return status: boolean on whether the files are valid
        :return list: list of CZI files
        """

        if self.debug:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} START")

        slide_id = []
        for file in czi_files:
            filename = os.path.splitext(file)
            if filename[1] == ".czi":
                slide_id.append(int(re.sub("[^0-9]", "", str(re.findall(r"slide\d+", filename[0])))))

        total_slides_cnt = len(slide_id)
        unique_slides_cnt = len(set(slide_id))
        msg = f"TOTAL CZI SLIDES COUNT: {total_slides_cnt}; UNIQUE CZI SLIDES COUNT: {unique_slides_cnt}"
        
        if unique_slides_cnt == total_slides_cnt and unique_slides_cnt > 0:
            msg2 = "NO DUPLICATE FILES; CONTINUE"
        else:
            self.multiple_slides = list(set([i for i in slide_id if slide_id.count(i)>1]))
            msg2 = f"{total_slides_cnt-unique_slides_cnt} DUPLICATE SLIDE(S) EXIST(S); multiple_slides with physical IDs={self.multiple_slides}"
            
        if self.debug:
            print(msg, msg2, sep="\n")
        self.fileLogger.logevent(msg)
        self.fileLogger.logevent(msg2)

        return

    def all_slide_meta_data_exists_in_database(self, czi_files):
        """Determines whether or not all the slide info is already 
        in the datbase

        :param list: list of CZI files
        :return status: boolean on whether the files are valid
        :return list: list of CZI files
        """

        db_validation_problem = False
        processed_czifiles = []
        unprocessed_czifiles = czi_files
        
        if self.debug:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} START")

        # active slides
        active_query = self.sqlController.session.query(Slide)\
            .filter(Slide.scan_run_id == self.scan_id)\
            .filter(Slide.active == True)
        active_query_results = self.sqlController.session.execute(active_query)
        active_results = [x for x in active_query_results]
        active_db_slides_cnt = len(active_results)
        
        # need to check for inactive so we don't repeat the process
        # remove the inactive czi files from the czi_files list
        inactive_query = self.sqlController.session.query(Slide)\
            .filter(Slide.scan_run_id == self.scan_id)\
            .filter(Slide.active == False)
        inactive_query_results = self.sqlController.session.execute(inactive_query)
        inactive_results = [x for x in inactive_query_results]
        inactive_db_slides_cnt = len(inactive_results)
        for row in inactive_results:
            if row[0].file_name in czi_files:
                czi_files.remove(row[0].file_name)

        msg = f"Active DB SLIDES COUNT: {active_db_slides_cnt}"
        msg += f"\nInactive DB SLIDES COUNT: {inactive_db_slides_cnt}"
        if self.debug:
            print(msg)            
        self.fileLogger.logevent(msg)

        if active_db_slides_cnt > len(czi_files):
            # clean slide table in db for prep_id; submit all
            try:
                db_validation_problem = active_query.delete()
                self.sqlController.session.commit()
            except Exception as e:
                msg = f"ERROR DELETING ENTRIES IN 'slide' TABLE: {e}"
                print(msg)
                self.fileLogger.logevent(msg)
                db_validation_problem = True
        elif active_db_slides_cnt > 0 and active_db_slides_cnt < len(czi_files):
            for row in active_results:
                processed_czifiles.append(row[0].file_name)
            unprocessed_czifiles = set(czi_files).symmetric_difference(set(processed_czifiles))
            czi_files = unprocessed_czifiles
            msg = f"OUTSTANDING SLIDES COUNT [TO EXTRACT SCENES FROM CZI]: {len(czi_files)}"
            if self.debug:
                print(msg)
            self.fileLogger.logevent(msg)
        elif active_db_slides_cnt == len(czi_files):
            # all files processed (db_slides_cnt==filecount); czi_files SHOULD BE EMPTY; processed_czifiles IS LIST OF ALL CZI FILES PROCESSED [FOR VERIFICATION]
            unprocessed_czifiles = []
            for row in active_results:
                processed_czifiles.append(row[0].file_name)
        self.session.close()
        return db_validation_problem, unprocessed_czifiles, processed_czifiles

    def check_czi_file_exists(self):
        """Check that the CZI files are placed in the correct location
        """
        
        self.input = self.fileLocationManager.get_czi()
        if not os.path.exists(self.input):
            print(f"{self.input} does not exist, we are exiting.")
            sys.exit()

        try:
            files = sorted(os.listdir(self.input))
            nfiles = len(files)
            if nfiles == 0:
                print(f"There are no CZI files in:\n{self.input}")
                sys.exit()
            self.fileLogger.logevent(f"Input FOLDER: {self.input}")
            self.fileLogger.logevent(f"FILE COUNT: {nfiles}")
        except OSError as e:
            print(e)
            sys.exit()

        return files

    def extract_slide_scene_data(self, czi_file: str):
        """Extracts the scene data from the CZI file and creates a preview image
        I don't see the point in creating a full-size preview image. It also crashes my computer.
        """

        czi = CZIManager(czi_file)

        scale_factor = DOWNSCALING_FACTOR
        czi_filename_without_extension = os.path.splitext(os.path.basename(czi_file))[0]
        os.makedirs(self.fileLocationManager.slides_preview, exist_ok=True)
        
        slide_preview_path = os.path.join(self.fileLocationManager.slides_preview, f'{czi_filename_without_extension}.png')

        if not os.path.isfile(slide_preview_path):

            if self.debug:
                print(f'CREATING SLIDE PREVIEWS: {slide_preview_path}')
            
            mosaic_data = czi.file.read_mosaic(C=0, scale_factor=scale_factor) #captures first channel
            image_data = ((mosaic_data - mosaic_data.min()) / (mosaic_data.max() - mosaic_data.min()) * 255).astype(np.uint8)
            if image_data.shape[0] == 1:
                image_data = image_data.reshape(image_data.shape[1], image_data.shape[2])
            image_data = scaled(image_data)
            
            # Convert to PIL Image
            img = Image.fromarray(image_data, mode='L')  # 'L' mode for 8-bit grayscale

            # Downsample full-size image to 30%
            width, height = img.size
            new_width = int(width * 0.3)
            new_height = int(height * 0.3)
            
            # Calculate the scaling factor to make width <= 1000px
            max_width = 1000
            width, height = img.size
            scale = min(max_width / width, 1)  # Don't upscale if width is already <= 1000px
            
            # Calculate new dimensions
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize the image proportionally
            img_scaled = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save scaled image to a BytesIO object
            img_scaled_byte_arr = io.BytesIO()
            img_scaled.save(img_scaled_byte_arr, format='PNG')
            img_scaled_byte_arr = img_scaled_byte_arr.getvalue()

            # Save scaled image
            if self.debug:
                print(f'Saving scaled image to {slide_preview_path}')
            with open(slide_preview_path, 'wb') as f:
                f.write(img_scaled_byte_arr)

        if self.debug:
            if os.path.isfile(slide_preview_path):
                print(f'Slide preview exists: {slide_preview_path}')
            else:
                print(f'Slide preview does not exist, creating: {slide_preview_path}')


    def parallel_extract_slide_meta_data_and_insert_to_database(self, file_key):
        """
        A helper method to define some methods for extracting metadata.
        First test if existing slide is in the database, if not, add it.
        We find it by animal, slide physical id, and scan run id.
        """
        infile, scan_id = file_key

        czi_file = os.path.basename(os.path.normpath(infile))
        czi = CZIManager(infile)
        czi_metadata = czi.extract_metadata_from_czi_file(czi_file, infile)
        slide_physical_id = int(re.findall(r"slide\d+", infile)[0][5:])
        file_name = os.path.basename(os.path.normpath(infile))
        # Start new Slide for each czi file
        slide = Slide()
        slide.scan_run_id = scan_id
        slide.slide_physical_id = slide_physical_id
        slide.slide_status = "Good"
        slide.processed = False
        slide.file_size = os.path.getsize(infile)
        slide.file_name = file_name
        slide.created = datetime.fromtimestamp(Path(os.path.normpath(infile)).stat().st_mtime)
        slide.scenes = len([elem for elem in czi_metadata.values()][0].keys())
        #####TODO
        self.checksum = os.path.join(self.fileLocationManager.www, 'checksums', 'slides_preview')
        os.makedirs(self.checksum, exist_ok=True)
        
        checksum_file = os.path.join(self.checksum, str(slide.file_name).replace('.czi', '.sha256'))
        if os.path.isfile(checksum_file):
            with open(checksum_file) as f: # The with keyword automatically closes the file when you are done
                readable_hash = f.read()
            slide.checksum = readable_hash
        
        self.session.begin()
        self.session.add(slide)
        self.session.commit()

    
        """Add entry to the table that prepares the user Quality Control interface"""
        for series_index in range(slide.scenes):
            scene_number = series_index + 1
            try:
                #channels = range(czi_metadata[slide.file_name][series_index]["channels"])
                channels = range(czi_metadata[file_name][series_index]["channels"])
            except KeyError:
                print(f'Channel error with slide file name={slide.file_name} file system name= {file_name}')
                sys.exit()
            channel_counter = 0
            try:
                width, height = czi_metadata[file_name][series_index]["dimensions"]
            except KeyError:
                print(f'Width, height error with slide file name={slide.file_name} file system name= {file_name}')
                sys.exit()
            tif_list = []
            for _ in channels:
                tif = SlideCziTif()
                tif.FK_slide_id = slide.id
                tif.czifile = slide.file_name
                tif.scene_number = scene_number
                tif.file_size = 0
                tif.active = 1
                tif.width = width
                tif.height = height
                tif.scene_index = series_index
                channel_counter += 1
                newtif = "{}_S{}_C{}.tif".format(infile, scene_number, channel_counter)
                newtif = newtif.replace(".czi", "").replace("__", "_")
                tif.file_name = os.path.basename(newtif)
                tif.channel = channel_counter
                tif.processing_duration = 0
                tif.created = time.strftime("%Y-%m-%d %H:%M:%S")
                tif_list.append(tif)
            if len(tif_list) > 0:
                self.session.add_all(tif_list)
                self.session.commit()
        return


    def correct_multiples(self):
        """
        This method will take care of slides that have multiple slide physical IDs. It will
        take the one with the higher ID and update them to use the lower ID (the first one).
        This way, rescans of the same slide can be placed in the CZI directory and the
        end user can view/modify them in the slide QC area.
        """

        if self.debug:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} START on {len(self.multiple_slides)} slides")
                    
        for slide_physical_id in self.multiple_slides:
            if self.debug:
                print(f"DEBUG: MODIFYING {slide_physical_id=}, {self.sqlController.scan_run.id=}, {slide_physical_id=}")
            self.sqlController.get_and_correct_multiples(self.sqlController.scan_run.id, slide_physical_id, self.debug)
            self.fileLogger.logevent(f'Updated tiffs to use multiple slide physical ID={slide_physical_id}')

    def reorder_scenes(self):
        """
        This method will order the scenes in the database by their scene number.
        It will also update the file names to reflect the correct order.
        """

        if self.debug:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} START")

        # get unique channels for all slides
        channels = self.sqlController.session.query(SlideCziTif.channel)\
            .join(Slide)\
            .filter(Slide.scan_run_id == self.sqlController.scan_run.id)\
            .filter(Slide.active == 1)\
            .filter(SlideCziTif.active == 1)\
            .distinct()
        channels = [channel[0] for channel in channels]
        print(f"DEBUG: Reordering scenes for channels: {channels}")

        for channel in channels:
            scenes = self.sqlController.session.query(SlideCziTif)\
                .join(Slide)\
                .filter(Slide.scan_run_id == self.sqlController.scan_run.id)\
                .filter(Slide.active == 1)\
                .filter(SlideCziTif.active == 1)\
                .filter(SlideCziTif.channel == channel)\
                .order_by(Slide.slide_physical_id)\
                .order_by(SlideCziTif.scene_number)\
                .order_by(SlideCziTif.channel)\
                    .all()

            for scene_order, scene in enumerate(scenes):
                channel = scene.channel
                scene.scene_order = scene_order
                self.session.add(scene)
                if self.debug:
                    print(f"DEBUG: Scene {scene.scene_number} - File Name: {scene.file_name}")

        self.session.commit()
