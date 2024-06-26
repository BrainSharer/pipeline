"""This module is responsible for extracting metadata from the CZI files.
"""

import os, sys, time, re, json
from datetime import datetime
from pathlib import Path
import numpy as np
import tifffile
import hashlib

from library.database_model.slide import Slide, SlideCziTif
from library.image_manipulation.czi_manager import CZIManager


class MetaUtilities:
    """Collection of methods used to extract meta-data from czi files and insert 
    into database. Also includes methods for validating information in 
    database and/or files [double-check]
    """

    def extract_slide_meta_data_and_insert_to_database(self):
        """
        -Scans the czi dir to extract the meta information for each tif file
        """

        workers = self.get_nworkers()
        if self.debug:
            print(f"DEBUG: START MetaUtilities::extract_slide_meta_data_and_insert_to_database")
            workers = 1

        #START VERIFICATION OF PROGRESS & VALIDATION OF FILES
        self.input = self.fileLocationManager.get_czi(self.rescan_number)
        czi_files = self.check_czi_file_exists()
        self.scan_id = self.get_user_entered_scan_id()
        file_validation_status, unique_files = self.file_validation(czi_files) #CHECK FOR DUPLICATE SLIDES
        db_validation_status, unprocessed_czifiles, processed_czifiles = self.all_slide_meta_data_exists_in_database(unique_files) #CHECK FOR DB SECTION ENTRIES
        if not file_validation_status and not db_validation_status:
            self.logevent("ERROR IN CZI FILES OR DB COUNTS")
            print("ERROR IN CZI FILES OR DB COUNTS")
            sys.exit()
        else:
            #FOR CZI FILES ALREADY PROCESSED; CHECK FOR SLIDE PREVIEW
            if self.debug:
                print(f"DEBUG: CHECKING FOR RAW SLIDE PREVIEW IMAGES")
            file_keys = []
            for czi_file in processed_czifiles:
                infile = os.path.join(self.input, czi_file)
                infile = infile.replace(" ","_").strip()
                file_keys.append([infile, self.scan_id])

            #self.run_commands_with_threads(self.extract_slide_scene_data, file_keys, workers) #SLIDE PREVIEW

        #PROCESS OUTSTANDING EXTRACTIONS (SCENES FROM SLIDE FILES)
        if len(unprocessed_czifiles) > 0:
            file_keys = []
            for unprocessed_czifile in unprocessed_czifiles:
                infile = os.path.join(self.input, unprocessed_czifile)
                infile = infile.replace(" ","_").strip()
                file_keys.append([infile, self.scan_id])

            if self.debug:
                print(f"DEBUG: ANALYZING {infile}")
            self.logevent(f"ANALYZING {infile}")
            
            if self.debug:
                print(f'DEBUG: extract_slide_meta_data_and_insert_to_database: FILES: {len(file_keys)}; WORKERS: {workers}')

            #self.run_commands_with_threads(self.extract_slide_scene_data, file_keys, workers) #SLIDE PREVIEW
            self.run_commands_with_threads(self.parallel_extract_slide_meta_data_and_insert_to_database, file_keys, workers)
            
        else:
            msg = "NOTHING TO PROCESS - SKIPPING"
            if self.debug:
                print(msg)
            self.logevent(msg)

    def get_user_entered_scan_id(self):
        """Get id in the "scan run" table for the current microscopy scan that 
        was entered by the user in the preparation phase
        """
        
        return self.sqlController.scan_run.id

    def file_validation(self, czi_files):
        """CHECK IF DUPLICATE SLIDE NUMBERS EXIST IN FILENAMES. If there are duplicates, record the ID.
        ALSO CHECKS CZI FORMAT
        CHECK DB COUNT FOR SLIDE TABLE

        :param czi_files: list of CZI files
        :return status: boolean on whether the files are valid
        :return list: list of CZI files
        """

        if self.debug:
            print(f"DEBUG: START MetaUtilities::file_validation")

        slide_id = []
        for file in czi_files:
            filename = os.path.splitext(file)
            if filename[1] == ".czi":
                slide_id.append(int(re.sub("[^0-9]", "", str(re.findall(r"slide\d+", filename[0])))))

        total_slides_cnt = len(slide_id)
        unique_slides_cnt = len(set(slide_id))
        msg = f"TOTAL CZI SLIDES COUNT: {total_slides_cnt}; UNIQUE CZI SLIDES COUNT: {unique_slides_cnt}"
        status = True
        
        if unique_slides_cnt == total_slides_cnt and unique_slides_cnt > 0:
            msg2 = "NO DUPLICATE FILES; CONTINUE"
        else:
            self.multiple_slides = list(set([i for i in slide_id if slide_id.count(i)>1]))
            msg2 = f"{total_slides_cnt-unique_slides_cnt} DUPLICATE SLIDE(S) EXIST(S); multiple_slides with physical IDs={self.multiple_slides}"
            
        if self.debug:
            print(msg, msg2, sep="\n")
        self.logevent(msg)
        self.logevent(msg2)
        

        return status, czi_files

    def all_slide_meta_data_exists_in_database(self, czi_files):
        """Determines whether or not all the slide info is already 
        in the datbase

        :param list: list of CZI files
        :return status: boolean on whether the files are valid
        :return list: list of CZI files
        """
        
        if self.debug:
            print(f"DEBUG: START MetaUtilities::all_slide_meta_data_exists_in_database")

        qry = self.sqlController.session.query(Slide).filter(
            Slide.scan_run_id == self.scan_id)
        query_results = self.sqlController.session.execute(qry)
        results = [x for x in query_results]
        db_slides_cnt = len(results)

        msg = f"DB SLIDES COUNT: {db_slides_cnt}"
        if self.debug:
            print(msg)
        self.logevent(msg)

        status = True
        completed_files = []
        if db_slides_cnt > len(czi_files):
            # clean slide table in db for prep_id; submit all
            try:
                status = qry.delete()
                self.sqlController.session.commit()
            except Exception as e:
                msg = f"ERROR DELETING ENTRIES IN 'slide' TABLE: {e}"
                if self.debug:
                    print(msg)
                self.logevent(msg)
                status = False
        elif db_slides_cnt > 0 and db_slides_cnt < len(czi_files):
            for row in results:
                completed_files.append(row[0].file_name)
            unprocessed_czifiles = set(czi_files).symmetric_difference(set(completed_files))
            czi_files = unprocessed_czifiles
            msg = f"OUTSTANDING SLIDES COUNT [TO EXTRACT SCENES FROM CZI]: {len(czi_files)}"
            if self.debug:
                print(msg)
            self.logevent(msg)
        elif db_slides_cnt == len(czi_files):
            # all files processed (db_slides_cnt==filecount); czi_files SHOULD BE EMPTY; completed_files IS LIST OF ALL CZI FILES PROCESSED [FOR VERIFICATION]
            czi_files = []
            for row in results:
                completed_files.append(row[0].file_name)
        self.session.close()
        return status, czi_files, completed_files

    def check_czi_file_exists(self):
        """Check that the CZI files are placed in the correct location
        """
        
        self.input = self.fileLocationManager.get_czi(self.rescan_number)
        if not os.path.exists(self.input):
            print(f"{self.input} does not exist, we are exiting.")
            sys.exit()
        try:
            files = os.listdir(self.input)
            nfiles = len(files)
            if nfiles < 1:
                print("There are no CZI files to work with, we are exiting.")
                sys.exit()
            self.logevent(f"Input FOLDER: {self.input}")
            self.logevent(f"FILE COUNT: {nfiles}")
        except OSError as e:
            print(e)
            sys.exit()

        return files

    def extract_slide_scene_data(self, file_keys: tuple):
        '''
        Extracts "raw" slide preview image from CZI file and stores it as a tiff file (with checksum)
        '''
        input_czi_file, scan_id = file_keys
        if self.debug:
            print(f"DEBUG: START MetaUtilities::extract_slide_scene_data")

        czi_file = os.path.basename(os.path.normpath(input_czi_file))
        czi = CZIManager(input_czi_file)

        #EXTRACT SLIDE PREVIEW IMAGE [IF !EXISTS]
        scale_factor = 0.5 #REMOVE HARD-CODING; WHERE?
        czi_filename_without_extension = os.path.splitext(os.path.basename(input_czi_file))[0]
        if not os.path.exists(self.fileLocationManager.slide_thumbnail_web):
            Path(self.fileLocationManager.slide_thumbnail_web).mkdir(parents=True, exist_ok=True)
        slide_preview = Path(self.fileLocationManager.slide_thumbnail_web, czi_filename_without_extension + '.tif')
        if not os.path.isfile(slide_preview): #CREATE SLIDE PREVIEW WITH CHECKSUM
            if self.debug:
                print(f'CREATING SLIDE PREVIEW: {slide_preview}')
            mosaic_data = czi.file.read_mosaic(C=0, scale_factor=scale_factor) #captures first channel
            image_data = ((mosaic_data - mosaic_data.min()) / (mosaic_data.max() - mosaic_data.min()) * 65535).astype(np.uint16)
            tifffile.imwrite(slide_preview, image_data, compression='zlib', bigtiff=True)
            with open(slide_preview, 'rb') as f:
                bytes = f.read()  # Read the entire file as bytes
                readable_hash = hashlib.sha256(bytes).hexdigest()
                checksum_file = slide_preview.with_suffix('.tif.sha256')
                with open(checksum_file, 'w') as f:
                    f.write(readable_hash)
        else:
            if self.debug:
                print(f'SLIDE PREVIEW EXISTS: {slide_preview}')

        #CREATE meta-data.json [IF !EXISTS]
        meta_data_file = 'meta-data.json'
        meta_store = os.path.join(self.fileLocationManager.prep, meta_data_file)
        if not os.path.isfile(meta_store):
            if self.debug:
                print(f'DEBUG: meta-data.json NOT FOUND; CREATING @ {meta_store}')
                
            czi_metadata = czi.extract_metadata_from_czi_file(czi_file, input_czi_file)
            with open(meta_store, 'w') as fh:
                json.dump(czi_metadata["json_meta"], fh, indent=4)


    def parallel_extract_slide_meta_data_and_insert_to_database(self, file_key):
        """
        A helper method to define some methods for extracting metadata.
        """
        infile, scan_id = file_key
        if self.debug:
            print(f"DEBUG: START MetaUtilities::parallel_extract_slide_meta_data_and_insert_to_database")

        #czi_metadata = load_metadata(infile)
        
        czi_file = os.path.basename(os.path.normpath(infile))
        czi = CZIManager(infile)
        czi_metadata = czi.extract_metadata_from_czi_file(czi_file, infile)
        
        # #CREATE meta-data.json [IF !EXISTS]
        # meta_data_file = 'meta-data.json'
        # meta_store = os.path.join(self.fileLocationManager.prep, meta_data_file)
        # if not os.path.isfile(meta_store):
        #     if self.debug:
        #         print(f'DEBUG: meta-data.json NOT FOUND; CREATING @ {meta_store}')
        #     with open(meta_store, 'w') as fh:
        #         json.dump(czi_metadata["json_meta"], fh, indent=4)
        
        slide = Slide()
        slide.scan_run_id = scan_id
        slide.slide_physical_id = int(re.findall(r"slide\d+", infile)[0][5:])
        slide.slide_status = "Good"
        slide.processed = False
        slide.file_size = os.path.getsize(infile)
        slide.file_name = os.path.basename(os.path.normpath(infile))
        slide.created = datetime.fromtimestamp(Path(os.path.normpath(infile)).stat().st_mtime)
        slide.scenes = len([elem for elem in czi_metadata.values()][0].keys())
        self.session.begin()
        self.session.add(slide)
        self.session.commit()
    
        """Add entry to the table that prepares the user Quality Control interface"""
        for series_index in range(slide.scenes):
            scene_number = series_index + 1
            channels = range(czi_metadata[slide.file_name][series_index]["channels"])
            channel_counter = 0 
            width, height = czi_metadata[slide.file_name][series_index]["dimensions"]
            tif_list = []
            for _ in channels:
                tif = SlideCziTif()
                tif.FK_slide_id = slide.id
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
        for slide_physical_id in self.multiple_slides:
            self.sqlController.get_and_correct_multiples(self.sqlController.scan_run.id, slide_physical_id)
            print(f'Updated tiffs to use multiple slide physical ID={slide_physical_id}')
