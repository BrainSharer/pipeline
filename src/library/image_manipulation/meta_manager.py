"""This module is responsible for extracting metadata from the CZI files.
"""

import os, sys, time, re, psutil
from datetime import datetime
from pathlib import Path
import operator

from library.controller.sql_controller import SqlController
from library.database_model.slide import Slide, SlideCziTif
from library.image_manipulation.czi_manager import CZIManager
from library.utilities.utilities_process import convert_size


class MetaUtilities:
    """Collection of methods used to extract meta-data from czi files and insert 
    into database. Also includes methods for validating information in 
    database and/or files [double-check]
    """

    def extract_slide_meta_data_and_insert_to_database(self):
        """REVISED FOR PARALLEL PROCESSING
        Scans the czi dir to extract the meta information for each tif file
        """

        INPUT = self.fileLocationManager.get_czi(self.rescan_number)
        czi_files = self.check_czi_file_exists()
        self.scan_id = self.get_user_entered_scan_id()
        print(f'\tScan run ID={self.scan_id}')
        file_validation_status, unique_files = self.file_validation(czi_files)
        db_validation_status,outstanding_files = self.all_slide_meta_data_exists_in_database(unique_files)
        if file_validation_status and db_validation_status:
            if len(outstanding_files) > 0:
                dict_target_filesizes = {}  # dict for symlink <-> target file size
                for filename in outstanding_files:
                    symlink = os.path.join(INPUT, filename)
                    target_file = Path(symlink).resolve()  # target of symbolic link
                    file_size = os.path.getsize(target_file)
                    dict_target_filesizes[filename] = file_size

                files_ordered_by_filesize_desc = dict(
                    sorted(
                        dict_target_filesizes.items(),
                        key=operator.itemgetter(1),
                        reverse=True,
                    )
                )
                file_keys = []
                for i, file in enumerate(files_ordered_by_filesize_desc.keys()):
                    infile = os.path.join(INPUT, file)
                    infile = infile.replace(" ","_").strip()
                    if i == 0:  # largest file
                        single_file_size = os.path.getsize(infile)

                    file_keys.append([infile, self.scan_id])

                ram_coefficient = 2

                mem_avail = psutil.virtual_memory().available
                batch_size = mem_avail // (single_file_size * ram_coefficient)
                msg = f"MEM AVAILABLE: {convert_size(mem_avail)}; [LARGEST] SINGLE FILE SIZE: {convert_size(single_file_size)}; BATCH SIZE: {round(batch_size,0)}"
                self.logevent(msg)
                
                workers = self.get_nworkers()
                print(f'working on parallel extract files={len(file_keys)}')
                self.run_commands_with_threads(self.parallel_extract_slide_meta_data_and_insert_to_database, file_keys, workers)
            else:
                msg = "NOTHING TO PROCESS - SKIPPING"
                print(msg)
                self.logevent(msg)
        else:
            self.logevent("ERROR IN CZI FILES (DUPLICATE) OR DB COUNTS")
            sys.exit()

    def get_user_entered_scan_id(self):
        """Get id in the "scan run" table for the current microscopy scan that 
        was entered by the user in the preparation phase
        """
        
        return self.sqlController.scan_run.id

    def file_validation(self, czi_files):
        """CHECK IF DUPLICATE SLIDE NUMBERS EXIST IN FILENAMES; ALSO CHECKS CZI FORMAT
        CHECK DB COUNT FOR SLIDE TABLE

        :param czi_files: list of CZI files
        :return status: boolean on whether the files are valid
        :return list: list of CZI files
        """

        slide_id = []
        for file in czi_files:
            filename = os.path.splitext(file)
            if filename[1] == ".czi":
                slide_id.append(
                    int(re.sub("[^0-9]", "", str(re.findall(r"slide\d+", filename[0]))))
                )

        total_slides_cnt = len(slide_id)
        unique_slides_cnt = len(set(slide_id))
        msg = f"CZI SLIDES COUNT: {total_slides_cnt}; UNIQUE CZI SLIDES COUNT: {unique_slides_cnt}"
        status = True
        if unique_slides_cnt == total_slides_cnt and unique_slides_cnt > 0:
            msg2 = "NO DUPLICATE FILES; CONTINUE"
        else:
            msg2 = f"{total_slides_cnt-unique_slides_cnt} DUPLICATE SLIDE(S) EXIST(S); STOP"
            status = False
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
        
        qry = self.sqlController.session.query(Slide).filter(
            Slide.scan_run_id == self.scan_id
        )
        query_results = self.sqlController.session.execute(qry)
        results = [x for x in query_results]
        db_slides_cnt = len(results)

        msg = f"DB SLIDES COUNT: {db_slides_cnt}"
        print(msg)
        self.logevent(msg)
        status = True
        if db_slides_cnt > len(czi_files):
            # clean slide table in db for prep_id; submit all
            try:
                status = qry.delete()
                self.sqlController.session.commit()
            except Exception as e:
                msg = f"ERROR DELETING ENTRIES IN 'slide' TABLE: {e}"
                print(msg)
                self.logevent(msg)
                status = False
        elif db_slides_cnt > 0 and db_slides_cnt < len(czi_files):
            completed_files = []
            for row in results:
                completed_files.append(row[0].file_name)
            outstanding_files = set(czi_files).symmetric_difference(
                set(completed_files)
            )
            czi_files = outstanding_files
            msg = f"OUTSTANDING SLIDES COUNT: {len(czi_files)}"
            print(msg)
            self.logevent(msg)
        elif db_slides_cnt == len(czi_files):
            # all files processed (db_slides_cnt==filecount); continue with empty list
            czi_files = []
        self.session.close()
        return status, czi_files

    def check_czi_file_exists(self):
        """Check that the CZI files are placed in the correct location
        """
        
        INPUT = self.fileLocationManager.get_czi(self.rescan_number)
        if not os.path.exists(INPUT):
            print(f"{INPUT} does not exist, we are exiting.")
            sys.exit()
        try:
            files = os.listdir(INPUT)
            nfiles = len(files)
            if nfiles < 1:
                print("There are no CZI files to work with, we are exiting.")
                sys.exit()
            self.logevent(f"INPUT FOLDER: {INPUT}")
            self.logevent(f"FILE COUNT: {nfiles}")
        except OSError as e:
            print(e)
            sys.exit()

        return files


    def parallel_extract_slide_meta_data_and_insert_to_database(self, file_key):
        """A helper method to define some methods for extracting metadata.
        """
        infile, scan_id = file_key
        #czi_metadata = load_metadata(infile)
        czi_file = os.path.basename(os.path.normpath(infile))
        czi = CZIManager(infile)
        czi_metadata = czi.extract_metadata_from_czi_file(czi_file, infile)

        slide = Slide()
        slide.scan_run_id = scan_id
        slide.slide_physical_id = int(re.findall(r"slide\d+", infile)[0][5:])
        slide.slide_status = "Good"
        slide.processed = False
        slide.file_size = os.path.getsize(infile)
        slide.file_name = os.path.basename(os.path.normpath(infile))
        slide.created = datetime.fromtimestamp(
            Path(os.path.normpath(infile)).stat().st_mtime
        )
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

def validate(session):
    try:
        # Try to get the underlying session connection, If you can get it, its up
        connection = session.connection()
        return True
    except:
        return False
