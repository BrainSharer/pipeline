from collections import defaultdict
import os, sys, glob, json, math
import inspect
import gzip
import shutil
import struct
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import dask
from dask import delayed
import dask.array as da
import imageio.v2 as imageio
from pathlib import Path
import csv
import cv2
import numpy as np
from compress_pickle import dump, load
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
import psutil
import warnings

from library.cell_labeling.cell_detector_trainer import CellDetectorTrainer
from library.cell_labeling.cell_utilities import (
    calc_moments_of_mask,
    calculate_correlation_and_energy,
    filter_cell_candidates,
    find_connected_segments,
    load_image,
    subtract_blurred_image,
    features_using_center_connected_components
)
from library.controller.sql_controller import SqlController
from library.database_model.annotation_points import AnnotationSession
from library.image_manipulation.file_logger import FileLogger
from library.image_manipulation.filelocation_manager import (
    ALIGNED_DIR,
    FileLocationManager,
)
from library.image_manipulation.parallel_manager import ParallelManager
from library.utilities.utilities_process import M_UM_SCALE, SCALING_FACTOR, get_scratch_dir, random_string, read_image, get_hostname

try:
    from settings import data_path, host, schema
except ImportError:
    print('Missing settings using defaults')
    data_path = "/net/birdstore/Active_Atlas_Data/data_root"
    host = "db.dk.ucsd.edu"
    schema = "brainsharer"


class CellMaker(ParallelManager):

    def __init__(self, animal, task, step=4, model="", channel=1, x=0, y=0, debug=False):
        """Set up the class with the name of the file and the path to it's location."""
        self.animal = animal
        self.task = task
        self.step = step
        self.model = model
        self.channel = channel
        self.section_count = 0
        self.test_x = x
        self.test_y = y
        self.hostname = get_hostname()
        self.fileLocationManager = FileLocationManager(animal)
        self.sqlController = SqlController(animal)
        self.debug = debug
        self.fileLogger = FileLogger(self.fileLocationManager.get_logdir(), self.debug)
        self.cell_label_path = os.path.join(self.fileLocationManager.prep, 'cell_labels')
        #####TODO put average cell someplace better
        self.avg_cell_img_file = Path(os.getcwd(), 'src', 'library', 'cell_labeling', 'average_cell_image.pkl')
        self.available_memory = int((psutil.virtual_memory().free / 1024**3) * 0.8)
        # These channels need to be defined for the create features process
        self.dye_channel = 0
        self.virus_channel = 0


    def report_status(self):
        print("RUNNING CELL MANAGER WITH THE FOLLOWING SETTINGS:")
        print("\tprep_id:".ljust(20), f"{self.animal}".ljust(20))
        print("\tDB host:".ljust(20), f"{host}".ljust(20))
        print("\tprocess host:".ljust(20), f"{self.hostname}".ljust(20))
        print("\tschema:".ljust(20), f"{schema}".ljust(20))
        print("\tdebug:".ljust(20), f"{str(self.debug)}".ljust(20))
        print("\ttask:".ljust(20), f"{str(self.task)}".ljust(20))
        print("\tavg cell image:".ljust(20), f"{str(self.avg_cell_img_file)}".ljust(20))
        print("\tavailable RAM:".ljust(20), f"{str(self.available_memory)}GB".ljust(20))
        print()


    def check_prerequisites(self, SCRATCH):
        '''
        cell labeling requires 
        a) available full-resolution images, 
        b) 2 channels (name, type), 
        c) scratch directory, 
        d) output directory, 
        E) cell_definitions (manual training of what cell looks like: average_cell_image.pkl), 
        F) models: /net/birdstore/Active_Atlas_Data/cell_segmentation/models/models_round_{self.step}_threshold_2000.pkl
        '''

        self.OUTPUT = self.fileLocationManager.get_cell_labels()

        if self.step and not Path(self.OUTPUT).is_dir(): #TRAINING/RE-TRAINING [FIRST RUN WILL JUST CREATE]
            self.OUTPUT = self.OUTPUT + f'{self.step}'

        if self.debug:
            print(f'Cell labels output dir: {self.OUTPUT}')
        self.fileLogger.logevent(f'Cell labels output dir: {self.OUTPUT}')

        self.SCRATCH = SCRATCH #TODO See if we can auto-detect nvme
        if self.debug:
            print(f'Temp storage location: {SCRATCH}')
        self.fileLogger.logevent(f'Temp storage location: {SCRATCH}')

        # CHECK FOR PRESENCE OF meta-data.json
        meta_data_file = 'meta-data.json'
        meta_store = os.path.join(self.fileLocationManager.prep, meta_data_file)

        try:
            meta_data_info = {}
            if os.path.isfile(meta_store):
                print(f'Found neuroanatomical tracing info; reading from {meta_store}')

                # verify you have 2 required channels
                with open(meta_store) as fp:
                    info = json.load(fp)
                self.meta_channel_mapping = info['Neuroanatomical_tracing']
                meta_data_info['Neuroanatomical_tracing'] = self.meta_channel_mapping

                # TODO: Move assertions to separate function (unit test) - maybe on send to log w/ error - missing file X
                # self.dyes = [item['description'] for item in info['Neuroanatomical_tracing']]
                # assert 'GFP' in self.dyes and 'NeurotraceBlue' in self.dyes
                # print('Two channels ready')
                # #self.fileLogger.logevent(f"Using 2 channels for automatic cell detection: {self.dyes}")

            else:
                # Create meta-data store (pull from database)
                if self.debug:
                    print(f'Meta-data file not found; Creating meta-data store @ {meta_store}')

                # steps to create
                channels_count = 3
                self.meta_channel_mapping = {
                    1: {
                        "mode": "dye",
                        "description": "NeurotraceBlue",
                        "channel_name": "C1",
                    },
                    3: {"mode": "virus", "description": "GFP", "channel_name": "C3"},
                }
                meta_data_info["Neuroanatomical_tracing"] = self.meta_channel_mapping

                with open(meta_store, 'w') as fp:
                    json.dump(meta_data_info, fp, indent=4)

        finally:
            # CHECK IF meta_data_info['Neuroanatomical_tracing'] CONTAINS A DYE AND VIRUS CHANNEL
            modes = [channel.get('mode') for channel in meta_data_info['Neuroanatomical_tracing'].values()]
            
            if 'dye' in modes and ('virus' in modes or 'ctb' in modes) :
                msg = "Neuroanatomical_tracing contains a dye channel and either virus or ctb channel."
            else:
                msg = "Neuroanatomical_tracing is missing either dye, virus or ctb channel."
                if self.debug:
                    print(msg)
                self.fileLogger.logevent(msg)
                raise ValueError(msg)

        # check for full-resolution images (tiff or ome-zarr)
        # checks tiff directory first, then ome-zarr [but must have at least 1 to proceed]
        for key, value in self.meta_channel_mapping.items():
            if value['mode'] == 'dye':
                dye_channel = value.get('channel_name')
            elif value['mode'] == 'virus' or value['mode'] == 'ctb':
                virus_marker_channel = value.get('channel_name')

        self.dye_channel = dye_channel[1]
        self.virus_channel = virus_marker_channel[1]
        found_dye_channel = False
        found_virus_marker_channel = False
        INPUT_dye = Path(self.fileLocationManager.get_full_aligned(channel=dye_channel[1]))
        INPUT_virus_marker = Path(self.fileLocationManager.get_full_aligned(channel=virus_marker_channel[1]))
        if INPUT_dye.exists():
            if self.debug:
                print(f'Full-resolution tiff stack found (dye channel): {INPUT_dye}')
            self.fileLogger.logevent(f'Full-resolution tiff stack found (dye channel): {INPUT_dye}')
            found_dye_channel = True
        else:
            print(f'Full-resolution tiff stack not found (dye channel). Expected location: {INPUT_dye}; will search for ome-zarr')
        if INPUT_virus_marker.exists():
            if self.debug:
                print(f'Full-resolution tiff stack found (virus/tracer channel): {INPUT_virus_marker}')
            self.fileLogger.logevent(f'Full-resolution tiff stack found (virus/tracer channel): {INPUT_virus_marker}')
            found_virus_marker_channel = True
        else:
            print(f'Full-resolution tiff stack not found (virus/tracer channel). Expected location: {INPUT_virus_marker}; will search for ome-zarr')

        if found_dye_channel == False:
            INPUT_dye = Path(self.fileLocationManager.get_neuroglancer(False, channel=dye_channel[1]) + '.zarr')
            INPUT_virus_marker = Path(self.fileLocationManager.get_neuroglancer(False, channel=virus_marker_channel[1]) + '.zarr')
            if INPUT_dye.exists():
                if self.debug:
                    print(f'Full-resolution ome-zarr stack found (dye channel): {INPUT_dye}')
                self.fileLogger.logevent(f'Full-resolution ome-zarr stack found (dye channel): {INPUT_dye}')
            else:
                print(f'Full-resolution ome-zarr stack not found (dye channel). Expected location: {INPUT_dye}; Exiting')
                sys.exit(1)
        if found_virus_marker_channel == False:
            if INPUT_virus_marker.exists():
                if self.debug:
                    print(f'Full-resolution ome-zarr stack found (virus/tracer channel): {INPUT_virus_marker}')
                self.fileLogger.logevent(f'full-resolution ome-zarr stack found (virus/tracer channel): {INPUT_virus_marker}')
            else:
                print(f'Full-resolution ome-zarr stack not found (virus/tracer channel). expected location: {INPUT_virus_marker}; exiting')
                sys.exit(1)

        # Check for cell training definitions file (average-cell_image.pkl)
        if self.avg_cell_img_file.is_file():
            if self.debug:
                print(f'Found cell training definitions file @ {self.avg_cell_img_file}')
            self.fileLogger.logevent(f'Found cell training definitions file @ {self.avg_cell_img_file}')

        # Check for model file (models_round_{self.step}_threshold_2000.pkl) in the models dir
        if self.model: #IF SPECIFIC MODEL SELECTED
            if self.debug:
                print(f'SEARCHING FOR SPECIFIC MODEL FILE: {self.model}')
            self.model_file = os.path.join('/net/birdstore/Active_Atlas_Data/cell_segmentation/models', f'models_{self.model}_round_{self.step}_threshold_2000.pkl')
        else:
            self.model_file = os.path.join('/net/birdstore/Active_Atlas_Data/cell_segmentation/models', f'models_round_{self.step}_threshold_2000.pkl')

        if os.path.exists(self.model_file):
            if self.debug:
                print(f'Found model file @ {self.model_file}')

            self.fileLogger.logevent(f'Found model file @ {self.model_file}')
        else:
            #IF STEP==1, MODEL FILE IS NOT REQUIRED (FIRST TRAINING)
            if self.step >1:
                print(f'Model file not found @ {self.model_file}')
                self.fileLogger.logevent(f'Model file not found @ {self.model_file}; Exiting')
                sys.exit(1)
            else:
                print('TRAINING MODEL CREATION MODE; NO SCORING')

        # check for available sections
        self.section_count = self.capture_total_sections('tif', INPUT_dye)
        if self.section_count == 0:
            print('No sections found; Exiting')
            self.fileLogger.logevent(f'no sections found; Exiting')
            sys.exit(1)

    def start_labels(self):
        '''1. Use dask to create virtual tiles of full-resolution images
                Which mode (dye/virus: Neurotrace/GFP is for which directory) -> rename input
                fluorescence_image = GFP a.k.a. virus channel (channel 3)
                nissel_stain_image = Neurotraceblue a.k.a. dye channel (channel 1) aka cell_body

           2. Identify cell candidates - image segmentation
                -this step will create pickle files totaling size (aggregate) of image stack (approx)
                @ end of step: in SCRATCH (1 compressed pickle file for each section - if cell candidates were detected)

           3. Create cell features
                @ start: check pickle files (count)
                @ end of step: in SCRATCH (1 csv file for each section - if cell candidates were detected)

           4. Detect cells; score cell candidate and classify as positive, negative, unknown
                @ start: csv file for each section with cell features [used in identification]
                @ end of step: csv file for each section where putative CoM cells detected with classification (positive, negative, unknown)
        '''
        self.fileLogger.logevent(f"DEBUG: start_labels - Steps 1 & 2 (revised); Start on image segmentation")
        if self.debug:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} Start")
            print(f"DEBUG: steps 1 & 2 (revised); Start on image segmentation")

        # TODO: Need to address scenario where >1 dye or virus channels are present [currently only 1 of each is supported]
        # SPINAL CORD WILL HAVE C1 (DYE) AND C2 (CTB) CHANNELS
        for channel_number, channel_data in self.meta_channel_mapping.items():
            if channel_data['mode'] == 'dye':
                self.dye_channel = channel_number
                self.fileLogger.logevent(f'Dye channel detected: {self.dye_channel}')
            elif channel_data['mode'] == 'virus' or channel_data['mode'] == 'ctb':
                self.virus_marker_channel = channel_number
                self.fileLogger.logevent(f'Virus or CTB channel detected: {self.virus_marker_channel}')
            elif channel_data['mode'] == 'unknown':
                continue
            else:
                msg = "Neuroanatomical_tracing is missing either dye or virus channel."
                if self.debug:
                    print(msg)
                    print(f'')
                self.fileLogger.logevent(msg)
                raise ValueError(msg)

        self.input_format = 'tif' #options are 'tif' and 'ome-zarr'

        if os.path.exists(self.avg_cell_img_file):
            avg_cell_img = load(self.avg_cell_img_file) #Load average cell image once
        else:
            print(f'Could not find {self.avg_cell_img_file}')
            sys.exit()

        #TODO: MOVE CONSTANTS TO SETTINGS?
        self.max_segment_size = 100000
        self.segmentation_threshold = 2000 
        self.cell_radius = 40

        if self.input_format == 'tif':
            input_path_dye = input_path_dye = self.fileLocationManager.get_full_aligned(channel=self.dye_channel)
            input_path_virus = self.fileLocationManager.get_full_aligned(channel=self.virus_marker_channel)
            self.section_count = self.capture_total_sections(self.input_format, input_path_dye) #Only need single/first channel to get total section count
        else:
            input_path_dye = Path(self.fileLocationManager.get_neuroglancer(False, channel=self.dye_channel) + '.zarr')
            input_path_virus = Path(self.fileLocationManager.get_neuroglancer(False, channel=self.virus_marker_channel) + '.zarr')

            # OME-ZARR Section count may be extracted from meta-data in folder or from meta-data in file [do not use database]

        file_keys = []
        for section in range(self.section_count):
            if section < 106:
                continue
            if self.section_count > 1000:
                str_section_number = str(section).zfill(4)
            else:
                str_section_number = str(section).zfill(3) 
            file_keys.append(
                [
                    self.animal,
                    section,
                    str_section_number,
                    self.segmentation_threshold,
                    self.cell_radius,
                    self.max_segment_size,
                    self.SCRATCH,
                    self.OUTPUT,
                    avg_cell_img,
                    self.model_file,
                    self.input_format,
                    input_path_dye,
                    input_path_virus,
                    self.step,
                    self.debug,
                ]
            )

        if self.debug:
            workers=1
            print(f'Running in debug mode with {workers} workers; {len(file_keys)} sections to process, out: {self.SCRATCH}')
        else:
            workers = math.floor(min([self.get_nworkers(), 10])*.5) # max 50% of prev. calcs [dask is ram intensive]
            print(f'running in parallel with {workers} workers; {len(file_keys)} sections to process, out: {self.SCRATCH}')
        self.run_commands_concurrently(self.detect_cells_all_sections, file_keys, workers)

    def identify_cell_candidates(self, file_keys: tuple) -> list:
        '''2. Identify cell candidates - PREV: find_examples()
                -Requires image tiling or dask virtual tiles prior to running

                This single method will be run in parallel for each section-consists of 3 sub-steps:
                A) subtract_blurred_image (average the image by subtracting gaussian blurred mean)
                B) identification of cell candidates based on connected segments
                C) filering cell candidates based on size and shape

           NOTE: '*_' (unpacking file_keys tuple will discard vars after debug if set); must modify if file_keys is changed
           :return: a list of cell candidates (dictionaries) for each section
        '''
        (
            animal,
            section,
            str_section_number,
            segmentation_threshold,
            cell_radius,
            max_segment_size,
            SCRATCH,
            OUTPUT,
            avg_cell_img,
            model_filename,
            input_format,
            input_path_dye,
            input_path_virus,
            step,
            debug,
            *_,
        ) = file_keys

        if not os.path.exists(input_path_virus):
            print(f'ERROR: {input_path_virus} not found')
            sys.exit(1)
        if not os.path.exists(input_path_dye):
            print(f'ERROR: {input_path_dye} not found')
            sys.exit(1)

        output_path = Path(SCRATCH, 'pipeline_tmp', animal, 'cell_candidates')
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = Path(output_path, f'extracted_cells_{str_section_number}.gz')

        #TODO check if candidates already extracted
        if os.path.exists(output_file):
            print(f'Cell candidates already extracted. Using: {output_file}')
            cell_candidates = load(output_file)
            return cell_candidates
        else:
            cell_candidates = []

        if debug:
            print(f'Starting identify_cell_candidates on section: {str_section_number}')

        # TODO: CLEAN UP - maybe extend dask to more dimensions?
        if input_format == 'tif':#section_number is already string for legacy processing 'tif' (zfill)
            input_file_virus = Path(input_path_virus, str_section_number + '.tif')
            input_file_dye = Path(input_path_dye, str_section_number + '.tif')
        else:
            store = parse_url(input_path_virus, mode="r").store
            reader = Reader(parse_url(input_path_virus))
            nodes = list(reader())
            image_node = nodes[0]  # first node is image pixel data
            dask_data = image_node.data
            total_sections = dask_data[0].shape[2]

            input_file_virus = []
            for img in dask_data[0][0][0]:
                input_file_virus.append(img)

            store = parse_url(input_path_dye, mode="r").store
            reader = Reader(parse_url(input_path_dye))
            nodes = list(reader())
            image_node = nodes[0]  # first node is image pixel data
            dask_data = image_node.data
            total_sections = dask_data[0].shape[2]

            input_file_dye = []
            for img in dask_data[0][0][0]:
                input_file_dye.append(img)

            # TODO: different processing for ome-zarr
            # see del_img_extract2.py (this folder) for more info

        # Create delayed tasks for loading the images (separate task list per channel)
        delayed_tasks_virus = [delayed(load_image)(path) for path in [input_file_virus]]
        delayed_tasks_dye = [delayed(load_image)(path) for path in [input_file_dye]]

        # Get shape without computing
        org_img_shape = dask.compute(delayed_tasks_virus[0].shape) 

        # Shape will be same for both channels (stores as y-axis then x-axis)
        x_dim = org_img_shape[0][1]
        y_dim = org_img_shape[0][0]

        # Create a Dask array from the delayed tasks (NOTE: DELAYED)
        image_stack_virus = [da.from_delayed(v, shape=(x_dim, y_dim), dtype='uint16') for v in delayed_tasks_virus]
        image_stack_dye = [da.from_delayed(v, shape=(x_dim, y_dim), dtype='uint16') for v in delayed_tasks_dye]
        data_virus = dask.compute(image_stack_virus[0])[0] #FULL IMAGE
        data_dye = dask.compute(image_stack_dye[0])[0] #FULL IMAGE

        # Swap x and y axes (read in y-axis, then x-axis but we want x,y)
        data_virus = np.swapaxes(data_virus, 1, 0)
        data_dye = np.swapaxes(data_dye, 1, 0)

        # FINAL VERSION BELOW:
        total_virtual_tile_rows = 5
        total_virtual_tile_columns = 2
        x_window = int(math.ceil(x_dim / total_virtual_tile_rows))
        y_window = int(math.ceil(y_dim / total_virtual_tile_columns))

        if debug:
            print(f'dask array created with following parameters: {x_window=}, {y_window=}; {total_virtual_tile_rows=}, {total_virtual_tile_columns=}')

        for row in range(total_virtual_tile_rows):
            for col in range(total_virtual_tile_columns):
                x_start = row*x_window
                x_end = x_window*(row+1)
                y_start = col*y_window
                y_end = y_window*(col+1)

                image_roi_virus = data_virus[x_start:x_end, y_start:y_end] #image_roi IS numpy array
                image_roi_dye = data_dye[x_start:x_end, y_start:y_end] #image_roi IS numpy array

                absolute_coordinates = (x_start, x_end, y_start, y_end)
                difference_ch3 = subtract_blurred_image(image_roi_virus) #calculate img difference for virus channel (e.g. fluorescence)

                connected_segments = find_connected_segments(difference_ch3, segmentation_threshold)

                if connected_segments[0] > 2:
                    # found cell candidate (first element of tuple is count)
                    difference_ch1 = subtract_blurred_image(image_roi_dye)  # Calculate img difference for dye channel (e.g. neurotrace)
                    cell_candidate = filter_cell_candidates(
                        animal,
                        section,
                        connected_segments,
                        max_segment_size,
                        cell_radius,
                        x_window,
                        y_window,
                        absolute_coordinates,
                        difference_ch1,
                        difference_ch3,
                    )
                    cell_candidates.extend(cell_candidate)  # Must use extend!
                    print(f"Found tile: {absolute_coordinates=}, section={str_section_number}")

        if len(cell_candidates) > 0:
            print(f'Saving {len(cell_candidates)} Cell candidates TO {output_file}')
            # if debug:
            #     print(f'Raw cell_candidates: {cell_candidates=}')
            dump(cell_candidates, output_file, compression="gzip", set_default_extension=True)

        print('Completed identify_cell_candidates')

        return cell_candidates

    def calculate_features(self, file_keys: tuple, cell_candidate_data: list) -> pd.DataFrame:
        '''Part of step 3. calculate cell features;

            This single method will be run in parallel for each section
            -consists of 5 sub-steps:
            A) load information from cell candidates (pickle files) - now passed as parameter
            B1) calculate_correlation_and_energy for channel 1
            B2) calculate_correlation_and_energy for channel 3
            C) features_using_center_connected_components(example)
            D) SAVE FEATURES (CSV FILE)

            cell_candidate_data: list of cell candidates (dictionaries) for each section
            each dictionary consists of:
            cell = {'animal': animal,
                        'section': section_number,
                        'area': object_area,
                        'absolute_coordinates_YX': (absolute_coordinates[2]+segment_col, absolute_coordinates[0]+segment_row),
                        'cell_shape_XY': (height, width),
                        'image_CH3': difference_ch3[row_start:row_end, col_start:col_end].T,
                        'image_CH1': difference_ch1[row_start:row_end, col_start:col_end].T,
                        'mask': segment_mask.T}

        '''

        (
            animal,
            section,
            str_section_number,
            segmentation_threshold,
            cell_radius,
            max_segment_size,
            SCRATCH,
            OUTPUT,
            avg_cell_img,
            model_filename,
            input_format,
            input_path_dye,
            input_path_virus,
            step,
            debug,
        ) = file_keys

        print(f'Starting function: calculate_features with {len(cell_candidate_data)} cell candidates')

        output_path = Path(SCRATCH, 'pipeline_tmp', animal, 'cell_features')
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = Path(output_path, f'cell_features_{str_section_number}.csv')

        # STEP 3-B) load information from cell candidates (pickle files from step 2 - cell candidate identification) **Now passed as parameter**
        output_spreadsheet = []
        for idx, cell in enumerate(cell_candidate_data):
            # STEP 3-C1, 3-C2) calculate_correlation_and_energy FOR CHANNELS 1 & 3 (ORG. FeatureFinder.py; calculate_features())
            ch1_corr, ch1_energy = calculate_correlation_and_energy(avg_cell_img["CH1"], cell['image_CH1'])
            ch3_corr, ch3_energy = calculate_correlation_and_energy(avg_cell_img['CH3'], cell['image_CH3'])

            # STEP 3-D) features_using_center_connected_components
            ch1_contrast, ch3_constrast, moments_data = features_using_center_connected_components(cell, debug)

            # Build features dictionary
            spreadsheet_row = {
                "animal": animal,
                "section": section,
                "index": idx,
                "row": cell["absolute_coordinates_YX"][0],
                "col": cell["absolute_coordinates_YX"][1],
                "area": cell["area"],
                "height": cell["cell_shape_XY"][1],
                "width": cell["cell_shape_XY"][0],
                "corr_CH1": ch1_corr,
                "energy_CH1": ch1_energy,
                "corr_CH3": ch3_corr,
                "energy_CH3": ch3_energy,
            }
            spreadsheet_row.update(moments_data[0])
            spreadsheet_row.update(moments_data[1]) #e.g. 'h1_mask' (6 items)
            spreadsheet_row.update({'contrast1': ch1_contrast, 'contrast3': ch3_constrast})
            output_spreadsheet.append(spreadsheet_row)


            fx = 26346
            fy = 5718
            #col = 5908
            #row = 27470
            srow = int(spreadsheet_row['row'])
            scol = int(spreadsheet_row['col'])
            if debug:
                # NOTE, col and row are switched
                if srow == fy and scol == fx:
                    for k,v in spreadsheet_row.items():
                        print(f'{k}: {v}')
                    mask = cell['mask']
                    print(f'fx: {fx}, fy: {fy} srow: {srow}, scol: {scol} self.test_x: {self.test_x}, self.test_y: {self.test_y}')
                    maskpath = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps/mask.npy'
                    np.save(maskpath, mask)
                    print(f'Found cell with mask shape {mask.shape} dtype {mask.dtype}')
                    print()

        df_features = pd.DataFrame(output_spreadsheet)
        df_features.to_csv(output_file, index=False)

        print(f'Saving {len(output_spreadsheet)} cell features to {output_file}')
        print('Completed calculate_features')

        return df_features


    def score_and_detect_cell(self, file_keys: tuple, cell_features: pd.DataFrame):
        ''' Part of step 4. detect cells; score cells based on features (prior trained models (30) used for calculation)'''

        (
            animal,
            section,
            str_section_number,
            segmentation_threshold,
            cell_radius,
            max_segment_size,
            SCRATCH,
            OUTPUT,
            avg_cell_img,
            model_filename,
            input_format,
            input_path_dye,
            input_path_virus,
            step,
            debug,
        ) = file_keys

        if debug and step == 1:
            print('Training model creation mode; no scoring')

        if  step > 1:
            model_file = load(model_filename)

        if debug:
            print(f'Starting function score_and_detect_cell on section {section}')

        def calculate_scores(features: pd.DataFrame, model):
            """
                Calculate scores, mean, and standard deviation for each feature.

                Args:
                features (pd.DataFrame): Input features.
                model: XGBoost model.

                Returns:
                tuple: Mean scores and standard deviation scores.
            """
            all_data = xgb.DMatrix(features)
            scores=np.zeros([features.shape[0], len(model)])

            for i, bst in enumerate(model):
                attributes = bst.attributes()
                try:
                    best_ntree_limit = int(attributes["best_ntree_limit"])
                except KeyError:
                    best_ntree_limit = 676
                scores[:, i] = bst.predict(all_data, iteration_range=[1, best_ntree_limit], output_margin=True)

            mean_scores = np.mean(scores, axis=1)
            std_scores = np.std(scores, axis=1)
            return mean_scores, std_scores

        def get_prediction_and_label(mean_scores: np.ndarray) -> list:
            """
                Get predictive cell labels based on mean scores.

                Args:
                mean_scores (np.ndarray): Mean scores.

                Returns:
                list: Predictive cell labels.
            """
            threshold = 1.5
            predictions = []

            for mean_score in mean_scores:
                mean_score = float(mean_score)
                classification = -2  # Default: cell candidate is not actually a cell

                if mean_score > threshold:  # Candidate is a cell
                    classification = 2
                elif -threshold <= mean_score <= threshold:
                    classification = 0  # UNKNOWN/UNSURE

                predictions.append(classification)

            return predictions

        drops = ['animal', 'section', 'index', 'row', 'col']        
        cell_features_selected_columns = cell_features.drop(drops,axis=1)

        # Step 4-2-1-2) calculate_scores(features) - calculates scores, labels, mean std for each feature
        if step > 1:
            mean_scores, std_scores = calculate_scores(cell_features_selected_columns, model_file)
            cell_features['mean_score'] = mean_scores
            cell_features['std_score'] = std_scores
            cell_features['predictions'] = np.array(get_prediction_and_label(mean_scores))

        # STEP 4-2-2) Stores dataframe as csv file
        if debug:
            print(f'Cell labels output dir: {OUTPUT}')
        Path(OUTPUT).mkdir(parents=True, exist_ok=True)
        cell_features.to_csv(Path(OUTPUT, f'detections_{str_section_number}.csv'), index=False)
        if debug:
            print('Completed detect_cell')


    def capture_total_sections(self, input_format: str, INPUT):
        '''Part of step 1. use dask to 'tile' images
        '''
        if input_format == 'tif': #Read full-resolution tiff files (for now)
            if os.path.exists(INPUT):
                total_sections = len(sorted(os.listdir(INPUT)))
            else:
                print(f'Error: input files {INPUT} not found')
                sys.exit()
        else:
            '''ALT processing: read OME-ZARR directly - only works on v. 0.4 as of 8-DEC-2023
            '''
            # Open the OME-Zarr file
            store = parse_url(INPUT, mode="r").store
            reader = Reader(parse_url(INPUT))
            nodes = list(reader())
            image_node = nodes[0]  # first node is image pixel data
            dask_data = image_node.data
            total_sections = dask_data[0].shape[2]
            del dask_data
        return total_sections

    def create_precomputed_annotations(self):
        xy_resolution = self.sqlController.scan_run.resolution
        z_resolution = self.sqlController.scan_run.zresolution

        spatial_dir = os.path.join(self.fileLocationManager.neuroglancer_data, 'predictions0', 'spatial0')
        info_dir = os.path.join(self.fileLocationManager.neuroglancer_data, 'predictions0')
        if os.path.exists(info_dir):
            print(f'Removing existing directory {info_dir}')
            shutil.rmtree(info_dir)
        os.makedirs(spatial_dir, exist_ok=True)
        point_filename = os.path.join(spatial_dir, '0_0_0.gz')
        info_filename = os.path.join(info_dir, 'info')
        dataframe_data = []
        for i in range(348):
            for j in range(0, self.sqlController.scan_run.height,200):
                x = j
                y = j
                dataframe_data.append([x, y, i])
        print(f'length of dataframe_data: {len(dataframe_data)}')
        with open(point_filename, 'wb') as outfile:
            buf = struct.pack('<Q', len(dataframe_data))
            pt_buf = b''.join(struct.pack('<3f', x, y, z) for (x, y, z) in dataframe_data)
            buf += pt_buf
            id_buf = struct.pack('<%sQ' % len(dataframe_data), *range(len(dataframe_data)))
            buf += id_buf
            bufout = gzip.compress(buf)
            outfile.write(bufout)

        chunk_size = [self.sqlController.scan_run.width, self.sqlController.scan_run.height, 348]
        info = {}
        spatial = {}
        spatial["chunk_size"] = chunk_size
        spatial["grid_shape"] = [1, 1, 1]
        spatial["key"] = "spatial0"
        spatial["limit"] = 1

        info["@type"] = "neuroglancer_annotations_v1"
        info["annotation_type"] = "CLOUD"
        info["by_id"] = {"key":"by_id"}
        info["dimensions"] = {"x":[str(xy_resolution),"μm"],
                            "y":[str(xy_resolution),"μm"],
                            "z":[str(z_resolution),"μm"]}
        info["lower_bound"] = [0,0,0]
        info["upper_bound"] = chunk_size
        info["properties"] = []
        info["relationships"] = []
        info["spatial"] = [spatial]    

        with open(info_filename, 'w') as infofile:
            json.dump(info, infofile, indent=2)
            print(f'Wrote {info} to {info_filename}')

    def extract_predictions(self):
        """
        Note, the point information from the CSV must be converted to 
        meters. 
        Parses cell label data from CSV files and inserts annotations with labels into the database.
        This method performs the following steps:
        1. Sets default properties for cell annotations.
        2. Initializes empty lists for points and child JSON objects.
        3. Generates a unique parent ID for the annotation session.
        4. Retrieves the XY and Z resolutions from the SQL controller.
        5. Constructs the path to the directory containing cell label CSV files.
        6. Checks if the CSV directory exists and exits if not found.
        7. Retrieves and sorts all CSV files in the directory.
        8. Parses each CSV file and extracts cell data if the file name contains 'detections_057'.
        9. Converts cell coordinates and predictions, and creates JSON objects for each detected cell.
        10. Aggregates the cell points and creates a cloud point annotation.
        11. Inserts the annotation with labels into the database if not in debug mode.
        Raises:
            SystemExit: If the CSV directory or files are not found.
            Exception: If there is an error inserting data into the database.
        Prints:
            Status messages indicating the progress and results of the parsing and insertion process.
        """

        default_props = ["#ffff00", 1, 1, 5, 3, 1]
        points = []
        childJsons = []
        parent_id = f"{random_string()}"

        # TODO: resolution stored IN meta-data.json
        # meta_data_file = 'meta-data.json'
        # meta_store = os.path.join(self.fileLocationManager.prep, meta_data_file)
        # if os.path.exists(meta_store):
        #     with open(meta_store, 'r') as fp:
        #         meta_data_info = json.load(fp)
        #     xy_resolution = meta_data_info['xy_resolution_unit']
        # else:
        #     xy_resolution = self.sqlController.scan_run.resolution

        xy_resolution = self.sqlController.scan_run.resolution
        z_resolution = self.sqlController.scan_run.zresolution
        found = 0

        dfpath = os.path.join(self.fileLocationManager.prep, 'cell_labels', 'all_predictions.csv')
        if os.path.exists(self.cell_label_path):
            print(f'Parsing cell labels from {self.cell_label_path}')
        else:
            print(f'ERROR: {self.cell_label_path} not found')
            sys.exit(1)
        dataframe_data = []
        detection_files = sorted(glob.glob( os.path.join(self.cell_label_path, f'detections_*.csv') ))
        if len(detection_files) == 0:
            print(f'Error: no csv files found in {self.cell_label_path}')
            sys.exit(1)

        for file_path in detection_files:
            rows = self.parse_csv(file_path)
            if rows:
                for row in rows:
                    prediction = float(row['predictions'])
                    section = float(row['section']) + 0.5 # Neuroglancer needs that extra 0.5
                    x = float(row['col'])
                    y = float(row['row'])

                    if prediction > 0:
                        if self.debug:
                            print(f'{prediction=} x={int(x)} y={int(y)} section={int(section)}')
                        dataframe_data.append([x, y, int(section - 0.5)])
                        x = x / M_UM_SCALE * xy_resolution
                        y = y / M_UM_SCALE * xy_resolution
                        section = section * z_resolution / M_UM_SCALE
                        found += 1
                        point = [x, y, section]
                        childJson = {
                            "point": point,
                            "type": "point",
                            "parentAnnotationId": f"{parent_id}",
                            "props": default_props
                        }
                        childJsons.append(childJson)
                        points.append(childJson["point"])

        print(f'Found {found} total neurons')
        if found == 0:
            print('No neurons found')
            sys.exit()

        FK_user_id = 1
        FK_prep_id = self.animal
        labels = ['MACHINE_SURE']
        id = None
        description = labels[0]
        cloud_points = {}

        cloud_points["source"] = points[0]
        cloud_points["centroid"] = np.mean(points, axis=0).tolist()
        cloud_points["childrenVisible"] = True
        cloud_points["type"] = "cloud"
        cloud_points["description"] = f"{description}"
        cloud_points["sessionID"] = f"{parent_id}"
        cloud_points["props"] = default_props
        cloud_points["childJsons"] = childJsons

        ###############################################
        # 'workaround' for db timeout [save to file and then sql insert from file]
        # long-term store in www folder for direct import
        annotations_dir = Path(self.fileLocationManager.neuroglancer_data, 'annotations')
        annotations_dir.mkdir(parents=True, exist_ok=True)
        annotations_file = str(Path(annotations_dir, labels[0]+'.json'))
        with open(annotations_file, 'w') as fh:
            json.dump(cloud_points, fh)
        print(f'Annotations saved to {annotations_file}')
        ###############################################

        df = pd.DataFrame(dataframe_data, columns=['x', 'y', 'section'])
        print(f'Found {len(df)} total neurons and writing to {dfpath}')

        df.to_csv(dfpath, index=False)

        ###############################################
        if not self.debug:
            label_objects = self.sqlController.get_labels(labels)
            label_ids = [label.id for label in label_objects]

            annotation_session = self.sqlController.get_annotation_session(self.animal, label_ids, FK_user_id)
            if annotation_session is not None:
                self.sqlController.delete_row(AnnotationSession, {"id": annotation_session.id})

            try:
                id = self.sqlController.insert_annotation_with_labels(FK_user_id, FK_prep_id, cloud_points, labels)
            except Exception as e:
                print(f'Error inserting data: {e}')

            if id is not None:
                print(f'Inserted annotation with labels with id: {id}')
            else:
                print('Error inserting annotation with labels')

    ##### start methods from cell pipeline
    def create_detections(self):
        """
        Used for automated cell labeling - final output for cells detected
        """
        print("Starting cell detections")
        self.report_status()
        scratch_tmp = get_scratch_dir()
        self.check_prerequisites(scratch_tmp)

        # if any error from check_prerequisites(), print error and exit
        # assert statement could be in unit test (separate)
        self.start_labels()
        print(f'Finished cell detections')


    def fix_coordinates(self):

        def check_df(csvfile, df):
            section = os.path.basename(csvfile).replace('detections_', '').replace('.csv', '')
            tif = str(section).zfill(3) + '.tif'
            maskpath = os.path.join(
                self.fileLocationManager.get_directory(channel=1, downsample=True, inpath='mask_placed_aligned'), tif )
            if os.path.exists(maskpath):
                mask = read_image(maskpath)
            else:
                print(f'ERROR: Mask not found {maskpath}')
                sys.exit(1)

            for index, df_row in df.iterrows():
                row = df_row['row']
                col = df_row['col']
                row = int(row // SCALING_FACTOR)
                col = int(col // SCALING_FACTOR)
                prediction = df_row['predictions']
                found = mask[row, col] > 0
                if prediction > 0 and not found:
                    # print(f'ERROR: Predicted cell {index=} not found at {row=}, {col=} {prediction=}')
                    df.loc[index, 'predictions'] = -2

            return df

        detection_files = sorted(glob.glob( os.path.join(self.cell_label_path, f'detections_*.csv') ))
        if len(detection_files) == 0:
            print(f'Error: no csv files found in {self.cell_label_path}')
            sys.exit(1)

        for csvfile in tqdm(detection_files):
            df = pd.read_csv(csvfile)
            df = check_df(csvfile, df)
            df.to_csv(csvfile, index=False)

    def train(self):
        '''
        METHODS TO [RE]TRAIN CELL DETECTOR MODEL

        HIGH LEVEL STEPS:
        1. Read all csv files in cell_label_path
        2. Concatenate all csv files
        3. Drop columns: animal, section, index, row, col, mean_score, std_score, predictions
        4. Add label column based on predictions
        5. Train model using concatenated data
        '''

        warnings.filterwarnings("ignore")

        #TODO: It seems like we could export single file from database with appropriate columns
        # would also suggest putting the human validated 'ground truth' files in separate directory (for auditing purposes)
        # maybe cell_labels/human_validated_{date} alone with a json file with details of the training (annotators, evaluation dates, sample sizes, other)

        if not Path(self.cell_label_path).is_dir() and self.debug:
            print(f"CREATE TRAINING MODEL")
            print(f"MISSING 'GROUND TRUTH' DIRECTORY; CREATING {self.cell_label_path}")
            Path(self.cell_label_path).mkdir()
            print(f"PLEASE ADD 'HUMAN_POSITIVE' DETECTION FILES TO {self.cell_label_path}")
            print(f"For template see: https://webdev.dk.ucsd.edu/docs/brainsharer/pipeline/modules/cell_labeling.html")
            sys.exit(1)

        print(f"Reading csv files from {self.cell_label_path}")
        detection_files = sorted(glob.glob(os.path.join(self.cell_label_path, f'detections_*.csv') ))
        if len(detection_files) == 0:
            print(f'Error: no csv files found in {self.cell_label_path}')
            sys.exit(1)
        
        dfs = []
        for csvfile in tqdm(detection_files, desc="Reading csv files"):
            df = pd.read_csv(csvfile)
            dfs.append(df)

        detection_features=pd.concat(dfs)
        detection_features_path = os.path.join(self.cell_label_path, 'detection_features.csv')
        # detection_features.to_csv(detection_features_path, index=False)
        print(detection_features.info())

        if self.debug:
            print(f'Found {len(dfs)} csv files in {self.cell_label_path}')
            print(f'Concatenated {len(detection_features)} rows from {len(dfs)} csv files')
        detection_features['label'] = np.where(detection_features['predictions'] > 0, 1, 0)
        # mean_score, predictions, std_score are results, not features

        drops = ['animal', 'section', 'index', 'row', 'col', 'mean_score', 'std_score', 'predictions'] 
        for drop in drops:
            if drop in detection_features.columns:
                detection_features.drop(drop, axis=1, inplace=True)

        print(f'Starting training on {self.animal} step={self.step} with {len(detection_features)} features')

        trainer = CellDetectorTrainer(self.animal, step=self.step) # Use Detector 4 as the basis
        new_models = trainer.train_classifier(detection_features, 676, 3, models = trainer.load_models()) # pass Detector 4 for training
        trainer = CellDetectorTrainer(self.animal, step=self.step + 1) # Be careful when saving the model. The model path is only relevant to 'step'. 
        # You need to use a new step to save the model, otherwise the previous models would be overwritten.
        trainer.save_models(new_models)

    def create_features(self):
        """This is not ready. I am testing on specific x,y,z coordinates to see if they match
        the results returned from the detection process. Once they match, we can pull
        coordinates from the database and run the detection process on them.
        Note, x and y are switched.
        """
        print("Starting cell detections")
        self.report_status()
        scratch_tmp = get_scratch_dir()
        self.check_prerequisites(scratch_tmp)

        LABEL = 'HUMAN_POSITIVE'
        label = self.sqlController.get_annotation_label(LABEL)
        annotation_session = self.sqlController.get_annotation_session(self.animal, label.id, 37, self.debug)

        if annotation_session is None:
            print(f'No annotations found for {LABEL}')
            sys.exit(1)
        xy_resolution = self.sqlController.scan_run.resolution
        z_resolution = self.sqlController.scan_run.zresolution
        
        try:
            data = annotation_session.annotation["childJsons"]
        except KeyError:
            print("No childJsons key in data")
            return
        
        section_data = defaultdict(list)
        for point in data:
            x, y, z = point["point"]
            x = int(round(x * M_UM_SCALE / xy_resolution))
            y = int(round(y * M_UM_SCALE / xy_resolution))
            section = int(np.round((z * M_UM_SCALE / z_resolution), 2))
            section_data[section].append((x,y))

        print(f'data length: {len(data)} length section data {len(section_data)}')
        avg_cell_img = load(self.avg_cell_img_file) #Load average cell image once

        if self.step:
            #USED FOR TRAINING/RE-TRAINING MODELS
            self.cell_label_path = self.cell_label_path + str(self.step)
            os.makedirs(self.cell_label_path, exist_ok=True)

        idx = 0
        for section in section_data:
            input_file_virus_path = os.path.join(self.fileLocationManager.get_directory(channel=self.virus_channel, downsample=False, inpath=ALIGNED_DIR), f'{str(section).zfill(3)}.tif')  
            input_file_dye_path = os.path.join(self.fileLocationManager.get_directory(channel=self.dye_channel, downsample=False, inpath=ALIGNED_DIR), f'{str(section).zfill(3)}.tif')  
            if os.path.exists(input_file_virus_path) and os.path.exists(input_file_dye_path):
                spreadsheet = []
                data_virus = load_image(input_file_virus_path)
                data_dye = load_image(input_file_dye_path)
                for x, y in section_data[section]:
                    print(f'processing coordinates {y=}, {x=}, {section=}')
                    idx += 1
                    height = 80
                    width = 80
                    x_start = y - (width//2)
                    x_end = x_start + 80
                    y_start = x - 40
                    y_end = y_start + 80

                    image_roi_virus = data_virus[x_start:x_end, y_start:y_end] #image_roi IS numpy array
                    image_roi_dye = data_dye[x_start:x_end, y_start:y_end] #image_roi IS numpy array
                    print(f'shape of image_roi_virus {image_roi_virus.shape} and shape of data_virus {image_roi_dye.shape}')

                    connected_segments = find_connected_segments(image_roi_virus, 2000)
                    n_segments, segment_masks, segment_stats, segment_location = (connected_segments)
                    segmenti = 0
                    #_, _, width, height, object_area = segment_stats[segmenti, :]
                    mask = segment_masks.copy()
                    #mask = mask.astype(np.uint8)
                    #mask[mask > 0] = 255
                    cell = {
                        "animal": self.animal,
                        "section": section,
                        "area": width*height,
                        "absolute_coordinates_YX": (y,x),
                        "cell_shape_XY": (height, width),
                        "image_CH3": image_roi_virus,
                        "image_CH1": image_roi_dye,
                        "mask": mask.T,
                    }

                    ch1_corr, ch1_energy = calculate_correlation_and_energy(avg_cell_img["CH1"], image_roi_dye)
                    ch3_corr, ch3_energy = calculate_correlation_and_energy(avg_cell_img['CH3'], image_roi_virus)
                    
                    # STEP 3-D) features_using_center_connected_components
                    ch1_contrast, ch3_constrast, moments_data = features_using_center_connected_components(cell, self.debug)

                    # Build features dictionary
                    spreadsheet_row = {
                        "animal": self.animal,
                        "section": section,
                        "index": idx,
                        "row": cell["absolute_coordinates_YX"][0],
                        "col": cell["absolute_coordinates_YX"][1],
                        "area": cell["area"],
                        "height": cell["cell_shape_XY"][1],
                        "width": cell["cell_shape_XY"][0],
                        "corr_CH1": ch1_corr,
                        "energy_CH1": ch1_energy,
                        "corr_CH3": ch3_corr,
                        "energy_CH3": ch3_energy,
                    }
                    spreadsheet_row.update(moments_data[0])
                    spreadsheet_row.update(moments_data[1]) #e.g. 'h1_mask' (6 items)
                    spreadsheet_row.update({'contrast1': ch1_contrast, 'contrast3': ch3_constrast, 'predictions': 2})
                    spreadsheet.append(spreadsheet_row)

                df_features = pd.DataFrame(spreadsheet)
                dfpath = os.path.join(self.cell_label_path, f'detections_{str(section).zfill(3)}.csv')
                df_features.to_csv(dfpath, index=False)
                print(f'Saved {len(df_features)} features to {dfpath}')
        print(f'Finished processing {idx} coordinates')


    @staticmethod
    def parse_csv(file_path):
        """Opens and parses a CSV file, returning its contents as a list of dictionaries."""
        try:
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                data = [row for row in reader]
            return data
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    @staticmethod
    def detect_cells_all_sections(file_keys: tuple):
        # launcher for multiprocessing of all (4) steps - all functions must be serializable ('pickleable')
        # class instances are not serializable; use static methods instead
        # notes: model_filename has placeholder, but cannot be loaded into file_keys (not serializable)
        # filelogger not currently active in multi-processing mode

        # think about how to inherit, or reinstantiate Pipeline/Filelogger
        # filemanager = FileLocationManager(file_key[0])
        # filelogger = FileLogger(filemanager.get_logdir())
        # cellmaker = filelogger(CellMaker)

        # currently debug (bool) is set at end of file_keys
        animal, *_ = file_keys
        cellmaker = CellMaker(animal, task=None)
        if file_keys[-1]: #Last element of tuple is debug
            print(f"DEBUG: auto_cell_labels - STEP 1 & 2 (Identify cell candidates)")

        cell_candidates = cellmaker.identify_cell_candidates(file_keys) #STEPS 1 & 2. virtual tiling and cell candidate identification

        if file_keys[-1]: #DEBUG
            print(f"DEBUG: found {len(cell_candidates)} cell candidates on section {file_keys[1]}")

        if len(cell_candidates) > 0: #continue if cell candidates were detected [note: this is list of all cell candidates]
            if file_keys[-1]: #DEBUG
                print(f"DEBUG: create cell features with identified cell candidates (auto_cell_labels - step 3)")
            cell_features = cellmaker.calculate_features(file_keys, cell_candidates) #Step 3. calculate cell features
            #print(f'type cell features {type(cell_features)}')
            #print(cell_features.head())
            if file_keys[-1]: #DEBUG
                print(f'DEBUG: start_labels - STEP 4 (Detect cells [based on features])')
                print(f'Cell features: {len(cell_features)}')
            cellmaker.score_and_detect_cell(file_keys, cell_features)
