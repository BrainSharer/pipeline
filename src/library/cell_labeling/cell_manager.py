from collections import defaultdict
import os, sys, glob, json, math
import re
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
import polars as pl #replacement for pandas (multi-core)
from tqdm import tqdm
import xgboost as xgb
import psutil
import warnings
try:
    import cupy as cp
except ImportError as ie:
    cp = None

from library.cell_labeling.cell_detector_trainer import CellDetectorTrainer
from library.cell_labeling.cell_utilities import (
    calc_moments_of_mask,
    calculate_correlation_and_energy,
    filter_cell_candidates,
    find_connected_segments,
    load_image,
    subtract_blurred_image,
    features_using_center_connected_components,
    find_available_backup_filename
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

    def __init__(self, animal: str, task: str, step: int = None, model: str = "", channel: int = 1, x: int = 0, y: int = 0, annotation_id: int = "", debug: bool = False):
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
        self.SCRATCH = get_scratch_dir()
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
        self.annotation_id = annotation_id

        #TODO: MOVE CONSTANTS TO SETTINGS?
        self.max_segment_size = 100000
        self.segmentation_threshold = 2000 
        self.cell_radius = 40
        self.ground_truth_filename = 'ground_truth.csv'


    def report_status(self):
        print("RUNNING CELL MANAGER WITH THE FOLLOWING SETTINGS:")
        print("\tprep_id:".ljust(20), f"{self.animal}".ljust(20))
        print("\tDB host:".ljust(20), f"{host}".ljust(20))
        print("\tprocess host:".ljust(20), f"{self.hostname}".ljust(20))
        print("\tschema:".ljust(20), f"{schema}".ljust(20))
        print("\tdebug:".ljust(20), f"{str(self.debug)}".ljust(20))
        print("\ttask:".ljust(20), f"{str(self.task)}".ljust(20))
        print("\tavg cell image:".ljust(20), f"{str(self.avg_cell_img_file)}".ljust(20))
        print("\tavg cell annotation_id:".ljust(20), f"{str(self.annotation_id)}".ljust(20), "[OPTIONAL FOR TRAINING]")
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
        
        if self.step == 1: #TRAINING [FIRST RUN WILL JUST CREATE]
            self.OUTPUT = self.OUTPUT + f'{self.step}'
        else:
            self.OUTPUT = self.OUTPUT
        

        if self.debug:
            print(f'Cell labels output dir: {self.OUTPUT}')
        self.fileLogger.logevent(f'Cell labels output dir: {self.OUTPUT}')

        # self.SCRATCH = SCRATCH #TODO See if we can auto-detect nvme
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

        # Check for specific model, if elected
        #TODO: consolidate model file location with cell_detector_base.py (DATA_PATH, MODELS)
        model_dir = Path('/net/birdstore/Active_Atlas_Data/cell_segmentation/models')
        if self.model: #IF SPECIFIC MODEL SELECTED
            if self.debug:
                print(f'SEARCHING FOR SPECIFIC MODEL FILE: {self.model}')

            if self.step:
                print(f'DEBUG: {self.step=}')
                self.model_file = Path(model_dir, f'models_{self.model}_step_{self.step}.pkl')
            else:
                # Find all matching model files and extract steps
                pattern = re.compile(rf'models_{re.escape(self.model)}_step_(\d+)\.pkl')
                
                # Get all matching files with their step numbers
                matching_files = []
                for f in model_dir.glob(f'models_{self.model}_step_*.pkl'):
                    match = pattern.match(f.name)
                    if match:
                        step = int(match.group(1))
                        matching_files.append((step, f))
                
                if matching_files:
                    # Sort by step number and pick the latest
                    matching_files.sort(reverse=True)
                    latest_step, latest_file = matching_files[0]
                    self.model_file = str(latest_file)
                    if self.debug:
                        print(f'Using latest step {latest_step} for model {self.model}')
                else:
                    raise FileNotFoundError(
                        f"No model files found for {self.model} in {model_dir}"
                    )
                
        else:
            self.model_file = Path(model_dir, 'models_round_4_threshold_2000.pkl')

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

        return (INPUT_dye, INPUT_virus_marker) #used for create_features

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
            # if section < 239:
            #     continue
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
            OUTPUT, _,
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

        cuda_available = cp.is_available()

        for row in range(total_virtual_tile_rows):
            for col in range(total_virtual_tile_columns):
                x_start = row*x_window
                x_end = x_window*(row+1)
                y_start = col*y_window
                y_end = y_window*(col+1)

                image_roi_virus = data_virus[x_start:x_end, y_start:y_end] #image_roi IS numpy array
                image_roi_dye = data_dye[x_start:x_end, y_start:y_end] #image_roi IS numpy array

                absolute_coordinates = (x_start, x_end, y_start, y_end)
                difference_ch3 = subtract_blurred_image(image_roi_virus, cuda_available) #calculate img difference for virus channel (e.g. fluorescence)

                connected_segments = find_connected_segments(difference_ch3, segmentation_threshold, cuda_available)

                if connected_segments[0] > 2:
                    if debug:
                        print(f'FOUND CELL CANDIDATE: COM-{absolute_coordinates=}, {cell_radius=}, {str_section_number=}')

                    # found cell candidate (first element of tuple is count)
                    difference_ch1 = subtract_blurred_image(image_roi_dye, cuda_available)  # Calculate img difference for dye channel (e.g. neurotrace)
                    
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
        
        if len(cell_candidates) > 0:
            if debug:
                print(f'Saving {len(cell_candidates)} Cell candidates TO {output_file}')
            # if debug:
            #     print(f'Raw cell_candidates: {cell_candidates=}')
            dump(cell_candidates, output_file, compression="gzip", set_default_extension=True)

        if debug:
            print('Completed identify_cell_candidates')
        return cell_candidates


    def calculate_features(self, file_keys: tuple, cell_candidate_data: list) -> pl.DataFrame:
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

        animal, section, str_section_number, _, _, _, SCRATCH, _, avg_cell_img, _, _, _, _, _, debug = file_keys

        print(f'Starting function: calculate_features with {len(cell_candidate_data)} cell candidates')

        cuda_available = cp.is_available()

        output_path = Path(SCRATCH, 'pipeline_tmp', animal, 'cell_features')
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = Path(output_path, f'cell_features_{str_section_number}.csv')

        # STEP 3-B) load information from cell candidates (pickle files from step 2 - cell candidate identification) **Now passed as parameter**
        output_spreadsheet = []
        for idx, cell in enumerate(cell_candidate_data):
            
            # STEP 3-C1, 3-C2) calculate_correlation_and_energy FOR CHANNELS 1 & 3 (ORG. FeatureFinder.py; calculate_features())
            # Initialize GPU data containers
            ch1_img_gpu = ch3_img_gpu = None
            results = {}
            if cuda_available:
                # Load all images to GPU once
                ch1_img_gpu = cp.asarray(cell['image_CH1'])
                ch3_img_gpu = cp.asarray(cell['image_CH3'])
                avg_ch1_gpu = cp.asarray(avg_cell_img["CH1"])
                avg_ch3_gpu = cp.asarray(avg_cell_img["CH3"])

                # Process correlation/energy on GPU
                results['ch1_corr'], results['ch1_energy'] = calculate_correlation_and_energy(
                    avg_ch1_gpu, ch1_img_gpu, cuda_available)
                
                results['ch3_corr'], results['ch3_energy'] = calculate_correlation_and_energy(
                    avg_ch3_gpu, ch3_img_gpu, cuda_available)
                
                # Process contrast/moments on GPU
                results['ch1_contrast'], results['ch3_contrast'], results['moments_data'] = \
                    features_using_center_connected_components(
                        {'image_CH1': ch1_img_gpu, 
                        'image_CH3': ch3_img_gpu,
                        'mask': cp.asarray(cell['mask'])},
                        cuda_available=cuda_available, debug=debug)
            else:
                results['ch1_corr'], results['ch1_energy'] = calculate_correlation_and_energy(
                    avg_cell_img["CH1"], cell['image_CH1'], cuda_available)
                
                results['ch3_corr'], results['ch3_energy'] = calculate_correlation_and_energy(
                    avg_cell_img["CH3"], cell['image_CH3'], cuda_available)

                results['ch1_contrast'], results['ch3_contrast'], results['moments_data'] = \
                    features_using_center_connected_components(cell, cuda_available=cuda_available, debug=debug)

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
                "corr_CH1": results['ch1_corr'],
                "energy_CH1": results['ch1_energy'],
                "corr_CH3": results['ch3_corr'],
                "energy_CH3": results['ch3_energy'],
                "contrast1": results['ch1_contrast'],
                "contrast3": results['ch3_contrast']
            }
            spreadsheet_row.update(results['moments_data'][0])  # Regular moments
            spreadsheet_row.update(results['moments_data'][1])  # Hu moments
            spreadsheet_row.update({'contrast1': results['ch1_contrast'], 'contrast3': results['ch1_contrast']})
            output_spreadsheet.append(spreadsheet_row)

            #TODO: EXPLORE IF HARD-CODED VARIABLES MAKE ANY DIFFERENCE HERE
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

        df_features = pl.DataFrame(output_spreadsheet)
        df_features.write_csv(output_file, separator=",")

        print(f'Saving {len(output_spreadsheet)} cell features to {output_file}')
        print('Completed calculate_features')

        return df_features


    def score_and_detect_cell(self, file_keys: tuple, cell_features: pl.DataFrame):
        ''' Part of step 4. detect cells; score cells based on features (prior trained models (30) used for calculation)'''

        _, section, str_section_number, _, _, _, SCRATCH, OUTPUT, _, model_filename, _, _, _, _, debug = file_keys

        #TODO: test if retraining or init model creating
        if model_filename:
            model_file = load(model_filename)
            model_msg = f'Training model: {model_filename}'
        else:
            model_msg = 'Training model creation mode; no scoring'
            
        if debug:
            print(model_msg)
            print(f'Starting function score_and_detect_cell on section {section}')

        def calculate_scores(features: pl.DataFrame, model):
            """
                Calculate scores, mean, and standard deviation for each feature.

                Args:
                features (pl.DataFrame): Input features.
                model: XGBoost model.

                Returns:
                tuple: Mean scores and standard deviation scores.
            """
            # Convert Polars DataFrame to pandas (XGBoost DMatrix prefers pandas)
            features_pd = features.to_pandas()

            # Ensure model is a list (for consistent processing)
            models = [model] if not isinstance(model, list) else model

            # Initialize scores array
            scores = np.zeros((features.shape[0], len(models)))

            for i, bst in enumerate(models):
                 # Get the model's expected features (excluding 'predictions' and 'idx')
                expected_features = [f for f in bst.feature_names if f not in ['predictions', 'idx']]

                # Select only the features the model should use
                data = features_pd[expected_features]

                # Create DMatrix (optimized for XGBoost)
                dmat = xgb.DMatrix(data)

                # Get best_ntree_limit (fallback to default if missing)
                best_ntree_limit = int(bst.attributes().get("best_ntree_limit", 676))

                # Predict
                scores[:, i] = bst.predict(
                    dmat,
                    iteration_range=[1, best_ntree_limit],
                    output_margin=True,
                    validate_features=False  # Safe because we filtered features
                )

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
        cell_features_selected_columns = cell_features.drop(drops)

        # Step 4-2-1-2) calculate_scores(features) - calculates scores, labels, mean std for each feature
        if model_filename: #IF MODEL FILE EXISTS, USE TO SCORE
            mean_scores, std_scores = calculate_scores(cell_features_selected_columns, model_file)
            cell_features = cell_features.with_columns([
                pl.Series("mean_score", mean_scores),
                pl.Series("std_score", std_scores),
                pl.Series("predictions", get_prediction_and_label(mean_scores))
            ])

        # STEP 4-2-2) Stores dataframe as csv file
        if debug:
            print(f'Cell labels output dir: {OUTPUT}')
        Path(OUTPUT).mkdir(parents=True, exist_ok=True)
        cell_features.write_csv(Path(OUTPUT, f'detections_{str_section_number}.csv'), separator=",")
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

        #TODO: See if converting to polars is faster
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

        if not Path(self.cell_label_path).is_dir(): #MISSING cell_labels
            if self.step: #TRAINING; CHECK IF 'GROUND TRUTH' DIRECTORY EXISTS
                self.INPUT = self.cell_label_path + f'{self.step}'
                if not Path(self.INPUT).is_dir():
                    print(f"MISSING 'TRAINING' DIRECTORY: {self.INPUT}")
                    print("RUN 'detect' TASK TO CREATE DIRECTORY AND AUTO-DETECT NEURONS")
                    sys.exit(1)
                else:
                    print(f"'GROUND TRUTH' DIRECTORY FOUND @ {self.INPUT}")
                    ground_truth_file = Path(self.INPUT, self.ground_truth_filename)
                    if not ground_truth_file.is_file():
                        print(f"MISSING 'GROUND TRUTH' FILE: {ground_truth_file}")
                        print("PLEASE ADD 'GROUND TRUTH' FILE TO DIRECTORY")
                        print("For template see: https://webdev.dk.ucsd.edu/docs/brainsharer/pipeline/modules/cell_labeling.html")
                        sys.exit(1)
        else:
            if self.step: #TRAINING; CHECK IF 'GROUND TRUTH' DIRECTORY EXISTS
                self.INPUT = self.cell_label_path + f'{self.step}'
            else:
                print('MISSING STEP NUMBER FOR TRAINING')
                sys.exit(1)
        
        print("PROCEEDING WITH TRAINING MODEL WITH THE FOLLOWING PARAMETERS:")
        print("\tprep_id:".ljust(20), f"{self.animal}".ljust(20))
        print("\tstep:".ljust(20), f"{self.step}".ljust(20))
        print("\tground truth directory:".ljust(20), f"{self.INPUT}".ljust(20))
        print("\tmodel:".ljust(20), f"{self.model}".ljust(20))
        print()
        
        agg_detection_features = Path(self.SCRATCH, 'pipeline_tmp', self.animal, 'detection_features.csv')
        if self.debug:
            print(f'Aggregated detection features tmp storage: {agg_detection_features}')
        
        if agg_detection_features.exists():
            backup_filename = find_available_backup_filename(agg_detection_features)
            os.rename(agg_detection_features, backup_filename)
            print(f'BACKUP STORED: {backup_filename}')

        print(f"Reading csv files from {self.INPUT}")
        detection_files = sorted(
            f for f in glob.glob(os.path.join(self.INPUT, "detections_*.csv"))
            if not re.search(r"detections_features\.csv$", f)
        )
        if len(detection_files) == 0:
            print(f'Error: no csv files found in {self.INPUT}')
            sys.exit(1)
        
        # Read files and collect column names
        column_order = None
        dfs = []
        schemas = {}

        for csvfile in tqdm(detection_files, desc="Checking CSV schemas"):
            try:
                df = pl.read_csv(csvfile)

                # Store column order from the first file
                if column_order is None:
                    column_order = df.columns
                
                # Ensure 'predictions' column exists, default to 0.0 if missing
                if "predictions" not in df.columns:
                    df = df.with_columns(pl.lit(0.0).alias("predictions"))

                # Convert all numeric columns to Float64 to ensure compatibility
                df = df.with_columns([
                    df[col].cast(pl.Float64) for col in df.columns if df[col].dtype in [pl.Int64, pl.Float32, pl.Int32]
                ])
                dfs.append(df)
                schemas[csvfile] = set(df.columns)  # Store column names as a set
            except Exception as e:
                print(f"Error reading {csvfile}: {e}")

        if "predictions" not in column_order:
            column_order.append("predictions")

        # Find the common columns while keeping order from the first file
        common_columns = [col for col in column_order if all(col in df.columns for df in dfs)]

        # Keep only common columns
        dfs = [df.select(common_columns) for df in dfs]

        detection_features = pl.concat(dfs, how="vertical")
        # detection_features.write_csv(agg_detection_features) #ONLY FOR AUDIT
        
        if self.debug:
            print(f'Found {len(dfs)} csv files in {self.INPUT}')
            print(f'Concatenated {len(detection_features)} rows from {len(dfs)} csv files')
            print(detection_features.describe())

        if "predictions" in detection_features.columns:
            detection_features = detection_features.with_columns(
                (pl.col("predictions") > 0).cast(pl.Int64).alias("label")
            )
        else:
            print("Error: 'predictions' column not found in detection_features.")
            print(detection_features.columns)
        # mean_score, predictions, std_score are results, not features

        # Add predictions column with default value -2 if it doesn't exist
        if 'predictions' not in detection_features.columns:
            detection_features = detection_features.with_columns(
                pl.lit(-2).alias('predictions')
            )

        detection_features = detection_features.with_columns(
            pl.when(pl.col('predictions') > 0)  # Replace with actual logic
            .then(1)
            .otherwise(0)
            .alias("label")
        )

        non_feature_columns = {'animal', 'section', 'index', 'row', 'col', 'mean_score', 'std_score'}
        detection_features = detection_features.drop(
            [col for col in non_feature_columns if col in detection_features.columns]
        )

        if self.debug:
            print(f'Starting training on {self.animal}, {self.model=}, step={self.step} with {len(detection_features)} features')

        trainer = CellDetectorTrainer(self.animal, step=self.step) # Use Detector 4 as the basis (default)
        np_model, model_filename = trainer.load_models(self.model, self.step)

        if self.debug:
            print(f'USING MODEL LOCATION: {model_filename}')

        local_scratch = Path(self.SCRATCH, 'pipeline_tmp', self.animal)

        #TODO - MOVE CONSTANTS SOMEWHERE ELSE; REMOVE HARD-CODING
        if self.step == 1:
            new_models = trainer.train_classifier(features = detection_features, local_scratch = local_scratch, model_filename = model_filename, niter = 676, depth = 3, debug = self.debug)
        else:
            new_models = trainer.train_classifier(features = detection_features, local_scratch = local_scratch, model_filename = model_filename, niter = 676, depth = 3, debug = self.debug, models = np_model) # pass Detector 4 for training

        trainer = CellDetectorTrainer(self.animal, step=self.step + 1) # Be careful when saving the model. The model path is only relevant to 'step'. 
        # You need to use a new step to save the model, otherwise the previous models would be overwritten.
        trainer.save_models(new_models, model_filename, local_scratch)


    def create_features(self):
        '''
        USED TO CREATE 'GROUND TRUTH' USER ANNOTATION SETS FOR MODEL TRAINING
        1) IF step IS PROVIDED; GROUND TRUTH ANNOTATIONS WILL BE STORED IN cell_labels/{step} DIRECTORY
        2) EXISTING ML-GENERATED DETECTIONS SHOULD HAVE BEEN RUN FIRST
        3) GROUND TRUTH FILE BACKED UP IF EXISTS, NEW ONE STORED IN cell_labels/{step} DIRECTORY [AUDIT]
        4) RE-PROCESS ML-DETECTION (FOR SECTIONS WITH GROUND TRUTH)
        '''

        """This is not ready. I am testing on specific x,y,z coordinates to see if they match
        the results returned from the detection process. Once they match, we can pull
        coordinates from the database and run the detection process on them.
        Note, x and y are switched.
        """

        if self.debug:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} Start")
        
        self.report_status()
        scratch_tmp = get_scratch_dir()
        INPUT_dye, INPUT_virus_marker = self.check_prerequisites(scratch_tmp)

        if self.debug:
            print(f'Cell labels output dir: {self.OUTPUT}')
        self.fileLogger.logevent(f'Cell labels output dir: {self.OUTPUT}')

        if not os.path.exists(self.OUTPUT):
            print(f'ML-GENERATED DETECTIONS DIRECTORY DOES NOT EXIST: {self.OUTPUT}')
            print(f'PLEASE RUN detect TASK FIRST; EXITING')
            sys.exit(1)
        else:
            print(f'ML-GENERATED DETECTIONS DIRECTORY FOUND: {self.OUTPUT}')
            print(f'CHECK FOR EXISTING ML-GENERATED DETECTIONS IN {self.OUTPUT}')
            detection_files = sorted(glob.glob( os.path.join(self.OUTPUT, f'detections_*.csv') ))
            if len(detection_files) == 0:
                print(f'Error: no csv files found in {self.cell_label_path}')
                sys.exit(1)
            else:
                print(f'CHECK FOR EXISTING GROUND TRUTH FILE IN {self.OUTPUT}')
                ground_truth_file = Path(self.OUTPUT, self.ground_truth_filename)
                if ground_truth_file.exists():
                    backup_filename = find_available_backup_filename(ground_truth_file)
                    os.rename(ground_truth_file, backup_filename)
                    print(f'BACKUP STORED: {backup_filename}')

        #TODO: ASSUMES THERE IS ONLY 1 HUMAN_POSITIVE ANNOTATION LABEL PER PREP_ID (SHOULD BE CONFIRMED)
        LABEL = 'HUMAN_POSITIVE' #DEFAULT LABEL
        if not self.annotation_id:
            label = self.sqlController.get_annotation_label(LABEL)
            annotation_session = self.sqlController.get_annotation_session(self.animal, label.id, 37, self.debug)
        else:
            #TODO ALLOW FOR MANUAL LABELING
            annotation_session = self.sqlController.get_annotation_by_id(self.annotation_id)     
            LABEL = annotation_session.annotation["description"]

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
     
        #WRITE ANNOTATIONS TO DISK FOR AUDIT TRAIL
        with open(ground_truth_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['section', 'x', 'y', 'id'])
        
            for section, points in section_data.items():
                for point in points:
                    writer.writerow([section, point[0], point[1], LABEL])

        #RE-PROCESS ML-DETECTIONS [FOR SECTIONS WITH GROUND TRUTH]
        avg_cell_img = load(self.avg_cell_img_file)

        idx = 0
        for section in section_data:
            input_file_virus_path = os.path.join(INPUT_virus_marker, f'{str(section).zfill(3)}.tif')  
            input_file_dye_path = os.path.join(INPUT_dye, f'{str(section).zfill(3)}.tif')  
            
            if os.path.exists(input_file_virus_path) and os.path.exists(input_file_dye_path):
                spreadsheet = []
                data_virus = load_image(input_file_virus_path)
                data_dye = load_image(input_file_dye_path)
                #TODO: Where did Kui get his area, height, width?
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
                    spreadsheet_row.update({'contrast1': ch1_contrast, 'contrast3': ch3_constrast, 'mean_score': 0,	'std_score': 0, 'predictions': 2})#added missing columns (mean, std)
                    spreadsheet.append(spreadsheet_row)

                df_features = pl.DataFrame(spreadsheet)
                df_features = df_features.with_columns([
                    pl.col("mean_score").cast(pl.Float64),
                    pl.col("std_score").cast(pl.Float64)
                ])
                            
                dfpath = os.path.join(self.OUTPUT, f'detections_{str(section).zfill(3)}.csv')

                # DO NOT REMOVE EXISTING DETECTIONS AS GROUND TRUTH FILE WILL LIKELY NOT INCLUDE ALL CELLS, JUST SAMPLING
                if os.path.exists(dfpath):
                    existing_df = pl.read_csv(dfpath)
                    df_features = pl.concat([existing_df, df_features])

                df_features.write_csv(dfpath)
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
            if file_keys[-1]: #DEBUG
                print(f'DEBUG: start_labels - STEP 4 (Detect cells [based on features])')
                print(f'Cell features: {len(cell_features)}')
            cellmaker.score_and_detect_cell(file_keys, cell_features)
