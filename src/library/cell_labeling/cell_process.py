from collections import defaultdict
import os, sys, glob, json
from datetime import datetime
import re
import inspect
import gzip
import shutil
import struct
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from pathlib import Path
import csv
import numpy as np
from compress_pickle import dump, load
import pandas as pd
import polars as pl #replacement for pandas (multi-core)
from tqdm import tqdm
import xgboost as xgb
import psutil
import warnings
from library.cell_labeling.del_cell_detector_trainer import CellDetectorTrainer
from library.utilities.cell_utilities import (
    calculate_correlation_and_energy,
    find_connected_segments,
    load_image,
    features_using_center_connected_components,
    find_available_backup_filename
)
from library.cell_labeling.cell_manager import CellSegmenter
from library.cell_labeling.cell_annotations import CellAnnotations
from library.cell_labeling.cell_ui import Cell_UI
from library.controller.sql_controller import SqlController
from library.image_manipulation.prep_manager import PrepCreater
from library.image_manipulation.file_logger import FileLogger
from library.image_manipulation.filelocation_manager import FileLocationManager
from library.image_manipulation.parallel_manager import ParallelManager

from library.utilities.utilities_process import M_UM_SCALE, SCALING_FACTOR, get_scratch_dir, random_string, read_image, get_hostname
import uuid

try:
    from settings import data_path, host, schema
except ImportError:
    print('Missing settings using defaults')
    data_path = "/net/birdstore/Active_Atlas_Data/data_root"
    host = "db.dk.ucsd.edu"
    schema = "brainsharer"


class CellMaker(
    ParallelManager,
    PrepCreater,
    CellSegmenter,
    CellAnnotations,
    Cell_UI):
    """
    This is the main class that handles the CellBoost functionality
    """
    TASK_SEGMENT = "Creating cell segmentation"
    TASK_DETECT = "Creating cell detections"
    TASK_EXTRACT = "Extracting cell labels and features; store as annotation set"
    TASK_TRAIN = "[Re]training CellBoost model"
    TASK_NG_PREVIEW = "Creating neuroglancer preview"

    def __init__(self, animal: str, 
                 channel: int = None,
                 task: str = '', step: int = None, 
                 model: str = "",
                 run_pruning: bool = False,
                 prune_x_range: list[int] | None = None, 
                 prune_y_range: list[int] | None = None,
                 prune_amin: int = 100,
                 prune_amax: int = 10000,
                 annotation_id: int = "", 
                 sampling: int = 0, 
                 segment_size_min: int = 100, 
                 segment_size_max: int = 100000, 
                 segment_gaussian_sigma: int = 350, 
                 segment_gaussian_kernel: int = 401, 
                 segment_threshold: int = 2000, 
                 cell_radius: int = 40, 
                 process_range: list[int] | None = None, 
                 prune_annotation_ids: list[int] | int | None = None, 
                 prune_combine_method: str = "union",
                 arg_uuid: str = None,
                 debug: bool = False):
        """Set up the class with the name of the file and the path to it's location."""
        self.animal = animal
        self.task = task
        self.debug = debug
        self.fileLocationManager = FileLocationManager(animal, data_path=data_path)
        self.sqlController = SqlController(animal)
        self.hostname = get_hostname()
        self.step = step
        self.model = model
        self.channel = channel
        self.use_scratch = False # set to True to use scratch space (defined in - utilities.utilities_process::get_scratch_dir)
        self.SCRATCH = get_scratch_dir()
        self.section_count = 0
        self.fileLogger = FileLogger(self.fileLocationManager.get_logdir(), self.debug)
        #####TODO put average cell someplace better
        self.avg_cell_img_file = Path(os.getcwd(), 'src', 'library', 'cell_labeling', 'average_cell_image.pkl')
        self.available_memory = int((psutil.virtual_memory().free / 1024**3) * 0.8)
        # These channels need to be defined for the create features process
        self.dye_channel = 0
        self.virus_channel = 0
        self.annotation_id = annotation_id
        self.process_sections = process_range
        if arg_uuid:
            self.set_id = arg_uuid #for debug of prev. uuid
        else:
            self.set_id = uuid.uuid4().hex #for tracking unique: cell_labels, histogram, image DIFF layer, annotation set
        self.cell_label_path = Path(self.fileLocationManager.prep, 'cell_labels' + '_' + str(self.set_id))
        self.downsample = False

        #SEGMENTATION PARAMETERS
        self.segment_size_min = segment_size_min
        self.segment_size_max = segment_size_max
        self.gaussian_blur_standard_deviation_sigmaX = segment_gaussian_sigma
        self.gaussian_blur_kernel_size_pixels = (segment_gaussian_kernel, segment_gaussian_kernel) #in pixels
        self.segmentation_threshold = segment_threshold #in pixels
        self.cell_radius = cell_radius #in pixels

        self.segmentation_make_smaller = False
        self.ground_truth_filename = 'ground_truth.csv'
        self.sampling = sampling

        #PRUNING PARAMETERS
        self.run_pruning = run_pruning
        self.prune_x_range = prune_x_range if prune_x_range and len(prune_x_range) >= 2 else []
        self.prune_y_range = prune_y_range if prune_y_range and len(prune_y_range) >= 2 else []
        self.prune_amin = prune_amin
        self.prune_amax = prune_amax
        self.prune_annotation_ids = prune_annotation_ids
        self.prune_combine_method = prune_combine_method


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
        print("\tstart time:".ljust(20), f"{str(datetime.now())}".ljust(20))
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
            
            self.OUTPUT = self.OUTPUT + '_' + str(self.set_id)

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

                # verify you have >=2 required channels
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
                        "mode": "counterstain",
                        "description": "NeurotraceBlue",
                        "channel_name": "C1",
                    },
                    3: {"mode": "label_of_interest", "description": "GFP", "channel_name": "C3"},
                }
                meta_data_info["Neuroanatomical_tracing"] = self.meta_channel_mapping

                with open(meta_store, 'w') as fp:
                    json.dump(meta_data_info, fp, indent=4)

        finally:
            # CHECK IF meta_data_info['Neuroanatomical_tracing'] CONTAINS A DYE AND VIRUS CHANNEL
            modes = [channel.get('mode') for channel in meta_data_info['Neuroanatomical_tracing'].values()]
            
            if ('dye' in modes or 'counterstain' in modes) and ('virus' in modes or 'ctb' in modes or 'counterstain' in modes or 'label_of_interest' in modes):
                msg = "Neuroanatomical_tracing contains a dye/counterstain channel and either virus, ctb or label_of_interest channel."
            else:
                msg = "Neuroanatomical_tracing is missing either dye/counterstain, and either virus, ctb or label_of_interest channel."
                if self.debug:
                    print(msg)
                self.fileLogger.logevent(msg)
                raise ValueError(msg)
        
        # check for full-resolution images (tiff or ome-zarr)
        # checks tiff directory first, then ome-zarr [but must have at least 1 to proceed]
        for key, value in self.meta_channel_mapping.items():
            if value['mode'] == 'dye' or value['mode'] == 'counterstain':
                counterstain_channel = value.get('channel_name')
            else:
                if value['mode'] == 'virus' or value['mode'] == 'ctb' or value['mode'] == 'label_of_interest':
                    label_of_interest_channel = value.get('channel_name')
        
        if isinstance(counterstain_channel, (list, tuple)) and len(counterstain_channel) > 1:
            # Handle list/tuple case: take the second element and strip 'C' (if it's a string)
            channel = counterstain_channel[1]
            self.counterstain_channel = self.dye_channel = (
                str(channel).lstrip('C')  # Ensure it's a string before stripping
                if isinstance(channel, str) 
                else channel  # Keep as-is if not a string (e.g., int)
            )
        else:
            # Handle string case: strip 'C' if it starts with 'C'
            self.counterstain_channel = self.dye_channel = (
                str(counterstain_channel)[1:].lstrip('C')  # Ensure string, slice, then strip
                if isinstance(counterstain_channel, str) and counterstain_channel.startswith('C')
                else counterstain_channel  # Fallback (int, non-'C' strings, etc.)
            )
        if isinstance(label_of_interest_channel, (list, tuple)) and len(label_of_interest_channel) > 1:
            # Handle list/tuple: take second element, strip 'C' if it's a string
            channel = label_of_interest_channel[1]
            self.label_of_interest_channel = self.virus_channel = (
                str(channel).lstrip('C')  # Force string and strip 'C'
                if isinstance(channel, str) 
                else channel  # Keep as-is if not a string (e.g., int)
            )
        else:
            # Handle string (strip 'C' if present) or non-list inputs
            self.label_of_interest_channel = self.virus_channel = (
                str(label_of_interest_channel)[1:].lstrip('C')  # Ensure string, slice, strip
                if isinstance(label_of_interest_channel, str) and label_of_interest_channel.startswith('C')
                else label_of_interest_channel  # Fallback (int, non-'C' strings, etc.)
            )

        if self.channel: #labeled channel argument ALWAYS takes precedence
            self.label_of_interest_channel = self.channel

        found_dye_channel = False
        found_virus_marker_channel = False
        
        #21-AUG-2025 Song-Mao special processing
        # INPUT_dye = Path(self.fileLocationManager.get_full(channel=self.counterstain_channel))
        # INPUT_virus_marker = Path(self.fileLocationManager.get_full(channel=self.label_of_interest_channel))
        INPUT_dye = Path(self.fileLocationManager.get_full_aligned(channel=self.counterstain_channel))
        INPUT_virus_marker = Path(self.fileLocationManager.get_full_aligned(channel=self.label_of_interest_channel))
        
        if INPUT_dye.exists():
            if self.debug:
                print(f'Full-resolution tiff stack found (counterstain channel): {INPUT_dye}')
            self.fileLogger.logevent(f'Full-resolution tiff stack found (counterstain channel): {INPUT_dye}')
            found_dye_channel = True
        else:
            print(f'Full-resolution tiff stack not found (counterstain channel). Expected location: {INPUT_dye}')
            print('UNABLE TO PROCEED; EXITING')
            sys.exit(1)
        if INPUT_virus_marker.exists():
            if self.debug:
                print(f'Full-resolution tiff stack found (label_of_interest/virus/tracer channel): {INPUT_virus_marker}')
            self.fileLogger.logevent(f'Full-resolution tiff stack found (label_of_interest/virus/tracer channel): {INPUT_virus_marker}')
            found_virus_marker_channel = True
        else:
            print(f'Full-resolution tiff stack not found (label_of_interest/virus/tracer channel). Expected location: {INPUT_virus_marker}')
            print('UNABLE TO PROCEED; EXITING')
            sys.exit(1)

        if found_dye_channel == False or found_virus_marker_channel == False:
            print(f'One or more full-resolution tiff stacks not found. Expected locations:')
            if not found_dye_channel:
                print(f' - Counterstain channel: {INPUT_dye}')
            if not found_virus_marker_channel:
                print(f' - Label of interest/virus/tracer channel: {INPUT_virus_marker}')
        #OME-Zarr NOT CURRENTLY SUPPORTED
        # if found_dye_channel == False:
        #     INPUT_dye = Path(self.fileLocationManager.get_neuroglancer(False, channel=dye_channel[1]) + '.zarr')
        #     INPUT_virus_marker = Path(self.fileLocationManager.get_neuroglancer(False, channel=virus_marker_channel[1]) + '.zarr')
        #     if INPUT_dye.exists():
        #         if self.debug:
        #             print(f'Full-resolution ome-zarr stack found (dye channel): {INPUT_dye}')
        #         self.fileLogger.logevent(f'Full-resolution ome-zarr stack found (dye channel): {INPUT_dye}')
        #     else:
        #         print(f'Full-resolution ome-zarr stack not found (dye channel). Expected location: {INPUT_dye}; Exiting')
        #         sys.exit(1)
        # if found_virus_marker_channel == False:
        #     if INPUT_virus_marker.exists():
        #         if self.debug:
        #             print(f'Full-resolution ome-zarr stack found (virus/tracer channel): {INPUT_virus_marker}')
        #         self.fileLogger.logevent(f'full-resolution ome-zarr stack found (virus/tracer channel): {INPUT_virus_marker}')
        #     else:
        #         print(f'Full-resolution ome-zarr stack not found (virus/tracer channel). expected location: {INPUT_virus_marker}; exiting')
        #         sys.exit(1)

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

        return (INPUT_dye, INPUT_virus_marker, meta_data_info) #used for create_features


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

        _, section, str_section_number, _, _, _, _, _, _, SCRATCH, _, avg_cell_img, _, _, _, _, _, task, set_id, _, _, debug = file_keys

        print(f'Starting function: calculate_features with {len(cell_candidate_data)} cell candidates')
        
        output_path = Path(SCRATCH, 'pipeline_tmp', self.animal, 'cell_features_' + str(set_id))
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = Path(output_path, f'cell_features_{str_section_number}.csv')

        #TODO check if features already extracted
        if os.path.exists(output_file):
            if self.debug:
                print(f'Cell features already extracted. Using: {output_file}')
            df_features = pl.read_csv(output_file)
            return df_features
        else:
            output_spreadsheet = []

        # STEP 3-B) load information from cell candidates (pickle files from step 2 - cell candidate identification) **Now passed as parameter**
        for idx, cell in enumerate(cell_candidate_data):
            # STEP 3-C1, 3-C2) calculate_correlation_and_energy FOR CHANNELS 1 & 3 (ORG. FeatureFinder.py; calculate_features())
            results = {}
            results['ch1_corr'] = results['ch1_energy'] = results['ch3_corr'] = results['ch3_energy'] =  results['ch1_contrast'] = results['ch3_contrast'] = 0.0
            if task != 'segment': #no feature calc for segment task
                results['ch1_corr'], results['ch1_energy'] = calculate_correlation_and_energy(avg_cell_img["CH1"], cell['image_CH1'])
                results['ch3_corr'], results['ch3_energy'] = calculate_correlation_and_energy(avg_cell_img["CH3"], cell['image_CH3'])
                results['ch1_contrast'], results['ch3_contrast'], results['moments_data'] = features_using_center_connected_components(cell, debug=debug)

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
                "corr_CH1": results['ch1_corr'],
                "energy_CH1": results['ch1_energy'],
                "corr_CH3": results['ch3_corr'],
                "energy_CH3": results['ch3_energy'],
                "contrast1": results['ch1_contrast'],
                "contrast3": results['ch3_contrast']
            }
            if task != 'segment':
                spreadsheet_row.update(results['moments_data'][0])  # Regular moments
                spreadsheet_row.update(results['moments_data'][1])  # Hu moments
            output_spreadsheet.append(spreadsheet_row)

            #TODO: EXPLORE IF HARD-CODED VARIABLES MAKE ANY DIFFERENCE HERE
            #TODO: remove - seems like testing code
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
                    maskpath = Path(self.prep, 'mask.npy')
                    # maskpath = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps/mask.npy'
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

        _, section, str_section_number, _, _, _, _, _, _, _, OUTPUT, _, model_filename, _, _, _, _, task, set_id, _, _, debug = file_keys

        if self.debug:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} Start")

        if task == 'segment':
            print(f'WARNING: Task is "segment"; no cell scoring/detection will be performed')
        else:        
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
            if 'idx' not in features.columns: #temp 'dummy' column; remove before model training
                features = features.with_columns(
                    pl.lit(-1).alias('idx')  # Dummy value, or replace with meaningful default
                )
            # Convert Polars DataFrame to pandas (XGBoost DMatrix prefers pandas)
            features_pd = features.to_pandas()
            model_features = model[0].feature_names
            missing = set(model_features) - set(features_pd.columns)
            if missing:
                    raise ValueError(f"Missing features in input: {missing}")

            # Reorder and filter to match model
            features_pd = features_pd[model_features]

            all_data = xgb.DMatrix(features_pd)
            scores=np.zeros([features_pd.shape[0], len(model)])

            for i, bst in enumerate(model):
                best_ntree_limit = int(bst.attributes().get("best_ntree_limit", 676))
                scores[:, i] = bst.predict(all_data, iteration_range=[1, best_ntree_limit], output_margin=True)

            mean_scores = np.mean(scores, axis=1)
            std_scores = np.std(scores, axis=1)

            # for i, bst in enumerate(model):
            #     attributes = bst.attributes()
            #     try:
            #         best_ntree_limit = int(attributes["best_ntree_limit"])
            #     except KeyError:
            #         best_ntree_limit = 676
            #     scores[:, i] = bst.predict(all_data, iteration_range=[1, best_ntree_limit], output_margin=True)

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

                if mean_score > threshold:  # Candidate is a cell/neuron
                    classification = 2
                elif -threshold <= mean_score <= threshold:
                    classification = 0  # UNKNOWN/UNSURE

                predictions.append(classification)

            return predictions

        drops = ['animal', 'section', 'index', 'row', 'col']
        try:        
            cell_features_selected_columns = cell_features.drop(drops)
        except:
            print(f'CELL FEATURES MISSING COLUMN(S); CHECK cell_features_{str_section_number}.csv FILE')
            print(cell_features.columns)
            sys.exit(1)

        if task != 'segment':
            # Step 4-2-1-2) calculate_scores(features) - calculates scores, labels, mean std for each feature
            if model_filename: #IF MODEL FILE EXISTS, USE TO SCORE
                mean_scores, std_scores = calculate_scores(cell_features_selected_columns, model_file)
                cell_features = cell_features.with_columns([
                    pl.Series("mean_score", mean_scores),
                    pl.Series("std_score", std_scores),
                    pl.Series("predictions", get_prediction_and_label(mean_scores))
                ])
        else:
            # IF TASK IS 'segment', NO SCORING IS DONE; ONLY CELL CANDIDATES ARE IDENTIFIED
            cell_features = cell_features.with_columns([
                pl.lit(0.0).alias("mean_score"),  # Use alias() to name the column
                pl.lit(0.0).alias("std_score"),
                pl.lit(2).alias("predictions")  # 2 is putative cell
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

    #TODO: possibly deprecated (see extract_predictions)
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
        #precomputed, annotation volume with spatial indexing
        labels = ['ML_POSITIVE']
        sampling = self.sampling

        xy_resolution = self.sqlController.scan_run.resolution
        z_resolution = self.sqlController.scan_run.zresolution
        w = int(self.sqlController.scan_run.width)
        h = int(self.sqlController.scan_run.height)
        z_length = len(os.listdir(self.fileLocationManager.section_web)) #SHOULD ALWAYS EXIST FOR ALL BRAIN STACKS

        #READ, CONSOLIDATE PREDICTION FILES, SAMPLE AS NECESSARY
        dfpath = os.path.join(self.fileLocationManager.prep, 'cell_labels', 'all_predictions.csv')
        if os.path.exists(self.cell_label_path):
            print(f'Parsing cell labels from {self.cell_label_path}')
        else:
            print(f'ERROR: {self.cell_label_path} not found')
            sys.exit(1)
        
        detection_files = sorted(glob.glob(os.path.join(self.cell_label_path, f'detections_*.csv') ))
        if len(detection_files) == 0:
            print(f'Error: no csv files found in {self.cell_label_path}')
            sys.exit(1)

        dfs = []
        for file_path in detection_files:
            # Read CSV with Polars (much faster than pandas)
            try:
                df = pl.read_csv(file_path)
                if df.is_empty():
                    continue
                    
                # Filter and process in one go
                filtered = df.filter(
                (pl.col("predictions").cast(pl.Float32) > 0))
                
                if filtered.is_empty():
                    continue
                
                # Append to dataframe data
                dfs.append(filtered.select([
                    pl.col("col").alias("x"),
                    pl.col("row").alias("y"),
                    (pl.col("section") - 0.5).cast(pl.Int32).alias("section")
                ]))
            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        # Concatenate all dataframes at once
        final_df = pl.concat(dfs) if dfs else pl.DataFrame()

        #random sampling for model training, if selected
        # Choose between full dataset or sampled subset
        if self.sampling:
            sampled_df = final_df.sample(n=sampling, seed=42) if final_df.height > sampling else final_df
            df = sampled_df
            print(f"Sampling enabled - randomly sampled neuron count: {len(df)}")
        else:
            df = final_df
            print(f"No sampling - using all {df.height} points")
        
        if len(df) == 0:
            print('No neurons found')
            sys.exit()

        ###############################################
        #ALL ANNOTATION POINTS ARE STORED IN df VARIABLE AT THIS POINT
        ###############################################

        ###############################################
        # SAVE OUTPUTS
        ###############################################
        # SAVE CSV FORMAT (x,y,z POINTS IN CSV) *SAMPLED DATA
        if not df.is_empty():
            df.write_csv(dfpath)

        # SAVE JSON FORMAT *SAMPLED DATA
        annotations_dir = Path(self.fileLocationManager.neuroglancer_data, 'annotations') #ref 'drawn_directory
        annotations_dir.mkdir(parents=True, exist_ok=True)
        annotations_file = str(Path(annotations_dir, labels[0]+'.json'))
        # Populate points variable after sampling
        sampled_points = (
            df.sort("section")
            .select(["x", "y", "section"])
            .to_numpy()
            .tolist()
        )
        with open(annotations_file, 'w') as fh:
            json.dump(sampled_points, fh)

        ###########################################################
        # PRECOMPUTED ANNOTATION VOLUME CREATION
        ###########################################################
        def write_simple_annotations(output_dir, points):
            # Create required directories
            by_id_dir = os.path.join(output_dir, 'by_id')
            os.makedirs(by_id_dir, exist_ok=True)

            # Build label mapping for enum_values/enum_labels
            label_to_value = {}
            enum_labels = []
            enum_values = []
            for pt, label in points:
                if label not in label_to_value:
                    label_to_value[label] = len(enum_labels)
                    enum_labels.append(label)
                    enum_values.append(len(enum_values))

            # Create info file (compliant with spec, with enum)
            info = {
                "@type": "neuroglancer_annotations_v1",
                "dimensions": {
                    "x": [1, "nm"],
                    "y": [1, "nm"],
                    "z": [1, "nm"]
                },
                "lower_bound": [0, 0, 0],
                "upper_bound": [1000, 1000, 1000],  # Adjust as needed
                "annotation_type": "POINT",
                "properties": [
                    {
                        "id": "label",
                        "type": "uint32",
                        "enum_values": enum_values,
                        "enum_labels": enum_labels
                    }
                ],
                "relationships": [],
                "by_id": {
                    "key": "by_id"
                },
                "spatial": [
                    {
                        "key": "spatial0",
                        "grid_shape": [1, 1, 1],
                        "chunk_size": [1000, 1000, 1000],
                        "limit": 1000
                    }
                ]
            }
            with open(os.path.join(output_dir, 'info'), 'w') as f:
                json.dump(info, f, indent=2)

            # Write each annotation as a separate file in by_id
            annotation_ids = []
            for idx, (pt, label) in enumerate(points):
                ann_id = idx + 1  # uint64 id, start from 1
                annotation_ids.append(ann_id)
                ann_path = os.path.join(by_id_dir, str(ann_id))
                with open(ann_path, 'wb') as f:
                    # Write position as 3 float32
                    f.write(struct.pack('<3f', *pt))
                    # Write label as uint32 (property)
                    f.write(struct.pack('<I', label_to_value[label]))
                    # No relationships, so nothing more

            # --- Write spatial index (spatial0/0_0_0) ---
            spatial_dir = os.path.join(output_dir, 'spatial0')
            os.makedirs(spatial_dir, exist_ok=True)
            spatial_path = os.path.join(spatial_dir, '0_0_0')
            with open(spatial_path, 'wb') as f:
                # Number of annotations (uint64le)
                f.write(struct.pack('<Q', len(points)))
                # For each annotation: position (3 float32), label (uint32)
                for idx, (pt, label) in enumerate(points):
                    ann_id = idx + 1
                    f.write(struct.pack('<3f', *pt))
                    f.write(struct.pack('<I', label_to_value[label]))
                # For each annotation: annotation id (uint64le)
                for ann_id in annotation_ids:
                    f.write(struct.pack('<Q', ann_id))

        def write_precomputed_annotations(sampled_points, output_dir):

            #testing
            sampled_points = [
                (19498, 12953, 70),
                (19319, 12953, 70),
                (19319, 15948, 70)
            ]
            xy_resolution = 325  # nanometers
            z_resolution = 20000  # nanometers
            print(f'DEBUG= {xy_resolution=},{xy_resolution=},{z_resolution=}')
            print(f'DEBUG: {sampled_points=}')

            # Validate inputs [for testing]
            if not sampled_points:
                raise ValueError("sampled_points is empty")
            if not all(len(pt) == 3 for pt in sampled_points):
                raise ValueError("Each point must have 3 coordinates (x, y, z)")
            if not (xy_resolution > 0 and z_resolution > 0):
                raise ValueError(f"Invalid resolutions: xy_resolution={xy_resolution}, z_resolution={z_resolution}")
            
            # Scale sampled_points from voxels to nanometers
            scaled_points = [
                [x * xy_resolution, y * xy_resolution, z * z_resolution]
                for x, y, z in sampled_points
            ]
            
            print(f"DEBUG: \"Scaled\" points sample: {scaled_points[:2]}")
            
            #CONVERSION FROM micrometers (μm) to meters (m)
            # Use voxel units (no conversion to micrometers or meters)
            # dimensions = {
            #     "x": [str(xy_resolution), "voxels"],  # No conversion, using voxel resolution
            #     "y": [str(xy_resolution), "voxels"],  # Same for y
            #     "z": [str(z_resolution), "voxels"]   # Same for z
            # }
            dimensions = {
                "x": [str(xy_resolution), "nm"],
                "y": [str(xy_resolution), "nm"],
                "z": [str(z_resolution), "nm"]
            }
            print(f'DEBUG: {dimensions=}')

            shape = (w, h, z_length)
            print(f'DEBUG: {shape=}')
            preferred_chunk_size = (128, 128, 64)
            # adjusted_chunk_size = tuple(min(p, s) for p, s in zip(preferred_chunk_size, shape))

            #adjust grid_shape using physical units (nanometers):
            adjusted_chunk_size = (
                preferred_chunk_size[0] * xy_resolution,
                preferred_chunk_size[1] * xy_resolution,
                preferred_chunk_size[2] * z_resolution,
            )

            # Sharding configuration (1 shard per 256 annotations, min 2 shards)
            num_annotations = len(sampled_points)
            annotations_per_shard = 256
            num_shards = max(2, (num_annotations + annotations_per_shard - 1) // annotations_per_shard)
            # shard_bits = int(np.ceil(np.log2(num_shards)))
            shard_bits = 8
            minishard_bits = 8
            preshift_bits = 0
            #print(f'DEBUG: shard count: {num_shards} ({annotations_per_shard/shard})')

            # Create sharding specification
            sharding_spec = {
                "@type": "neuroglancer_uint64_sharded_v1",
                "hash": "identity",
                "minishard_bits": minishard_bits,
                "shard_bits": shard_bits,
                "preshift_bits": preshift_bits,
                "minishard_index_encoding": "gzip",
                "data_encoding": "gzip"
            }
            print(f'DEBUG: {sharding_spec=}')

            # Calculate bounds
            lower_bound = [0, 0, 0]
            # upper_bound = list(shape)

            #Calculate upper_bound based on scaled points
            xs, ys, zs = zip(*scaled_points)
            upper_bound = [
                int(np.ceil(max(xs))),
                int(np.ceil(max(ys))),
                int(np.ceil(max(zs)))
            ]

            if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in lower_bound + upper_bound):
                raise ValueError("Invalid bounds calculated from sampled_points")
            print(f"DEBUG: Bounds: lower={lower_bound}, upper={upper_bound}")

            #adjust chunk_size accordingly using physical units (nanometers):
            grid_shape = [
                int(np.ceil((upper_bound[0] - lower_bound[0]) / adjusted_chunk_size[0])),
                int(np.ceil((upper_bound[1] - lower_bound[1]) / adjusted_chunk_size[1])),
                int(np.ceil((upper_bound[2] - lower_bound[2]) / adjusted_chunk_size[2]))
            ]
            print(f"DEBUG: Grid shape: {grid_shape}")

            # Estimate limit (max annotations per grid cell)
            # Use the maximum number of points in any chunk, or a default
            points_by_chunk = defaultdict(list)

            for i, (x, y, z) in enumerate(scaled_points): #chunk_key must be computed from scaled_points
                chunk_x = int(x // adjusted_chunk_size[0])
                chunk_y = int(y // adjusted_chunk_size[1])
                chunk_z = int(z // adjusted_chunk_size[2])
                chunk_key = (chunk_x, chunk_y, chunk_z)
                points_by_chunk[chunk_key].append((i, x, y, z))
            limit = max(len(points) for points in points_by_chunk.values()) if points_by_chunk else 1000

            # # Calculate chunk_size (physical size in meters)
            # chunk_size = [
            #     adjusted_chunk_size[0] * (xy_resolution / 1_000_000),
            #     adjusted_chunk_size[1] * (xy_resolution / 1_000_000),
            #     adjusted_chunk_size[2] * (z_resolution / 1_000_000),
            # ]
            #chunk_size needs physical units [nanometers to meters]
            chunk_size = [
                adjusted_chunk_size[0] / 1_000_000,
                adjusted_chunk_size[1] / 1_000_000,
                adjusted_chunk_size[2] / 1_000_000,
            ]
            # chunk_size = [
            #     preferred_chunk_size[0] * xy_resolution / 1_000_000,
            #     preferred_chunk_size[1] * xy_resolution / 1_000_000,
            #     preferred_chunk_size[2] * z_resolution / 1_000_000,
            # ]

            # Create info dictionary
            info = {
                "@type": "neuroglancer_annotations_v1",
                "annotation_type": "POINT",
                "by_id": {
                    "key": "by_id/"
                },
                "dimensions": dimensions,
                "by_type": {
                    "POINT": "by_type/POINT"
                },
                "spatial": [
                    {
                        "key": "spatial_index/",
                        "grid_shape": grid_shape,
                        "chunk_size": chunk_size,
                        "limit": 1000
                    }
                ],
                "properties": [
                    {
                        "id": "radius",
                        "type": "float32",
                        "default": 10.0  # Point size in nanometers
                    }
                ],
                "relationships": [],
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "sharding": sharding_spec
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "info").write_text(json.dumps(info, indent=2))

            # Write sharded annotation ID index
            id_dir = output_dir / "by_id"
            id_dir.mkdir(parents=True, exist_ok=True)

            # Group annotations by shard
            annotations_by_shard = defaultdict(list)
            for i, (x, y, z) in enumerate(scaled_points):
                shard_index = (i >> minishard_bits) % (1 << shard_bits)
                annotations_by_shard[shard_index].append((i, x, y, z))

            # Write shard files
            for shard_index, annotations in annotations_by_shard.items():
                shard_file = id_dir / f"{shard_index:016x}"
                minishards = defaultdict(list)
                for i, x, y, z in annotations:
                    minishard_index = i % (1 << minishard_bits)
                    annotation_data = json.dumps({
                        "@type": "neuroglancer_annotation",
                        "id": str(i),
                        "point": [float(x), float(y), float(z)],
                        "type": "point",  # Lowercase to match schema
                        "properties": {"radius": 5.0}
                    }).encode('utf-8')
                    compressed_data = gzip.compress(annotation_data)
                    minishards[minishard_index].append((i, compressed_data))

                with shard_file.open("wb") as fh:
                    minishard_index_data = []
                    offset = 0
                    for minishard_idx in range(1 << minishard_bits):
                        if minishard_idx in minishards:
                            entries = minishards[minishard_idx]
                            entries.sort(key=lambda x: x[0])
                            minishard_start = offset
                            packed_data = b"".join(compressed_data for i, compressed_data in entries)
                            fh.write(packed_data)
                            offset += len(packed_data)
                            minishard_end = offset
                            minishard_index_data.append((minishard_idx, minishard_start, minishard_end))
                        else:
                            minishard_index_data.append((minishard_idx, offset, offset))

                    index_data = b""
                    for minishard_idx, start, end in minishard_index_data:
                        # index_data += struct.pack("<QQQ", minishard_idx, start, end)
                        index_data += struct.pack("<QQ", start, end)
                    compressed_index = gzip.compress(index_data)
                    fh.write(compressed_index)

            # Write annotation type index
            type_dir = output_dir / "by_type/POINT"
            type_dir.mkdir(parents=True, exist_ok=True)
            type_file = type_dir / "0"
            with type_file.open("wb") as fh:
                fh.write(struct.pack("<I", len(sampled_points)))
                for i in range(len(sampled_points)):
                    fh.write(struct.pack("<Q", i))

            # Write spatial index per chunk
            spatial_dir = output_dir / "spatial_index"
            spatial_dir.mkdir(parents=True, exist_ok=True)
            for chunk_key, points in points_by_chunk.items():
                chunk_filename = f"{chunk_key[0]}_{chunk_key[1]}_{chunk_key[2]}.gz"
                path = spatial_dir / chunk_filename
                try:
                    with gzip.open(path, "wb") as fh:
                        fh.write(struct.pack("<I", len(points)))
                        packed_data = b"".join(
                            struct.pack("<Qfff", pid, float(x), float(y), float(z))
                            for pid, x, y, z in points
                        )
                        fh.write(packed_data)
                except IOError as e:
                    raise IOError(f"Failed to write spatial index file {path}: {e}")

        out_dir = Path(annotations_dir, labels[0] + '.precomputed_ann1')
        if os.path.exists(out_dir):
            print(f'Removing existing directory {out_dir}')
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        print(f'Creating precomputed annotations in {out_dir}')
        # write_precomputed_annotations(sampled_points, out_dir)

        sampled_points = [
            (19498, 12953, 70),  # Example points (x, y, z) in voxel space
            (19319, 12953, 70),
            (19319, 15948, 70)
        ]
        write_simple_annotations(out_dir, sampled_points)
    


        
    # LIKELY DEPRECATED (PROCESSES TEXT-BASED ANNOTATIONS, AND FREQUENTLY HAS TOO MANY) - BETTER TO USE extract_predictions_precomputed
    # TODO: remove                 
    def extract_predictions2(self):
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
        FK_user_id = 1
        labels = ['MACHINE_SURE']

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
        sampling = self.sampling
        found = 0

        dfpath = os.path.join(self.fileLocationManager.prep, 'cell_labels', 'all_predictions.csv')
        if os.path.exists(self.cell_label_path):
            print(f'Parsing cell labels from {self.cell_label_path}')
        else:
            print(f'ERROR: {self.cell_label_path} not found')
            sys.exit(1)
        
        detection_files = sorted(glob.glob(os.path.join(self.cell_label_path, f'detections_*.csv') ))
        if len(detection_files) == 0:
            print(f'Error: no csv files found in {self.cell_label_path}')
            sys.exit(1)

        dfs = []
        for file_path in detection_files:
            # Read CSV with Polars (much faster than pandas)
            try:
                df = pl.read_csv(file_path)
                if df.is_empty():
                    continue
                    
                # Filter and process in one go
                filtered = df.filter(
                (pl.col("predictions").cast(pl.Float32) > 0))
                
                if filtered.is_empty():
                    continue
                    
                # Process all rows at once
                x_vals = (filtered["col"].cast(pl.Float32) / M_UM_SCALE * xy_resolution).to_list()
                y_vals = (filtered["row"].cast(pl.Float32) / M_UM_SCALE * xy_resolution).to_list()
                sections = ((filtered["section"].cast(pl.Float32) + 0.5) * z_resolution / M_UM_SCALE).to_list()
                
                # Append to dataframe data
                dfs.append(filtered.select([
                    pl.col("col").alias("x"),
                    pl.col("row").alias("y"),
                    (pl.col("section") - 0.5).cast(pl.Int32).alias("section")
                ]))
                
                # Generate annotations
                for x, y, section in zip(x_vals, y_vals, sections):
                    found += 1
                    point = [x, y, section]
                    childJsons.append({
                        "point": point,
                        "type": "point",
                        "parentAnnotationId": f"{parent_id}",
                        "props": default_props
                    })
                    points.append(point)
            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        print(f'Found {found} total neurons')
        if found == 0:
            print('No neurons found')
            sys.exit()

        # Concatenate all dataframes at once
        final_df = pl.concat(dfs) if dfs else pl.DataFrame()

        #random sampling for model training, if selected
        # Choose between full dataset or sampled subset
        if self.sampling:
            sampled_df = final_df.sample(n=sampling, seed=42) if final_df.height > sampling else final_df
            df = sampled_df
            print(f"Sampling enabled - randomly sampled neuron count: {len(df)}")
        else:
            df = final_df
            print(f"No sampling - using all {df.height} points")

        # Define the voxel size (in nm)
        voxel_size = (
            int(xy_resolution * 1000),
            int(xy_resolution * 1000),
            int(z_resolution * 1000)
        )
        # Voxel coordinate computation from selected df
        x_vox = (df["x"].cast(pl.Float32) / M_UM_SCALE * xy_resolution)
        y_vox = (df["y"].cast(pl.Float32) / M_UM_SCALE * xy_resolution)
        z_vox = ((df["section"].cast(pl.Float32) + 0.5) * z_resolution / M_UM_SCALE)

        # Bounding box in nm
        x_min = int(x_vox.min() * voxel_size[0])
        y_min = int(y_vox.min() * voxel_size[1])
        z_min = int(z_vox.min() * voxel_size[2])

        x_max = int((x_vox.max() + 1) * voxel_size[0])
        y_max = int((y_vox.max() + 1) * voxel_size[1])
        z_max = int((z_vox.max() + 1) * voxel_size[2])

        lower_bound = [x_min, y_min, z_min]
        upper_bound = [x_max, y_max, z_max]

        # Final export annotations (replacing earlier block)
        x_list = x_vox.to_list()
        y_list = y_vox.to_list()
        z_list = z_vox.to_list()

        childJsons_final = []
        points_final = []

        for x, y, z in zip(x_list, y_list, z_list):
            point = [x, y, z]
            childJsons_final.append({
                "point": point,
                "type": "point",
                "parentAnnotationId": f"{parent_id}",
                "props": default_props
            })
            points_final.append(point)

        cloud_points = {
            "source": points_final[0],
            "centroid": np.mean(points_final, axis=0).tolist(),
            "childrenVisible": True,
            "type": "cloud",
            "description": labels[0],
            "sessionID": f"{parent_id}",
            "props": default_props,
            "childJsons": childJsons_final
        }

        export_points = points_final

        
        # Save outputs
         ###############################################
        # 'workaround' for db timeout [save to file and then sql insert from file]
        # long-term store in www folder for direct import
        annotations_dir = Path(self.fileLocationManager.neuroglancer_data, 'annotations')
        annotations_dir.mkdir(parents=True, exist_ok=True)
        annotations_file = str(Path(annotations_dir, labels[0]+'.json'))
        with open(annotations_file, 'w') as fh:
            json.dump(cloud_points, fh)
            
        if not sampled_df.is_empty():
            sampled_df.write_csv(dfpath)

        #GRAPHENE EXPORT - NOT WORKING
        #ref: https://github.com/seung-lab/cloud-volume/wiki/Graphene
        # annotations_np = sampled_df.to_numpy()
        # annotations_file_graphene = Path(annotations_dir) / f"{labels[0]}.graphene"
        # annotations_graphene = Graphene(labels[0])
        # for annotation in annotations_np:
        #     x, y, section = annotation
        #     point = np.array([x, y, section])  # x, y, z coordinates
        #     annotations_graphene.add_point(point, id=len(annotations_graphene.points))  # add point annotation
        # serialized_annotations = annotations_graphene.serialize()
        # with open(annotations_file_graphene, "wb") as fh:
        #     fh.write(serialized_annotations)

        #ZARR2 EXPORT - NOT WORKING
        # annotations_np = sampled_df.to_numpy()
        # annotations_file_zarr = Path(annotations_dir) / f"{labels[0]}.zarr"
        # # Create a Zarr group in v2 format
        # store = zarr.DirectoryStore(annotations_file_zarr)
        # root = zarr.group(store=store, overwrite=True)
        # annotations = []
        # for i, (x, y, z) in enumerate(annotations_np):
        #     annotations.append({
        #         "id": i,
        #         "point": [float(x), float(y), float(z)],
        #         "type": "point",
        #         "description": f"annotation {i}"
        #     })
        # annotation_data = json.dumps({"annotations": annotations})
        # annotation_bytes = annotation_data.encode('utf-8')
        # # Store the serialized JSON as a single chunk (dataset "0")
        # ds = root.create_dataset("0", shape=(len(annotation_bytes),), dtype="u1")
        # ds[:] = np.frombuffer(annotation_bytes, dtype="u1")
        # # Add metadata (.zattrs) required by Neuroglancer
        # zattrs = {
        #     "type": "neuroglancer_annotations",
        #     "annotations": {
        #         "description": f"Annotations for {labels[0]}",
        #         "annotation_type": "point"
        #     },
        #     "dimensions": {
        #         "x": [1e-9, "m"],
        #         "y": [1e-9, "m"],
        #         "z": [1e-9, "m"]
        #     }
        # }
        # with open(annotations_file_zarr / ".zattrs", "w") as f:
        #     json.dump(zattrs, f)

        #PRECOMPUTED EXPORT - (ADAPTED FROM FILE @ neuroglancer/python/neuroglancer/write_annotations.py)
        #ref: https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md#info-json-file-format

        #(annotations in binary format) -> info file created automatically by write_annotations
        output_dir = Path(annotations_dir, f"{labels[0]}.precomputed")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        #POTENTIAL ADAPTATION: ADD LABELS FOR EACH POINT (e.g. cell type)
        #############################################
        #CREATE info FILE
        #############################################
        annotations = []            
        for i, (x, y, z) in enumerate(export_points):  # Use export_points here instead of sampled_x, sampled_y, sampled_section
            annotations.append({
                "id": i,
                "point": [x, y, z],
                "properties": {
                    "label": 1
                }
            })

        info = {
            "@type": "neuroglancer_annotations_v1",
            "dimensions": {
                "x": [voxel_size[0], "nm"],
                "y": [voxel_size[1], "nm"],
                "z": [voxel_size[2], "nm"]
            },
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "annotation_type": "POINT",
            "properties": [
                {"id": "label", "type": "uint8"}
            ],
            "relationships": [],
            "by_id": {
                "key": "by_id"
            },
            "spatial": [
                {
                    "key": "spatial0",
                    "grid_shape": [1, 1, 1],
                    "chunk_size": [
                        upper_bound[0] - lower_bound[0],
                        upper_bound[1] - lower_bound[1],
                        upper_bound[2] - lower_bound[2]
                    ],
                    "limit": 10000  # how many annotations are displayed at each zoom level.
                }
            ]
        }
        # spatial= {
        #             "key": "spatial0",
        #             "grid_shape": [1, 1, 1],
        #             "chunk_size": [
        #                 upper_bound[0] - lower_bound[0],
        #                 upper_bound[1] - lower_bound[1],
        #                 upper_bound[2] - lower_bound[2]
        #             ],
        #             "limit": 10000  # how many annotations are displayed at each zoom level.
        #         }
        # chunk_size = [self.sqlController.scan_run.width, self.sqlController.scan_run.height, 348]
        
        # info = {}
        # info["@type"] = "neuroglancer_annotations_v1"
        # info["annotation_type"] = "POINT"
        # info["by_id"] = {"key":"by_id"}
        # info["dimensions"] = {"x":[str(xy_resolution),"μm"],
        #                     "y":[str(xy_resolution),"μm"],
        #                     "z":[str(z_resolution),"μm"]}
        # info["lower_bound"] = [0,0,0]
        # info["upper_bound"] = chunk_size
        # info["properties"] = []
        # info["relationships"] = []
        # info["spatial"] = [spatial]  

        # Write the 'info' file with the annotations
        with open(os.path.join(output_dir, "info"), "w") as f:
            json.dump(info, f, indent=2)

        #############################################
        #CREATE 'by-id' FOLDER (individual annotation points' labels)
        #############################################
        def encode_annotation(annotation):
            x, y, z = annotation["point"]
            
            # Ensure label is an integer
            label = 1  # Use 1 or any other integer to represent "MACHINE_SURE"
            
            # Encode as: 3 x float32 + 1 uint8 (label)
            buf = struct.pack("<fffB", x, y, z, label)
            buf += b"\x00" * (4 - len(buf) % 4)  # Pad to 4-byte alignment
            return buf


        for ann in annotations:
            binary = encode_annotation(ann)
            ann_path = os.path.join(output_dir, "by_id", str(ann["id"]))
            
            # Ensure the directory exists before writing the binary file
            os.makedirs(os.path.dirname(ann_path), exist_ok=True)

            with open(ann_path, "wb") as f:
                f.write(binary)

        #############################################
        #CREATE 'spacial0' FOLDER (individual annotation points in binary format)
        #############################################
        spatial_dir = os.path.join(output_dir, "spatial")
        os.makedirs(spatial_dir, exist_ok=True)

        def write_spatial_chunk(chunk_key, chunk_data):
            chunk_path = os.path.join(spatial_dir, f"{chunk_key}.bin")
            with open(chunk_path, "wb") as f:
                f.write(chunk_data)

        def create_spatial_chunk(annotations, chunk_size):
            # Create a chunk from annotations within the chunk bounds
            chunk_data = b""
            for annotation in annotations:
                x, y, z = annotation["point"]
                # Ensure padding is applied and convert to binary format
                buf = struct.pack("<fffB", x, y, z, annotation["properties"]["label"])
                buf += b"\x00" * (4 - len(buf) % 4)  # pad to 4-byte alignment
                chunk_data += buf
            return chunk_data

        # Example to create and write chunks (assuming chunk size and annotations are available)
        chunk_size = (64, 64, 64)  # Example chunk size (you may want to change this)
        chunk_data = create_spatial_chunk(annotations, chunk_size)
        write_spatial_chunk("spatial0", chunk_data)  # Write the spatial chunk for the given key


        ###############################################
        # if not self.debug:
        #     label_objects = self.sqlController.get_labels(labels)
        #     label_ids = [label.id for label in label_objects]

        #     annotation_session = self.sqlController.get_annotation_session(self.animal, label_ids, FK_user_id)
        #     if annotation_session is not None:
        #         self.sqlController.delete_row(AnnotationSession, {"id": annotation_session.id})

        #     try:
        #         id = self.sqlController.insert_annotation_with_labels(FK_user_id, self.animal, cloud_points, labels)
        #     except Exception as e:
        #         print(f'Error inserting data: {e}')

        #     if id is not None:
        #         print(f'Inserted annotation with labels with id: {id}')
        #     else:
        #         print('Error inserting annotation with labels')


    ##### start methods from cell pipeline
    def create_detections(self):
        """
        Used for automated cell labeling - final output for cells detected
        """
        print(self.TASK_DETECT)
        self.process_sections = []  # all sections
        if not self.process_sections:
            print('Processing all sections')
        else:
            print(f'Processing sections: {self.process_sections}')
        self.report_status()
        if self.use_scratch:
            scratch_tmp = get_scratch_dir()
        else:
            scratch_tmp = str(Path('/', 'data'))
        
        _, _, meta_data_info = self.check_prerequisites(scratch_tmp)

        assert (meta_data_info), (
            f"MISSING META-DATA INFO"
        )
        
        self.start_labels(scratch_tmp)
        print(f'Finished cell segmentation - extracting predictions')
        self.create_annotations(meta_data_info)
        self.ng_prep()


    def segment(self):
        """
        Used for automated cell labeling, but just first 2 steps: segmentation
        """
        print(self.TASK_SEGMENT)
        if not self.process_sections:
            print('Segmenting all sections')
        else:
            print(f'Segmenting sections: {self.process_sections}')

        self.report_status()
        if self.use_scratch:
            scratch_tmp = get_scratch_dir()
        else:
            scratch_tmp = str(Path('/', 'data'))
        _, _, meta_data_info = self.check_prerequisites(scratch_tmp)
        
        assert (meta_data_info), (
            f"MISSING META-DATA INFO"
        )
        self.start_labels(scratch_tmp)
        if self.debug:
            print()
            print('*'*50)
        print(f'Finished cell segmentation - extracting predictions')
        if self.channel:
            self.create_annotations(self.task, meta_data_info, self.channel)
        else:
            self.create_annotations(self.task, meta_data_info)
        self.ng_prep()
        # out_dir = Path(scratch_tmp, 'pipeline_tmp', self.animal)
        # if os.path.exists(out_dir):
        #     print(f'Removing existing directory {out_dir}')
        #     delete_in_background(out_dir)

    def ng_preview(self):
        """
        Used to quickly generate json states for Neuroglancer
        Includes:
        -annotation set (if uuid included)
        -database entries compatible with brainsharer
        """
        print(self.TASK_NG_PREVIEW)
        self.report_status()
        self.gen_ng_preview()


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

        print(self.TASK_TRAIN)
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
        
        print("PROCEEDING WITH THE FOLLOWING PARAMETERS:")
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

        non_feature_columns = {'animal', 'section', 'index', 'row', 'col', 'mean_score', 'std_score', 'predictions'}
        detection_features = detection_features.drop(
            [col for col in non_feature_columns if col in detection_features.columns]
        )

        if self.debug:
            print(f'Starting training on {self.animal}, {self.model=}, step={self.step} with {len(detection_features)} features')

        trainer = CellDetectorTrainer(self.animal, step=self.step) # Use Detector 4 as the basis (default)
        np_model, model_filename = trainer.load_models(self.model, self.step)

        if self.debug:
            print(f'USING MODEL LOCATION [WILL CREATE IF NOT EXIST]: {model_filename}')

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


    def create_annotations(self, meta_data_info):
        '''
        Used for generating annotation sets in neuroglancer-compatible format
        currently supports segmentation layer (large qty of points) or annotation layer (smaller qty of points with labels)
        '''
        print(self.TASK_EXTRACT)
        if self.use_scratch:
            scratch_tmp = get_scratch_dir()
        else:
            scratch_tmp = str(Path('/', 'data'))

        #TODO: define
        meta_data_info = None
        segment_ch = None

        df_points, counterstain_channel, label_of_interest_channel, *expected_shape = self.extract_point_annotations(self.task, meta_data_info, segment_ch)

        dest_format = 'segmentation' #vector-based
        # dest_format = 'annotation' #text-based

        if dest_format == 'segmentation':
            print('Creating segmentation layer format')
            self.create_segmentation_layer(df_points, expected_shape, scratch_tmp, counterstain_channel, label_of_interest_channel)
        else:
            print('Creating annotation layer format')
            self.create_annotation_layer(df_points, expected_shape, scratch_tmp, counterstain_channel, label_of_interest_channel)

        # self.create_annotations(self.task, meta_data_info, segment_ch, dest_format, scratch_tmp)


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


