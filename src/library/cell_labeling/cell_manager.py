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
import warnings

from library.cell_labeling.cell_detector_trainer import CellDetectorTrainer
from library.cell_labeling.cell_utilities import calculate_correlation_and_energy, find_connected_segments, load_image, subtract_blurred_image
from library.controller.sql_controller import SqlController
from library.database_model.annotation_points import AnnotationSession
from library.image_manipulation.file_logger import FileLogger
from library.image_manipulation.filelocation_manager import ALIGNED, ALIGNED_DIR, FileLocationManager
from library.image_manipulation.parallel_manager import ParallelManager
from library.utilities.utilities_process import M_UM_SCALE, SCALING_FACTOR, get_scratch_dir, random_string, read_image


class CellMaker(ParallelManager):

    def __init__(self, animal, task, step=4, channel=1, debug=False):
        """Set up the class with the name of the file and the path to it's location."""
        self.animal = animal
        self.task = task
        self.step = step
        self.channel = channel
        self.section_count = 0
        self.fileLocationManager = FileLocationManager(animal)
        self.sqlController = SqlController(animal)
        self.debug = debug
        self.fileLogger = FileLogger(self.fileLocationManager.get_logdir(), self.debug)
        self.cell_label_path = os.path.join(self.fileLocationManager.prep, 'cell_labels')
        #####TODO put average cell someplace better
        self.avg_cell_img_file = Path(os.getcwd(), 'src', 'library', 'cell_labeling', 'average_cell_image.pkl')

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

            if 'dye' in modes and 'virus' in modes:
                msg = "Neuroanatomical_tracing contains both dye and virus channels."
            else:
                msg = "Neuroanatomical_tracing is missing either dye or virus channel."
                if self.debug:
                    print(msg)
                self.fileLogger.logevent(msg)
                raise ValueError(msg)

        # check for full-resolution images (tiff or ome-zarr)
        # checks tiff directory first, then ome-zarr [but must have at least 1 to proceed]
        for key, value in self.meta_channel_mapping.items():
            if value['mode'] == 'dye':
                dye_channel = value.get('channel_name')
            elif value['mode'] == 'virus':
                virus_channel = value.get('channel_name')

        found_dye_channel = False
        found_virus_channel = False
        INPUT_dye = Path(self.fileLocationManager.get_full_aligned(channel=dye_channel[1]))
        INPUT_virus = Path(self.fileLocationManager.get_full_aligned(channel=virus_channel[1]))
        if INPUT_dye.exists():
            if self.debug:
                print(f'Full-resolution tiff stack found (dye channel): {INPUT_dye}')
            self.fileLogger.logevent(f'Full-resolution tiff stack found (dye channel): {INPUT_dye}')
            found_dye_channel = True
        else:
            print(f'Full-resolution tiff stack not found (dye channel). Expected location: {INPUT_dye}; will search for ome-zarr')
        if INPUT_virus.exists():
            if self.debug:
                print(f'Full-resolution tiff stack found (virus channel): {INPUT_virus}')
            self.fileLogger.logevent(f'Full-resolution tiff stack found (virus channel): {INPUT_virus}')
            found_virus_channel = True
        else:
            print(f'Full-resolution tiff stack not found (virus channel). Expected location: {INPUT_virus}; will search for ome-zarr')

        if found_dye_channel == False:
            INPUT_dye = Path(self.fileLocationManager.get_neuroglancer(False, channel=dye_channel[1]) + '.zarr')
            INPUT_virus = Path(self.fileLocationManager.get_neuroglancer(False, channel=virus_channel[1]) + '.zarr')
            if INPUT_dye.exists():
                if self.debug:
                    print(f'Full-resolution ome-zarr stack found (dye channel): {INPUT_dye}')
                self.fileLogger.logevent(f'Full-resolution ome-zarr stack found (dye channel): {INPUT_dye}')
            else:
                print(f'Full-resolution ome-zarr stack not found (dye channel). Expected location: {INPUT_dye}; Exiting')
                sys.exit(1)
        if found_virus_channel == False:
            if INPUT_virus.exists():
                if self.debug:
                    print(f'Full-resolution ome-zarr stack found (virus channel): {INPUT_virus}')
                self.fileLogger.logevent(f'full-resolution ome-zarr stack found (virus channel): {INPUT_virus}')
            else:
                print(f'Full-resolution ome-zarr stack not found (virus channel). expected location: {INPUT_virus}; exiting')
                sys.exit(1)

        # Check for cell training definitions file (average-cell_image.pkl)
        if self.avg_cell_img_file.is_file():
            if self.debug:
                print(f'Found cell training definitions file @ {self.avg_cell_img_file}')
            self.fileLogger.logevent(f'Found cell training definitions file @ {self.avg_cell_img_file}')

        # Check for model file (models_round_{self.step}_threshold_2000.pkl) in the models dir
        self.model_file = os.path.join('/net/birdstore/Active_Atlas_Data/cell_segmentation/models', f'models_round_{self.step}_threshold_2000.pkl')
        if os.path.exists(self.model_file):
            if self.debug:
                print(f'Found model file @ {self.model_file}')

            self.fileLogger.logevent(f'Found model file @ {self.model_file}')
        else:
            print(f'Model file not found @ {self.model_file}')
            self.fileLogger.logevent(f'Model file not found @ {self.model_file}; Exiting')
            sys.exit(1)

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
        for channel_number, channel_data in self.meta_channel_mapping.items():
            if channel_data['mode'] == 'dye':
                self.dye_channel = channel_number
                self.fileLogger.logevent(f'Dye channel detected: {self.dye_channel}')
            elif channel_data['mode'] == 'virus':
                self.virus_channel = channel_number
                self.fileLogger.logevent(f'Virus channel detected: {self.virus_channel}')
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

        self.max_segment_size = 100000
        self.segmentation_threshold = 2000 
        self.cell_radius = 40

        if self.input_format == 'tif':
            input_path_dye = input_path_dye = self.fileLocationManager.get_full_aligned(channel=self.dye_channel)
            input_path_virus = self.fileLocationManager.get_full_aligned(channel=self.virus_channel)
            self.section_count = self.capture_total_sections(self.input_format, input_path_dye) #Only need single/first channel to get total section count
        else:
            input_path_dye = Path(self.fileLocationManager.get_neuroglancer(False, channel=self.dye_channel) + '.zarr')
            input_path_virus = Path(self.fileLocationManager.get_neuroglancer(False, channel=self.virus_channel) + '.zarr')

            # OME-ZARR Section count may be extracted from meta-data in folder or from meta-data in file [do not use database]

        file_keys = []
        for section in range(self.section_count):
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
            debug,
            *_,
        ) = file_keys

        if not os.path.exists(input_path_virus):
            print(f'ERROR: {input_path_virus} not found')
            sys.exit(1)
        if not os.path.exists(input_path_dye):
            print(f'ERROR: {input_path_dye} not found')
            sys.exit(1)

        if debug:
            print(f'Starting identify_cell_candidates on section: {str_section_number}')

        def filter_cell_candidates(
            animal,
            section_number,
            connected_segments,
            max_segment_size,
            cell_radius,
            x_window,
            y_window,
            absolute_coordinates,
            difference_ch1,
            difference_ch3,
        ):
            """PART OF STEP 2. Identify cell candidates:  Area is for the object, where pixel values are not zero, 
            Segments are filtered to remove those that are too large or too small"""
            n_segments, segment_masks, segment_stats, segment_location = (connected_segments)
            cell_candidates = []
            for segmenti in range(n_segments):
                _, _, width, height, object_area = segment_stats[segmenti, :]
                if object_area > max_segment_size:
                    continue
                segment_row, segment_col = segment_location[segmenti, :]

                row_start = int(segment_row - cell_radius)
                col_start = int(segment_col - cell_radius)
                if row_start < 0 or col_start < 0:
                    continue
                row_end = int(segment_row + cell_radius)
                col_end = int(segment_col + cell_radius)
                if (
                    row_end > x_window or col_end > y_window
                ):  # row evaluates with x-axis (width), col evaluates with y-axis (height)
                    continue
                segment_mask = (segment_masks[row_start:row_end, col_start:col_end] == segmenti)
                cell = {
                    "animal": animal,
                    "section": section_number,
                    "area": object_area,
                    "absolute_coordinates_YX": (
                        absolute_coordinates[2] + segment_col,
                        absolute_coordinates[0] + segment_row,
                    ),
                    "cell_shape_XY": (height, width),
                    "image_CH3": difference_ch3[row_start:row_end, col_start:col_end].T,
                    "image_CH1": difference_ch1[row_start:row_end, col_start:col_end].T,
                    "mask": segment_mask.T,
                }                                        
                cell_candidates.append(cell)
            return cell_candidates

        output_path = Path(SCRATCH, 'pipeline_tmp', animal, 'cell_candidates')
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = Path(output_path, f'extracted_cells_{str_section_number}.gz')

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

        cell_candidates=[]
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
            ch1_contrast, ch3_constrast, moments_data = self.features_using_center_connected_components(cell)

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
            srow = int(spreadsheet_row['row'])
            scol = int(spreadsheet_row['col'])
            if debug:
                if srow == 5718 and scol == 26346:
                    for k,v in spreadsheet_row.items():
                        print(f'{k}: {v}')
                    print('Found cell')
                    print()

        df_features = pd.DataFrame(output_spreadsheet)
        df_features.to_csv(output_file, index=False)

        print(f'Saving {len(output_spreadsheet)} cell features to {output_file}')
        print('Completed calculate_features')

        return df_features

    def features_using_center_connected_components(self, cell_candidate_data):   
        '''Part of step 3. calculate cell features'''
        def mask_mean(mask,image):
            mean_in=np.mean(image[mask==1])
            mean_all=np.mean(image.flatten())
            return (mean_in-mean_all)/(mean_in+mean_all)    # calculate the contrast: mean

        def append_string_to_every_key(dictionary, post_fix): 
            return dict(zip([keyi + post_fix for keyi in dictionary.keys()],dictionary.values()))

        def calc_moments_of_mask(mask):   
            '''
            calculate moments (how many) and Hu Moments (7)
            Moments(
                    double m00,
                    double m10,
                    double m01,
                    double m20,
                    double m11,
                    double m02,
                    double m30,
                    double m21,
                    double m12,
                    double m03
                    );
            Hu Moments are described in this paper: 
            https://www.researchgate.net/publication/224146066_Analysis_of_Hu's_moment_invariants_on_image_scaling_and_rotation

            NOTE: image moments (weighted average of pixel intensities) are used to calculate centroid of arbritary shapes in opencv library
            '''
            mask = mask.astype(np.float32)
            moments = cv2.moments(mask)

            huMoments = cv2.HuMoments(moments)
            moments = append_string_to_every_key(moments, f'_mask')
            return (moments, {'h%d'%i+f'_mask':huMoments[i,0]  for i in range(7)}) #return first 7 Hu moments e.g. h1_mask

        mask = cell_candidate_data['mask']  

        moments_data = calc_moments_of_mask(mask)
        #ids, counts = np.unique(mask, return_counts=True)
        #print(f'mask shape: {mask.shape}')
        #print(f'{ids=}, {counts=}')

        # Calculate constrasts relative to mask
        ch1_contrast = mask_mean(mask, cell_candidate_data['image_CH1'])
        ch3_constrast = mask_mean(mask, cell_candidate_data['image_CH3'])

        return ch1_contrast, ch3_constrast, moments_data

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
            debug,
        ) = file_keys
        model_file = load(model_filename)
        if debug:
            print(f'Starting function score_and_detect_cell on section {section}')

        def calculate_scores(features: pd.DataFrame, model):
            all = xgb.DMatrix(features) #RENAME VARIABLE
            scores=np.zeros([features.shape[0], len(model)])
            for i in range(len(model)):
                bst = model[i]
                attributes = bst.attributes()
                try:
                    best_ntree_limit = int(attributes["best_ntree_limit"])
                except KeyError:
                    best_ntree_limit = 676
                scores[:,i] = bst.predict(all, iteration_range=[1, best_ntree_limit], output_margin=True)

            _mean = np.mean(scores, axis=1)
            _std = np.std(scores, axis=1)
            return _mean, _std

        def get_prediction_and_label(_mean) -> list:
            threshold = 1.5
            predictions = []
            for __mean in _mean:
                __mean = float(__mean)
                classification = -2 #Default: cell candidate is not actually a cell
                if __mean > threshold: #Candidate is a cell
                    classification = 2
                elif __mean > -threshold and __mean <= threshold:
                    classification = 0 #UNKNOWN/UNSURE
                predictions.append(classification)
            return predictions

        drops = ['animal', 'section', 'index', 'row', 'col']        
        cell_features_selected_columns = cell_features.drop(drops,axis=1)
        # Step 4-2-1-2) calculate_scores(features) - calculates scores, labels, mean std for each feature
        _mean, _std = calculate_scores(cell_features_selected_columns, model_file)
        # Step 4-2-1-3) predictive cell labeling based on mean
        cell_features['mean_score'] = _mean
        cell_features['std_score'] = _std
        cell_features['predictions'] = np.array(get_prediction_and_label(_mean)) #PUTATIVE ID: POSITIVE (2), NEGATIVE (-2), UNKNOWN/UNSURE (0)

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
        METHODS TO TRAIN CELL DETECTOR MODEL

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
        detection_files = sorted(glob.glob( os.path.join(self.cell_label_path, f'detections_00*.csv') ))
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
        detection_features=detection_features.drop(drops,axis=1)

        print(f'Starting training on {self.animal} step={self.step} with {len(detection_features)} features')

        trainer = CellDetectorTrainer(self.animal, step=self.step) # Use Detector 4 as the basis
        new_models = trainer.train_classifier(detection_features, 676, 3, models = trainer.load_models()) # pass Detector 4 for training
        trainer = CellDetectorTrainer(self.animal, step=self.step + 1) # Be careful when saving the model. The model path is only relevant to 'step'. 
        # You need to use a new step to save the model, otherwise the previous models would be overwritten.
        trainer.save_models(new_models)

    def create_features(self):
        # col=11347, row=17614
        #col=5718, row=26346

        #TODO: Why hard-coded variables?
        col = 5718
        row = 26346
        section = 0
        print(f'processing coordinates {col=}, {row=}, {section=}')
        idx = 0
        avg_cell_img = load(self.avg_cell_img_file) #Load average cell image once
        height = 80
        width = 80
        x_start = col - (width//2)
        x_end = x_start + 80
        y_start = row - 40
        y_end = y_start + 80

        input_file_virus_path = os.path.join(self.fileLocationManager.get_directory(channel=3, downsample=False, inpath=ALIGNED_DIR), f'{str(section).zfill(3)}.tif')  
        input_file_dye_path = os.path.join(self.fileLocationManager.get_directory(channel=1, downsample=False, inpath=ALIGNED_DIR), f'{str(section).zfill(3)}.tif')  
        data_virus = load_image(input_file_virus_path)
        data_dye = load_image(input_file_dye_path)

        image_roi_virus = data_virus[x_start:x_end, y_start:y_end] #image_roi IS numpy array
        image_roi_dye = data_dye[x_start:x_end, y_start:y_end] #image_roi IS numpy array
        print(f'shape of image_roi_virus {image_roi_virus.shape} and shape of data_virus {image_roi_dye.shape}')

        connected_segments = find_connected_segments(image_roi_virus, 2000)
        n_segments, segment_masks, segment_stats, segment_location = (connected_segments)
        print(f'Found {n_segments} segments')
        print(f'{segment_stats=}')
        print(f'{segment_location=}')
        print(f'segment masks shape {segment_masks.shape}')

        segmenti = 1
        segment_row, segment_col = segment_location[segmenti, :]
        _, _, width, height, object_area = segment_stats[segmenti, :]
        #segment_row, segment_col = segment_location[segmenti, :]
        print(f'{segment_row=}, {segment_col=}, {width=}, {height=}, {object_area=}')
        cell = {
            "animal": self.animal,
            "section": section,
            "area": object_area,
            "absolute_coordinates_YX": (col,row),
            "cell_shape_XY": (height, width),
            "image_CH3": image_roi_virus,
            "image_CH1": image_roi_dye,
            "mask": segment_masks,
        }

        #TODO: see calculate_features() ~line 489 [consolidate in cell_utilities.py or similar]
        #to avoid duplicate code
        ch1_corr, ch1_energy = calculate_correlation_and_energy(avg_cell_img["CH1"], image_roi_dye)
        ch3_corr, ch3_energy = calculate_correlation_and_energy(avg_cell_img['CH3'], image_roi_virus)

        # STEP 3-D) features_using_center_connected_components
        ch1_contrast, ch3_constrast, moments_data = self.features_using_center_connected_components(cell)

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
        spreadsheet_row.update({'contrast1': ch1_contrast, 'contrast3': ch3_constrast})

        for k,v in spreadsheet_row.items():
            print(f'{k}: {v}')  


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
            print(f'type cell features {type(cell_features)}')
            print(cell_features.head())
            if file_keys[-1]: #DEBUG
                print(f'DEBUG: start_labels - STEP 4 (Detect cells [based on features])')
                print(f'Cell features: {len(cell_features)}')
            cellmaker.score_and_detect_cell(file_keys, cell_features)

