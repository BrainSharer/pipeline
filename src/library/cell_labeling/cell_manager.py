import gzip
import shutil
import struct
import os, sys, glob, json, math
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
import xgboost as xgb

from library.database_model.annotation_points import AnnotationSession
from library.utilities.utilities_process import M_UM_SCALE, random_string


def detect_cells_all_sections(file_keys: tuple):
    #LAUNCHER FOR MULTIPROCESSING OF ALL (4) STEPS - ALL FUNCTIONS MUST BE SERIALIZABLE ('PICKLEABLE')
    #CLASS INSTANCES ARE NOT SERIALIZABLE; USE STATIC METHODS INSTEAD
    #NOTES: model_filename HAS PLACEHOLDER, BUT CANNOT BE LOADED INTO file_keys (NOT SERIALIZABLE)
    #FILELOGGER NOT CURRENTLY ACTIVE IN MULTI-PROCESSING MODE
    
    #think about how to inherit, or reinstantiate Pipeline/Filelogger
    # filemanager = FileLocationManager(file_key[0])
    # filelogger = FileLogger(filemanager.get_logdir())
    # cellmaker = filelogger(CellMaker)

    # currently debug (bool) is set at end of file_keys

    cellmaker = CellMaker()
    if file_keys[-1]: #LAST ELEMENT OF TUPLE IS DEBUG
        print(f"DEBUG: auto_cell_labels - STEP 1 & 2 (IDENTIFY CELL CANDIDATES)")

    cell_candidates = cellmaker.identify_cell_candidates(file_keys) #STEPS 1 & 2. VIRTUAL TILING AND CELL CANDIDATE IDENTIFICATION
    
    if file_keys[-1]: #DEBUG
        print(f"DEBUG: FOUND {len(cell_candidates)} CELL CANDIDATES ON SECTION {file_keys[1]}")

    if len(cell_candidates) > 0: #CONTINUE IF CELL CANDIDATES WERE DETECTED [NOTE: THIS IS LIST OF ALL CELL CANDIDATES]
        if file_keys[-1]: #DEBUG
            print(f"DEBUG: CREATE CELL FEATURES WITH IDENTIFIED CELL CANDIDATES (auto_cell_labels - STEP 3)")
        cell_features = cellmaker.calculate_features(file_keys, cell_candidates) #STEP 3. CALCULATE CELL FEATURES
        print(f'type cell features {type(cell_features)}')
        print(cell_features.head())
        if file_keys[-1]: #DEBUG
            print(f'DEBUG: start_labels - STEP 4 (DETECT CELLS [BASED ON FEATURES])')
            print(f'CELL FEATURES: {len(cell_features)}')
        cellmaker.score_and_detect_cell(file_keys, cell_features)

class CellMaker():
    """ML autodetection of cells in image"""

    def __init__(self):
        """Set up the class with the name of the file and the path to it's location."""
        self.channel = 1
        self.section_count = 0


    def check_prerequisites(self, SCRATCH):
        '''
        CELL LABELING REQUIRES 
        A) AVAILABLE FULL-RESOLUTION IMAGES, 
        B) 2 CHANNELS (NAME, TYPE), 
        C) SCRATCH DIRECTORY, 
        D) OUTPUT DIRECTORY, 
        E) cell_definitions (manual training of what cell looks like: average_cell_image.pkl), 
        F) models: models_example_json.pkl
        '''
        # CHECK FOR OME-ZARR (NOT IMPLEMENTED AS OF 22-OCT-2023)
        # INPUT = self.fileLocationManager.get_ome_zarr(channel=self.channel)
        # print(f'OME-ZARR FOUND: {INPUT}') #SEND TO LOG FILE

        # CHECK FOR FULL-RESOLUTION TIFF IMAGES (IF OME-ZARR NOT PRESENT)
        INPUT = self.fileLocationManager.get_full_aligned(channel=self.channel)
        if os.path.exists(INPUT):
            if self.debug:
                print(f'FULL-RESOLUTION TIFF STACK FOUND: {INPUT}')
        else:
            print(f'FULL-RESOLUTION TIFF STACK NOT FOUND: {INPUT}; EXITING')
            sys.exit(1)
        self.fileLogger.logevent(f'FULL-RESOLUTION TIFF STACK FOUND: {INPUT}')

        self.OUTPUT = self.fileLocationManager.get_cell_labels()
        if self.debug:
            print(f'CELL LABELS OUTPUT DIR: {self.OUTPUT}')
        self.fileLogger.logevent(f'CELL LABELS OUTPUT DIR: {self.OUTPUT}')

        self.SCRATCH = SCRATCH #TODO SEE IF WE CAN AUTO-DETECT NVME
        if self.debug:
            print(f'TEMP STORAGE LOCATION: {SCRATCH}')
        self.fileLogger.logevent(f'TEMP STORAGE LOCATION: {SCRATCH}')

        # CHECK FOR PRESENCE OF meta-data.json
        meta_data_file = 'meta-data.json'
        meta_store = os.path.join(self.fileLocationManager.prep, meta_data_file)

        try:
            meta_data_info = {}
            if os.path.isfile(meta_store):
                print(f'FOUND NEUROANATOMICAL TRACING INFO; READING FROM {meta_store}')

                # verify you have 2 required channels
                with open(meta_store) as fp:
                    info = json.load(fp)
                self.meta_channel_mapping = info['Neuroanatomical_tracing']
                meta_data_info['Neuroanatomical_tracing'] = self.meta_channel_mapping

                # TODO: MOVE ASSERTIONS TO SEPARATE FUNCTION (UNIT TEST) - maybe on send to log w/ error - missing file X
                # self.dyes = [item['description'] for item in info['Neuroanatomical_tracing']]
                # assert 'GFP' in self.dyes and 'NeurotraceBlue' in self.dyes
                # print('TWO CHANNELS READY')
                # #self.fileLogger.logevent(f"USING 2 CHANNELS FOR AUTOMATIC CELL DETECTION: {self.dyes}")

            else:
                # CREATE META-DATA STORE (PULL FROM DATABASE)
                if self.debug:
                    print(f'NOT FOUND; CREATING META-DATA STORE @ {meta_store}')

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

        # CHECK FOR CELL TRAINING DEFINITIONS FILE (average-cell_image.pkl)
        self.avg_cell_img_file = Path(os.getcwd(), 'src', 'library', 'cell_labeling', 'average_cell_image.pkl')
        if self.avg_cell_img_file.is_file():
            if self.debug:
                print(f'FOUND CELL TRAINING DEFINITIONS FILE @ {self.avg_cell_img_file}')
            self.fileLogger.logevent(f'FOUND CELL TRAINING DEFINITIONS FILE @ {self.avg_cell_img_file}')

        # CHECK FOR MODEL FILE (models_example_json.pkl) in the models dir
        self.model_file = os.path.join('/net/birdstore/Active_Atlas_Data/cell_segmentation/models', 'models_example_json.pkl')
        if os.path.exists(self.model_file):
            if self.debug:
                print(f'FOUND MODEL FILE @ {self.model_file}')

            self.fileLogger.logevent(f'FOUND MODEL FILE @ {self.model_file}')
        else:
            print(f'MODEL FILE NOT FOUND @ {self.model_file}')
            self.fileLogger.logevent(f'MODEL FILE NOT FOUND @ {self.model_file}; EXITING')
            sys.exit(1)

    def start_labels(self):
        '''1. USE DASK TO CREATE VIRTUAL TILES OF FULL-RESOLUTION IMAGES
                Which mode (dye/virus: Neurotrace/GFP is for which directory) -> rename input
                fluorescence_image = GFP a.k.a. virus channel (channel 3)
                nissel_stain_image = Neurotraceblue a.k.a. dye channel (channel 1) aka cell_body

           2. IDENTIFY CELL CANDIDATES - image segmentation
                -this step will create pickle files totaling size (aggregate) of image stack (approx)
                @ end of step: in SCRATCH (1 compressed pickle file for each section - if cell candidates were detected)

           3. CREATE CELL FEATURES
                @ start: check pickle files (count)
                @ end of step: in SCRATCH (1 csv file for each section - if cell candidates were detected)

           4. DETECT CELLS; SCORE CELL CANDIDATE AND CLASSIFY AS POSITIVE, NEGATIVE, UNKNOWN
                @ start: csv file for each section with cell features [used in identification]
                @ end of step: csv file for each section where putative CoM cells detected with classification (positive, negative, unknown)
        '''
        self.fileLogger.logevent(f"DEBUG: start_labels - STEPS 1 & 2 (REVISED); START ON IMAGE SEGMENTATION")
        if self.debug:
            print(f"DEBUG: start_labels - STEPS 1 & 2 (REVISED); START ON IMAGE SEGMENTATION")

        # TODO: Need to address scenario where >1 dye or virus channels are present [currently only 1 of each is supported]
        for channel_number, channel_data in self.meta_channel_mapping.items():
            if channel_data['mode'] == 'dye':
                self.dye_channel = channel_number
                self.fileLogger.logevent(f'DYE CHANNEL DETECTED: {self.dye_channel}')
            elif channel_data['mode'] == 'virus':
                self.virus_channel = channel_number
                self.fileLogger.logevent(f'VIRUS CHANNEL DETECTED: {self.virus_channel}')
            else:
                continue
                msg = "Neuroanatomical_tracing is missing either dye or virus channel."
                if self.debug:
                    print(msg)
                self.fileLogger.logevent(msg)
                raise ValueError(msg)

        self.input_format = 'tif' #options are 'tif' and 'ome-zarr'
        if os.path.exists(self.avg_cell_img_file):
            avg_cell_img = load(self.avg_cell_img_file) #LOAD AVERAGE CELL IMAGE ONCE
        else:
            print(f'Could not find {self.avg_cell_img_file}')
            sys.exit()

        self.max_segment_size = 100000
        self.segmentation_threshold = 2000 
        self.cell_radius = 40

        if self.input_format == 'tif':
            INPUT = input_path_dye = self.fileLocationManager.get_full_aligned(channel=self.dye_channel)
            input_path_virus = self.fileLocationManager.get_full_aligned(channel=self.virus_channel)
            self.section_count = self.capture_total_sections(self.input_format, INPUT) #ONLY NEED SINGLE/FIRST CHANNEL TO GET TOTAL SECTION COUNT
        else:
            INPUT = input_path_dye = self.fileLocationManager.get_ome_zarr(channel=self.dye_channel)
            input_path_virus = self.fileLocationManager.get_ome_zarr(channel=self.virus_channel)
            # OME-ZARR SECTION COUNT MAY BE EXTRACTED FROM META-DATA IN FOLDER [DO NOT USE DATABASE]

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
            print(f'RUNNING IN DEBUG MODE WITH {workers} WORKERS; {len(file_keys)} SECTIONS TO PROCESS, out: {self.SCRATCH}')
        else:
            workers = math.floor(min([self.get_nworkers(), 10])*.5) # MAX 50% OF PREV. CALCS [DASK IS RAM INTENSIVE]
            print(f'RUNNING IN PARALLEL WITH {workers} WORKERS; {len(file_keys)} SECTIONS TO PROCESS, out: {self.SCRATCH}')
        self.run_commands_concurrently(detect_cells_all_sections, file_keys, workers)

    def identify_cell_candidates(self, file_keys: tuple) -> int:
        '''2. IDENTIFY CELL CANDIDATES - PREV: find_examples()
                -REQUIRES IMAGE TILING OR DASK VIRTUAL TILES PRIOR TO RUNNING

                THIS SINGLE METHOD WILL BE RUN IN PARALLEL FOR EACH SECTION-CONSISTS OF 3 SUB-STEPS:
                A) subtract_blurred_image (average the image by subtracting gaussian blurred mean)
                B) identification of cell candidates based on connected segments
                C) filering cell candidates based on size and shape

           NOTE: '*_' (UNPACKING file_keys TUPLE WILL DISCARD VARS AFTER debug IF SET); MUST MODIFY IF file_keys IS CHANGED
        '''
        animal, section, str_section_number, segmentation_threshold, cell_radius, max_segment_size, SCRATCH, OUTPUT, avg_cell_img, model_filename, input_format, input_path_dye, input_path_virus, debug, *_ = file_keys

        if not os.path.exists(input_path_virus):
            print(f'ERROR: {input_path_virus} NOT FOUND')
            sys.exit(1)
        if not os.path.exists(input_path_dye):
            print(f'ERROR: {input_path_dye} NOT FOUND')
            sys.exit(1)

        if debug:
            print(f'STARTING identify_cell_candidates ON SECTION: {str_section_number}')
        def load_image(file: str):
            if os.path.exists(file):
                return imageio.imread(file)
            else:
                print(f'ERROR: {file} NOT FOUND')
                sys.exit(1)
        def subtract_blurred_image(image):
            '''PART OF STEP 2. IDENTIFY CELL CANDIDATES: average the image by subtracting gaussian blurred mean'''
            image = np.float32(image)
            small = cv2.resize(image, (0, 0), fx=0.05, fy=0.05, interpolation=cv2.INTER_AREA)
            blurred = cv2.GaussianBlur(small, ksize=(21, 21), sigmaX=10) # Blur the resized image
            relarge = cv2.resize(blurred, image.T.shape, interpolation=cv2.INTER_AREA) # Resize the blurred image back to the original size
            difference = image - relarge # Calculate the difference between the original and resized-blurred images
            return difference
        def find_connected_segments(image, segmentation_threshold) -> tuple:
            '''PART OF STEP 2. IDENTIFY CELL CANDIDATES'''
            n_segments, segment_masks, segment_stats, segment_location = cv2.connectedComponentsWithStats(np.int8(image > segmentation_threshold))
            segment_location = np.int32(segment_location)
            segment_location = np.flip(segment_location, 1) 
            return (n_segments, segment_masks, segment_stats, segment_location)
        def filter_cell_candidates(animal, section_number, connected_segments, max_segment_size, cell_radius, x_window, y_window, absolute_coordinates, difference_ch1, difference_ch3):
            '''PART OF STEP 2. IDENTIFY CELL CANDIDATES:  Area is for the object, where pixel values are not zero, Segments are filtered to remove those that are too large or too small'''
            n_segments, segment_masks, segment_stats, segment_location = connected_segments
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
                if row_end > x_window or col_end > y_window: #ROW EVALUATES WITH X-AXIS (WIDTH), COL EVALUATES WITH Y-AXIS (HEIGHT)
                    continue
                segment_mask = segment_masks[row_start:row_end, col_start:col_end] == segmenti
                candidate = {'animal': animal,
                        'section': section_number,
                        'area': object_area,
                        'absolute_coordinates_YX': (absolute_coordinates[2]+segment_col, absolute_coordinates[0]+segment_row),
                        'cell_shape_XY': (height, width),
                        'image_CH3': difference_ch3[row_start:row_end, col_start:col_end].T,
                        'image_CH1': difference_ch1[row_start:row_end, col_start:col_end].T,
                        'mask': segment_mask.T}
                cell_candidates.append(candidate)
            return cell_candidates

        output_path = Path(SCRATCH, 'pipeline_tmp', animal, 'cell_candidates')
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = Path(output_path, f'extracted_cells_{str_section_number}.gz')

        # TODO: CLEAN UP - maybe extend dask to more dimensions?
        if input_format == 'tif':#section_number IS ALREADY STRING FOR LEGACY PROCESSING 'tif' (zfill)
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

            # TODO: DIFFERENT PROCESSING FOR OME-ZARR
            # see del_img_extract2.py (this folder) for more info

        # Create delayed tasks for loading the images (SEPARATE TASK LIST PER CHANNEL)
        delayed_tasks_virus = [delayed(load_image)(path) for path in [input_file_virus]]
        delayed_tasks_dye = [delayed(load_image)(path) for path in [input_file_dye]]

        # GET SHAPE WITHOUT COMPUTING
        org_img_shape = dask.compute(delayed_tasks_virus[0].shape) 

        # SHAPE WILL BE SAME FOR BOTH CHANNELS (stores as y-axis then x-axis)
        x_dim = org_img_shape[0][1]
        y_dim = org_img_shape[0][0]

        # Create a Dask array from the delayed tasks (NOTE: DELAYED)
        image_stack_virus = [da.from_delayed(v, shape=(x_dim, y_dim), dtype='uint16') for v in delayed_tasks_virus]
        image_stack_dye = [da.from_delayed(v, shape=(x_dim, y_dim), dtype='uint16') for v in delayed_tasks_dye]
        data_virus = dask.compute(image_stack_virus[0])[0] #FULL IMAGE
        data_dye = dask.compute(image_stack_dye[0])[0] #FULL IMAGE

        # SWAP X AND Y AXES (READ IN Y-AXIS, THEN X-AXIS BUT WE WANT X,Y)
        data_virus = np.swapaxes(data_virus, 1, 0)
        data_dye = np.swapaxes(data_dye, 1, 0)

        # FINAL VERSION BELOW:
        total_virtual_tile_rows = 5
        total_virtual_tile_columns = 2
        x_window = int(math.ceil(x_dim / total_virtual_tile_rows))
        y_window = int(math.ceil(y_dim / total_virtual_tile_columns))

        if debug:
            print(f'DASK ARRAY CREATED WITH FOLLOWING PARAMETERS: {x_window=}, {y_window=}; {total_virtual_tile_rows=}, {total_virtual_tile_columns=}')

        cell_candidates=[]
        for row in range(total_virtual_tile_rows):
            for col in range(total_virtual_tile_columns):
                x_start = row*x_window
                x_end = x_window*(row+1)
                y_start = col*y_window
                y_end = y_window*(col+1)

                image_roi_virus = data_virus[x_start:x_end, y_start:y_end] #image_roi IS NUMPY ARRAY
                image_roi_dye = data_dye[x_start:x_end, y_start:y_end] #image_roi IS NUMPY ARRAY

                absolute_coordinates = (x_start, x_end, y_start, y_end)
                difference_ch3 = subtract_blurred_image(image_roi_virus) #CALCULATE IMG DIFFERENCE FOR VIRUS CHANNEL (e.g. FLUORESCENCE)

                connected_segments = find_connected_segments(difference_ch3, segmentation_threshold)

                if debug:
                    print(f'CONNECTED SEGMENTS FOR VIRUS CHANNEL (BLURRED IMG DIFFERENCE): {connected_segments[0]}')

                if connected_segments[0] > 2: #FOUND CELL CANDIDATE (first element of tuple is count)
                    if debug:
                        print(f'FOUND CELL CANDIDATE: COM-{absolute_coordinates=}, {cell_radius=}, {str_section_number=}')
                    difference_ch1 = subtract_blurred_image(image_roi_dye) #CALCULATE IMG DIFFERENCE FOR DYE CHANNEL (e.g. NEUROTRACE)
                    cell_candidate = filter_cell_candidates(animal, section, connected_segments, max_segment_size, cell_radius, x_window, y_window, absolute_coordinates, difference_ch1, difference_ch3)
                    # print(f'ADDING CELL CANDIDATE: {cell_candidate}')
                    cell_candidates.extend(cell_candidate) #MUST USE EXTEND!
                else:
                    if debug:
                        print(f'NO CELL CANDIDATE FOUND: COM-{absolute_coordinates=}, {str_section_number=}')

        if len(cell_candidates) > 0:
            print(f'SAVING {len(cell_candidates)} CELL CANDIDATES TO {output_file}')
            # if debug:
            #     print(f'RAW cell_candidates: {cell_candidates=}')
            dump(cell_candidates, output_file, compression="gzip", set_default_extension=True)

        if debug:
            print('COMPLETED identify_cell_candidates')

        return cell_candidates

    def calculate_features(self, file_keys: tuple, cell_candidate_data) -> pd.DataFrame:
        '''PART OF STEP 3. CALCULATE CELL FEATURES;

            THIS SINGLE METHOD WILL BE RUN IN PARALLEL FOR EACH SECTION
            -CONSISTS OF 5 SUB-STEPS:
            A) load information from cell candidates (pickle files) - now passed as parameter
            B1) calculate_correlation_and_energy FOR CHANNEL 1
            B2) calculate_correlation_and_energy FOR CHANNEL 3
            C) features_using_center_connected_components(example)
            D) SAVE FEATURES (CSV FILE)
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
        if debug:
            print(f'STARTING FUNCTION: calculate_features WITH {len(cell_candidate_data)} CELL CANDIDATES')

        output_path = Path(SCRATCH, 'pipeline_tmp', animal, 'cell_features')
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = Path(output_path, f'cell_features_{str_section_number}.csv')

        # STEP 3-B) load information from cell candidates (pickle files from step 2 - cell candidate identification) **NOW PASSED AS PARAMETER**
        output_spreadsheet = []
        for idx, cell in enumerate(cell_candidate_data):
            # STEP 3-C1, 3-C2) calculate_correlation_and_energy FOR CHANNELS 1 & 3 (ORG. FeatureFinder.py; calculate_features())
            ch1_corr, ch1_energy = self.calculate_correlation_and_energy(avg_cell_img["CH1"], cell['image_CH1'])
            ch3_corr, ch3_energy = self.calculate_correlation_and_energy(avg_cell_img['CH3'], cell['image_CH3'])

            # STEP 3-D) features_using_center_connected_components
            ch1_contrast, ch3_constrast, moments_data = self.features_using_center_connected_components(cell)

            # BUILD FEATURES DICTIONARY
            spreadsheet_row = {'animal': animal, 'section': section, 'index': idx, 'row': cell["absolute_coordinates_YX"][0], 'col': cell["absolute_coordinates_YX"][1], 'area': cell['area'], 
                                'height': cell['cell_shape_XY'][1], 'width': cell['cell_shape_XY'][0],  
                                'corr_CH1': ch1_corr, 'energy_CH1': ch1_energy, 'corr_CH3': ch3_corr, 'energy_CH3': ch3_energy}
            spreadsheet_row.update(moments_data[0])
            spreadsheet_row.update(moments_data[1]) #e.g. 'h1_mask' (6 items)
            spreadsheet_row.update({'contrast1': ch1_contrast, 'contrast3': ch3_constrast})
            output_spreadsheet.append(spreadsheet_row)

        if debug:
            print(f'SAVING {len(output_spreadsheet)} CELL FEATURES TO {output_file}')

        df_features = pd.DataFrame(output_spreadsheet)
        df_features.to_csv(output_file, index=False)

        if debug:
            print('COMPLETED calculate_features')

        return df_features

    def calculate_correlation_and_energy(self, avg_cell_img, cell_candidate_img):  
        '''PART OF STEP 3. 
        CALCULATE CELL FEATURES; CALCULATE CORRELATION [BETWEEN cell_candidate_img 
        AND avg_cell_img] and AND ENERGY FOR CELL CANIDIDATE
        NOTE: avg_cell_img AND cell_candidate_img CONTAIN RESPECTIVE CHANNELS PRIOR TO PASSING IN ARGUMENTS
        '''

        # ENSURE IMAGE ARRAYS TO SAME SIZE
        cell_candidate_img, avg_cell_img = self.equalize_array_size_by_trimming(cell_candidate_img, avg_cell_img)

        # COMPUTE NORMALIZED SOBEL EDGE MAGNITUDES USING GRADIENTS OF CANDIDATE IMAGE vs. GRADIENTS OF THE EXAMPLE IMAGE
        avg_cell_img_x, avg_cell_img_y = self.sobel(avg_cell_img)
        cell_candidate_img_x, cell_candidate_img_y = self.sobel(cell_candidate_img)

        # corr = the mean correlation between the dot products at each pixel location
        dot_prod = (avg_cell_img_x * cell_candidate_img_x) + (avg_cell_img_y * cell_candidate_img_y)
        corr = np.mean(dot_prod.flatten())      

        # energy: the mean of the norm of the image gradients at each pixel location
        mag = np.sqrt(cell_candidate_img_x **2 + cell_candidate_img_y **2)
        energy = np.mean((mag * avg_cell_img).flatten())  
        return corr, energy

    def equalize_array_size_by_trimming(self, array1, array2):
        '''PART OF STEP 3. CALCULATE CELL FEATURES; array1 and array 2 the same size'''
        size0 = min(array1.shape[0], array2.shape[0])
        size1 = min(array1.shape[1], array2.shape[1])
        array1 = self.trim_array_to_size(array1, size0, size1)
        array2 = self.trim_array_to_size(array2, size0, size1)
        return array1, array2    

    def trim_array_to_size(self, array, size0, size2):
        '''PART OF STEP 3. CALCULATE CELL FEATURES'''
        if(array.shape[0] > size0):
            size_difference = int((array.shape[0]-size0)/2)
            array = array[size_difference:size_difference+size0, :]
        if(array.shape[1] > size2):
            size_difference = int((array.shape[1]-size2)/2)
            array = array[:, size_difference:size_difference+size2]
        return array

    def sobel(self, img):
        '''PART OF STEP 3. CALCULATE CELL FEATURES; Compute the normalized sobel edge magnitudes'''
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        _mean = (np.mean(sobel_x) + np.mean(sobel_y))/2.
        _std = np.sqrt((np.var(sobel_x) + np.var(sobel_y))/2)
        sobel_x = (sobel_x - _mean) / _std
        sobel_y = (sobel_y - _mean) / _std
        return sobel_x, sobel_y

    def features_using_center_connected_components(self, cell_candidate_data):   
        '''PART OF STEP 3. CALCULATE CELL FEATURES'''
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

        # CALCULATE CONSTRASTS RELATIVE TO MASK
        ch1_contrast = mask_mean(mask, cell_candidate_data['image_CH1'])
        ch3_constrast = mask_mean(mask, cell_candidate_data['image_CH3'])

        return ch1_contrast, ch3_constrast, moments_data

    def score_and_detect_cell(self, file_keys: tuple, cell_features: pd.DataFrame):
        ''' PART OF STEP 4. DETECT CELLS; SCORE CELLS BASED ON FEATURES (PRIOR TRAINED MODELS (30) USED FOR CALCULATION)'''

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
            print(f'STARTING FUNCTION score_and_detect_cell ON SECTION {section}')

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
                classification = -2 #DEFAULT: CELL CANDIDATE IS NOT ACTUALLY A CELL
                if __mean > threshold: #CANDIDATE IS A CELL
                    classification = 2
                elif __mean > -threshold and __mean <= threshold:
                    classification = 0 #UNKNOWN/UNSURE
                predictions.append(classification)
            return predictions

        drops = ['animal', 'section', 'index', 'row', 'col']        
        cell_features_selected_columns = cell_features.drop(drops,axis=1)
        _mean, _std = calculate_scores(cell_features_selected_columns, model_file)#STEP 4-2-1-2) calculate_scores(features) - CALCULATES SCORES, LABELS, MEAN STD FOR EACH FEATURE

        # STEP 4-2-1-3) PREDICTIVE CELL LABELING BASED ON MEAN
        cell_features['mean_score'] = _mean
        cell_features['std_score'] = _std
        cell_features['predictions'] = np.array(get_prediction_and_label(_mean)) #PUTATIVE ID: POSITIVE (2), NEGATIVE (-2), UNKNOWN/UNSURE (0)

        # STEP 4-2-2) STORES DATAFRAME AS CSV FILE
        if debug:
            print(f'CELL LABELS OUTPUT DIR: {OUTPUT}')
        Path(OUTPUT).mkdir(parents=True, exist_ok=True)
        cell_features.to_csv(Path(OUTPUT, f'detections_{str_section_number}.csv'), index=False)
        if debug:
            print('COMPLETED detect_cell')

    def capture_total_sections(self, input_format: str, INPUT):
        '''PART OF STEP 1. USE DASK TO 'TILE' IMAGES
        '''
        if input_format == 'tif': #READ FULL-RESOLUTION TIFF FILES (FOR NOW)
            if os.path.exists(INPUT):
                total_sections = len(sorted(os.listdir(INPUT)))
            else:
                print(f'ERROR: INPUT FILEs {INPUT} NOT FOUND')
                sys.exit()
        else:
            '''ALT PROCESSING: READ OME-ZARR DIRECTLY - ONLY WORKS ON V. 0.4 AS OF 8-DEC-2023
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

    def parse_cell_labels(self):
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

        # TODO: RESOLUTION STORED IN meta-data.json
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
            print(f'ERROR: {self.cell_label_path} NOT FOUND')
            sys.exit(1)
        dataframe_data = []
        detection_files = sorted(glob.glob( os.path.join(self.cell_label_path, f'detections_*.csv') ))
        if len(detection_files) == 0:
            print(f'ERROR: NO CSV FILES FOUND IN {self.cell_label_path}')
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
        # 'WORKAROUND' FOR DB TIMEOUT [SAVE TO FILE AND THEN SQL INSERT FROM FILE]
        # LONG-TERM STORE IN www FOLDER FOR DIRECT IMPORT
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
        spatial_dir = os.path.join(self.fileLocationManager.neuroglancer_data, 'predictions', 'spatial0')
        info_dir = os.path.join(self.fileLocationManager.neuroglancer_data, 'predictions')
        if os.path.exists(spatial_dir):
            print(f'Removing existing directory {spatial_dir}')
            shutil.rmtree(spatial_dir)
        os.makedirs(spatial_dir, exist_ok=True)
        point_filename = os.path.join(spatial_dir, '0_0_0.gz')
        info_filename = os.path.join(info_dir, 'info')
        # dataframe_data = [(x*xy_resolution,y*xy_resolution,z*z_resolution) for x,y,z in dataframe_data]
        with open(point_filename, 'wb') as outfile:
            buf = struct.pack('<Q', len(dataframe_data))
            pt_buf = b''.join(struct.pack('<3f', x, y, z) for (x, y, z) in dataframe_data)
            buf += pt_buf
            id_buf = struct.pack('<%sQ' % len(dataframe_data), *range(len(dataframe_data)))
            buf += id_buf
            bufout = gzip.compress(buf)
            outfile.write(bufout)
            print(f'Wrote {len(dataframe_data)} neurons to {point_filename}')
            self.section_count = 348
            chunk_size = [self.sqlController.scan_run.width, self.sqlController.scan_run.height, self.section_count]
            info = {}
            spatial = {}
            spatial['chunk_size'] = chunk_size
            spatial['grid_shape'] = [1, 1, 1]
            spatial['key'] = 'spatial0'
            spatial['limit'] = 10000
            info['@type'] = "neuroglancer_annotations_v1"
            info['annotation_type'] = "POINT"
            info['by_id'] = {'key':'spatial0'}
            info['dimensions'] = {'x':[str(xy_resolution),'um'],
                                'y':[str(xy_resolution),'um'],
                                'z':[str(z_resolution),'um']}
            info['lower_bound'] = [0,0,0]
            info['upper_bound'] = chunk_size
            info['properties'] = []
            info['relationships'] = []
            info['spatial'] = [spatial]    

            with open(info_filename, 'w') as infofile:
                json.dump(info, infofile, indent=2)
                print(f'Wrote {info} to {info_filename}')

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

    def create_training():
        pass

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