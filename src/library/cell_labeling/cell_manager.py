import os, sys, glob, json, math, time
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import dask
from dask import delayed
import dask.dataframe as dd
import dask.array as da
import imageio.v2 as imageio
from pathlib import Path
import cv2
import numpy as np
import pickle as pkl
from compress_pickle import dump, load
import pandas as pd

sys.path.append(os.path.abspath('./../../'))
# from library.utilities.utilities_process import SCALING_FACTOR, test_dir
from library.image_manipulation.filelocation_manager import FileLocationManager


class CellMaker:
    """ML autodetection of cells in image"""
    

    def __init__(self, stack):
        """Set up the class with the name of the file and the path to it's location.

        """
        self.fileLocationManager = FileLocationManager(stack)
        self.channel = 1


    def check_prerequisites(self):
        '''
        CELL LABELING REQUIRES A) AVAILABLE FULL-RESOLUTION IMAGES, B) 2 CHANNELS (NAME, TYPE), C) SCRATCH DIRECTORY, D) OUTPUT DIRECTORY, E) cell_definitions (manual training of what cell looks like)
        TODO: MOVE ASSERTIONS TO SEPARATE FUNCTION (UNIT TEST)
        '''
        #CHECK FOR OME-ZARR (NOT IMPLEMENTED AS OF 22-OCT-2023)
        # INPUT = self.fileLocationManager.get_ome_zarr(channel=self.channel)
        # print(f'OME-ZARR FOUND: {INPUT}') #SEND TO LOG FILE

        #CHECK FOR FULL-RESOLUTION TIFF IMAGES (IF OME-ZARR NOT PRESENT)
        INPUT = self.fileLocationManager.get_full_aligned(channel=self.channel)
        print(f'FULL-RESOLUTION TIFF STACK FOUND: {INPUT}')
        self.logevent(f'FULL-RESOLUTION TIFF STACK FOUND: {INPUT}')

        OUTPUT = self.fileLocationManager.get_cell_labels()
        print(f'CELL LABELS OUTPUT DIR: {OUTPUT}')
        self.logevent(f'CELL LABELS OUTPUT DIR: {OUTPUT}')

        self.SCRATCH = '/scratch' #REMOVE HARD-CODING LATER; SEE IF WE CAN AUTO-DETECT NVME
        print(f'TEMP STORAGE LOCATION: {self.SCRATCH}')
        self.logevent(f'TEMP STORAGE LOCATION: {self.SCRATCH}')

        #CHECK FOR PRESENCE OF meta-data.json
        meta_data_file = 'meta-data.json'
        meta_store = os.path.join(self.fileLocationManager.prep, meta_data_file)

        if os.path.isfile(meta_store):
            #print(f'FOUND CHANNEL INFO; READING FROM {meta_store}')

            #verify you have 2 channels required
            with open(meta_store) as fp:
                info = json.load(fp)
            self.meta_channel_mapping = info['Neuroanatomical_tracing']

            #TODO: MOVE ASSERTIONS TO SEPARATE FUNCTION (UNIT TEST)
            # self.dyes = [item['description'] for item in info['Neuroanatomical_tracing']]
            # assert 'GFP' in self.dyes and 'NeurotraceBlue' in self.dyes
            # print('TWO CHANNELS READY')
            # self.logevent(f"USING 2 CHANNELS FOR AUTOMATIC CELL DETECTION: {self.dyes}")
  
        else:
            #CREATE META-DATA STORE (PULL FROM DATABASE)
            print(f'NOT FOUND; CREATING META-DATA STORE @ {meta_store}')
            
            #steps to create
            channels_count = 3
            self.meta_channel_mapping = {1:{'mode':'dye', 'description':'NeurotraceBlue', 'channel_name': 'C1'}, 3:{'mode':'virus', 'description':'GFP', 'channel_name': 'C3'}}
            meta_data_info = {}
            meta_data_info['Neuroanatomical_tracing'] = self.meta_channel_mapping

            with open(meta_store, 'w') as fp:
                json.dump(meta_data_info, fp, indent=4)

        #CHECK FOR CELL TRAINING DEFINITIONS FILE (average-cell_image.pkl)
        self.avg_cell_img_file = Path(os.getcwd(), 'src', 'library', 'cell_labeling', 'average_cell_image.pkl')
        if self.avg_cell_img_file.is_file():
            print(f'FOUND CELL TRAINING DEFINITIONS FILE @ {self.avg_cell_img_file}')
            self.logevent(f'FOUND CELL TRAINING DEFINITIONS FILE @ {self.avg_cell_img_file}')


    def start_labels(self):
        #only premotor cell type for now
        """1. USE DASK TO 'TILE' IMAGES (previous took most time ~50% of total time)******
                (replace: python $SCRIPT_DIR/generate_tif_tiles.py --animal XXX --fluorescence_image $PROJECT_DIR/data_for_test/fluorescence_image --nissel_stain_image $PROJECT_DIR/data_for_test/nissel_stain_image --disk output_directory)
                Which mode (dye/virus: Neurotrace/GFP is for which directory) -> rename input
                fluorescence_image = GFP a.k.a. virus channel (channel 3)
                nissel_stain_image = Neurotraceblue a.k.a. dye channel (channel 1) aka cell_body

           2. CREATE CELL EXAMPLES - image segmentation
           (python $SCRIPT_DIR/parallel_create_examples.py --animal XXX --disk output_directory --njobs 7)
           -where output_directory is OUTPUT
           -this step will create pickel files totaling size (aggregate) of image stack (approx)
           @ end of this step: we should have in SCRATCH (1 pickel file for each section)

           3. CREATE CELL FEATURES
           @start: check pickel files (count)
           (python $SCRIPT_DIR/parallel_create_features.py --animal XXX --disk output_directory --njobs 7)
           -where output_directory is OUTPUT
           @ end of this step: (1 csv file for each section)
           -add cleanup of SCRATCH (remove temp pickel files)

           4. DETECT CELLS; PRODUCE CONFIDENCE INTERVALS PER CELL 
           (python $SCRIPT_DIR/detect_cell_for_one_brain.py --animal XXX --disk output_directory --round 1 --model $PROJECT_DIR/data_for_test/model/models_example.pkl)
           -no rounds (remove)
           -aggregate all csv files into single for entire brain (current)
           -artifact @end: single csv file for entire brain (coordinates of CoM cells detected)
           from repo: full aligned brain images> 2. tiff image tiles> 3. cell examples> 4. cell features> 5.detection result
        """

        self.logevent(f"DEBUG: start_labels - STEP 1 & 2 (REVISED); START ON IMAGE SEGMENTATION")
                
        for channel_number, channel_data in self.meta_channel_mapping.items():
            if channel_data['mode'] == 'dye':
                self.dye_channel = channel_number
                self.logevent(f'DYE CHANNEL DETECTED: {self.dye_channel}')
            else:
                self.virus_channel = channel_number
                self.logevent(f'VIRUS CHANNEL DETECTED: {self.virus_channel}')

        #CONSTANTS FOR METHODS: 
        #TODO: MOVE SOMEWHERE BESIDES HERE
        segmentation_threshold = 2000 
        cell_radius = 40
        max_segment_size = 100000

        max_processes = self.get_nworkers() #TODO: AUTO-DETECT NUMBER OF CORES, USE MIN (cores, section_count)
        
        INPUT = self.fileLocationManager.get_full_aligned(channel=self.dye_channel)
        OUTPUT = self.SCRATCH
        self.input_format = 'tif' #options are 'tif' and 'ome-zarr'

        #OME-ZARR SECTION COUNT MAY BE EXTRACTED FROM META-DATA IN FOLDER
        section_count = self.capture_total_sections(self.input_format, INPUT) #ONLY NEED SINGLE/FIRST CHANNEL
        
        #ONLY LOAD AVERAGE CELL IMAGE ONCE
        avg_cell_img = load(self.avg_cell_img_file)

        file_keys = []
        for section_number in range(section_count):
            if section_count > 1000:
                str_section_number = str(section_number).zfill(4)
            else:
                str_section_number = str(section_number).zfill(3) 
            file_keys.append([str_section_number, segmentation_threshold, cell_radius, max_segment_size, OUTPUT, avg_cell_img])
        
        #TODO: remove comments [that skip this section]
        if self.debug:
            for file_key in tuple(file_keys):
                self.identify_cell_candidates(file_key)
        else:
            pass

        #     #self.run_commands_concurrently(ng.process_image, file_keys, workers)
        #     file_keys = [self.animal, disk = self.OUTPUT, segmentation_threshold=2000]
        #    self.run_commands_concurrently(create_features_for_all_sections, file_keys, njobs=max_processes)

        # self.create_features_for_all_sections(self.animal, disk = self.OUTPUT, segmentation_threshold=2000, njobs=max_processes)
        
        
        #TODO: REMOVE/INTEGRATE AT END OF CELL CANDIDATE IDENTIFICATION (PER SECTION) - FOR TESTING ONLY
        # section='001'
        # avg_cell_img = load(self.avg_cell_img_file)
        # self.calculate_features(section, avg_cell_img, OUTPUT)

        # if self.debug:
        #     for file_key in tuple(file_keys):
        #         self.calculate_features(file_key)
        # else:
        #     pass

        self.logevent(f"DEBUG: start_labels - STEP 4 (RUN DETECTION)")
        
        
        


    def identify_cell_candidates(self, file_key: tuple):
        '''2. IDENTIFY CELL CANDIDATES - PREV: find_examples()
                -REQUIRES IMAGE TILING OR DASK VIRTUAL TILES PRIOR TO RUNNING

                THIS SINGLE METHOD WILL BE RUN IN PARALLEL FOR EACH SECTION
                -CONSISTS OF 3 SUB-STEPS:
                A) subtract_blurred_image (average the image by subtracting gaussian blurred mean)
                B) example finder (identification of cell candidates)
                C) example saver (saving of cell candidates) 

                #2-1) LOOP THROUGH EACH OF TILES & PROCESS (PREV: load_and_preprocess_image())
                #STEP INVOLVES COMPARING EACH TILE IN BOTH CHANNELS AND FINDING DIFFERENCES
                
                #2-2) IDENTIFY POTENTIAL CELLS: find_connected_segments() #rename

                #2-3) GET EXAMPLES & SAVE cell candidates
        '''
        section_number, segmentation_threshold, cell_radius, max_segment_size, OUTPUT, avg_cell_img = file_key
        output_path = Path(OUTPUT, 'pipeline', self.animal, 'cell_candidates')
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = Path(output_path, f'extracted_cells_{section_number}.gz')
        print(f'DEBUG RUN - animal: {self.animal}, section: {section_number}, segmentation_threshold: {segmentation_threshold}, OUTPUT: {output_path}')

        #TODO: CLEAN UP - maybe extend dask to more dimensions?
        input_path_virus = self.fileLocationManager.get_full_aligned(channel=self.virus_channel)
        input_path_dye = self.fileLocationManager.get_full_aligned(channel=self.dye_channel)
        if self.input_format == 'tif':#section_number IS ALREADY STRING FOR LEGACY PROCESSING 'tif' (zfill)
            input_file_virus = Path(input_path_virus, section_number + '.tif')
            input_file_dye = Path(input_path_dye, section_number + '.tif')
        else:
            #TODO: DIFFERENT PROCESSING FOR OME-ZARR
            #see del_img_extract2.py (this folder) for more info
            pass

        # Create delayed tasks for loading the images (SEPARATE TASK LIST PER CHANNEL)
        delayed_tasks_virus = [delayed(self.load_image)(path) for path in [input_file_virus]]
        delayed_tasks_dye = [delayed(self.load_image)(path) for path in [input_file_dye]]
        
        #GET SHAPE WITHOUT COMPUTING
        org_img_shape = dask.compute(delayed_tasks_virus[0].shape) 

        #SHAPE WILL BE SAME FOR BOTH CHANNELS (stores as y-axis then x-axis)
        x_dim = org_img_shape[0][1]
        y_dim = org_img_shape[0][0]
        
        # Create a Dask array from the delayed tasks (NOTE: DELAYED)
        image_stack_virus = [da.from_delayed(v, shape=(x_dim, y_dim), dtype='uint16') for v in delayed_tasks_virus]
        image_stack_dye = [da.from_delayed(v, shape=(x_dim, y_dim), dtype='uint16') for v in delayed_tasks_dye]
        data_virus = dask.compute(image_stack_virus[0])[0] #FULL IMAGE
        data_dye = dask.compute(image_stack_dye[0])[0] #FULL IMAGE

        #SWAP X AND Y AXES (READ IN Y-AXIS, THEN X-AXIS BUT WE WANT X,Y)
        data_virus = np.swapaxes(data_virus, 1, 0)
        data_dye = np.swapaxes(data_dye, 1, 0)

        #FINAL VERSION BELOW:
        # x_window = virtual_tile_width = x_dim // total_virtual_tiles
        # y_window = virtual_tile_height = y_dim // total_virtual_tiles
        # total_virtual_tile_columns = total_virtual_tile_rows = total_virtual_tiles = 9

        #TODO: FOR TESTING - REMOVE START [RECTANGLE VIRTUAL TILES (WE WILL USE SEE ABOVE CALC GOING FORWARD)]
        total_virtual_tile_rows = 2
        total_virtual_tile_columns = 5
        x_window = virtual_tile_width = int(math.ceil(x_dim / total_virtual_tile_rows))
        y_window = virtual_tile_height = int(math.ceil(y_dim / total_virtual_tile_columns))
        #total_virtual_tile_columns = total_virtual_tile_rows = total_virtual_tiles = total_virtual_tile_rows * total_virtual_tile_columns
        #TODO: FOR TESTING - REMOVE END

        cell_candidates=[]
        for row in range(total_virtual_tile_rows):
            for col in range(total_virtual_tile_columns):
                x_start = row*x_window
                x_end = x_window*(row+1)
                y_start = col*y_window
                y_end = y_window*(col+1)
                
                image_roi_virus = data_virus[x_start:x_end, y_start:y_end] #image_roi IS NUMPY ARRAY
                image_roi_dye = data_dye[x_start:x_end, y_start:y_end] #image_roi IS NUMPY ARRAY
                
                #print(f'DEBUG: image_roi_virus.shape - {image_roi_virus.shape}')
                absolute_coordinates = (x_start, x_end, y_start, y_end)
                
                #CALCULATE DIFFERENCE BETWEEN VIRUS (e.g. FLUORESCENCE) AND DYE (e.g. NEUROTRACE)
                difference_ch3 = self.subtract_blurred_image(image_roi_virus)
                difference_ch1 = self.subtract_blurred_image(image_roi_dye)

                #FIND CONNECTED SEGMENTS (CELL CANDIDATES) -> returns tuple
                connected_segments = self.find_connected_segments(difference_ch3, segmentation_threshold)
                #print(f'debug connected_segments: {connected_segments}')

                if connected_segments[0] > 2: #FOUND CELL CANDIDATE (first element of tuple is count)
                    cell_candidates.append(self.filter_cell_candidates(section_number, connected_segments, max_segment_size, cell_radius, x_window, y_window, absolute_coordinates, difference_ch1, difference_ch3))
                # else:
                #     print(f'NO CELL CANDIDATE (COUNT {connected_segments[0]} DID NOT MEET THRESHOLD: 2)') #TODO: REMOVE

        print(f'DEBUG: len(cell_candidates): {len(cell_candidates)}')

        print(f'SAVE CELL CANDIDATES @ {output_file}')
        dump(cell_candidates, output_file, compression="gzip", set_default_extension=True)

        #MOVE ON TO FEATURE CALCULATION WHILE WE HAVE ALL DATA LOADED IN RAM
        if len(cell_candidates) > 0:
            self.calculate_features(section_number, avg_cell_img, OUTPUT)
        else:
            print(f'NO CELL CANDIDATES FOUND FOR SECTION {section_number}; SKIPPING FEATURE CALCULATION')
    

    def calculate_features(self, section, avg_cell_img, OUTPUT):
        '''3. CALCULATE CELL FEATURES

            THIS SINGLE METHOD WILL BE RUN IN PARALLEL FOR EACH SECTION
            -CONSISTS OF 6 SUB-STEPS:
            A) LOAD THE MANUAL ANNOTATIONS TRAINING DATA (AVERAGE HISTORICAL CELL IMAGE: average_cell_image.pkl)
            B) load information from cell candidates (pickle files)
            C1) calculate_correlation_and_energy FOR CHANNEL 1
            C2) calculate_correlation_and_energy FOR CHANNEL 3
            D) features_using_center_connectd_components(example)
            E) SAVE FEATURES (CSV FILE)

            REQUIRES PRIOR ID OF CELL CANDIDATES PRIOR TO RUNNING (ALL SECTIONS OR JUST SINGLE SECTION? - ASK KUI)

            PRIOR DOC: Master function, calls methods to calculate the features that are then stored in self.features
        '''
        output_path = Path(OUTPUT, 'pipeline', self.animal, 'cell_features')
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = Path(output_path, f'cell_features_{section}.csv')

        self.logevent(f"DEBUG: start_labels - STEP 3 (CREATE CELL FEATURES)")
        print(f"DEBUG: start_labels - STEP 3 (CREATE CELL FEATURES)")

        #STEP 3-A)
        #MOVED TO PRIOR STEP (CELL CANDIDATE IDENTIFICATION); NOW PASSED AS INPUT PARAMETER

        #STEP 3-B) load information from cell candidates (pickle files from step 2 - cell candidate identification)
        input_file = Path(f'/scratch/pipeline/{self.animal}/cell_candidates/', f'extracted_cells_{section}.gz')
        print(f'reading: {input_file}')
        cell_candidate_data = load(input_file)
        try:
            print(f'CELL CANDIDATES COUNT: {len(cell_candidate_data[0])}')
        except:
            print('NOT AVAILABLE')

        output_spreadsheet = []
        for cell in range(len(cell_candidate_data[0])):
            #VALIDATE FILE - MAY NOT BE NECESSARY IF WE DON'T HAVE FILES
            if self.animal == cell_candidate_data[0][cell]["animal"] and section == cell_candidate_data[0][cell]["section"]:
                absolute_coordinates_YX = cell_candidate_data[0][cell]["absolute_coordinates_YX"]
                image_CH3 = cell_candidate_data[0][cell]['image_CH3']
                image_CH1 = cell_candidate_data[0][cell]['image_CH1']
                
                #STEP 3-C1, 3-C2) calculate_correlation_and_energy FOR CHANNELS 1 & 3 (ORG. FeatureFinder.py; calculate_features())
                ch1_corr, ch1_energy = self.calculate_correlation_and_energy(avg_cell_img['CH1'], image_CH1)
                ch3_corr, ch3_energy = self.calculate_correlation_and_energy(avg_cell_img['CH3'], image_CH3)

                #STEP 3-D) features_using_center_connectd_components
                ch1_contrast, ch3_constrast, moments_data = self.features_using_center_connected_components(cell_candidate_data[0][cell])

                # print(f'DEBUG: absolute_coordinates_YX: {absolute_coordinates_YX}')
                # print(f'DEBUG: ch1_corr: {ch1_corr}, ch1_energy: {ch1_energy}, ch3_corr: {ch3_corr}, ch3_energy: {ch3_energy}')
                # print(f'DEBUG: ch1_contrast: {ch1_contrast}, ch3_constrast: {ch3_constrast}, moments_data: {moments_data}')

                #BUILD FEATURES DICTIONARY - can we remove 'label'?
                spreadsheet_row = {'animal': self.animal, 'section': section, 'index': cell, 'label': 0, 'area': cell_candidate_data[0][cell]['area'], 
                                      'height': cell_candidate_data[0][cell]['cell_shape_YX'][0], 'width': cell_candidate_data[0][cell]['cell_shape_YX'][1], 'row': absolute_coordinates_YX[0], 'col': absolute_coordinates_YX[1], 
                                      'corr_CH1': ch1_corr, 'energy_CH1': ch1_energy, 'corr_CH3': ch3_corr, 'energy_CH3': ch3_energy, 'contrast1': ch1_contrast, 'contrast3': ch3_constrast}
                spreadsheet_row.update(moments_data[0])
                output_spreadsheet.append(spreadsheet_row)

        # print(f'DEBUG: output_spreadsheet: {output_spreadsheet}')

        #TODO:contrast and moments are wrong
        #maybe move to dask dataframe, time permitting

        df = pd.DataFrame(output_spreadsheet)
        df.to_csv(output_file, index=False)


        #TODO: REMOVE - NOTES:
        #we need to provide absolute coordinates for each feature (already in pickle file: absolute_coordinates_YX)
        #this will replace: self.copy_information_from_examples(example)

        #OLD WILLIAM CONFUSING MESS - from calculate_features()
        # for tilei in range(len(self.Examples)):
        #     print(f'processing {tilei}')
        #     examples_in_tilei = self.Examples[tilei]
        #     if examples_in_tilei != []:
        #         for examplei in range(len(examples_in_tilei)):
        #             example=examples_in_tilei[examplei]
        #             self.featurei={}
        #             self.copy_information_from_examples(example) #skip because we have absolute coordinates **step 3B
        #             self.calculate_correlation_and_energy(example,channel=1) ***step 3C, 3D
        #             self.calculate_correlation_and_energy(example,channel=3)
        #             self.features_using_center_connectd_components(example)
        #             self.features.append(self.featurei)
        

    def calculate_correlation_and_energy(self, avg_cell_img, cell_candidate_img):  
        '''PART OF STEP 3. CALCULATE CELL FEATURES

        CALCULATE CORRELATION [BETWEEN cell_candidate_img AND avg_cell_img] and AND ENERGY FOR CELL CANIDIDATE

        NOTE: avg_cell_img AND cell_candidate_img CONTAIN RESPECTIVE CHANNELS PRIOR TO PASSING IN ARGUMENTS
        N.B. THIS METHOD CONSOLIDATED WITH calc_img_features()

        PRIOR DOCSTRING: 
        img = input image
        mean_s: the untrimmed mean image
        Computes the agreement between the gradient of the mean image and the gradient of this example
        mean_x,mean_y = the gradients of the particular image
        img_x,img_y = the gradients of the image
        '''

        #ENSURE IMAGE ARRAYS TO SAME SIZE
        cell_candidate_img, avg_cell_img = self.equalize_array_size_by_trimming(cell_candidate_img, avg_cell_img)

        #COMPUTE NORMALIZED SOBEL EDGE MAGNITUDES
        avg_cell_img_x, avg_cell_img_y = self.sobel(avg_cell_img)
        cell_candidate_img_x, cell_candidate_img_y = self.sobel(cell_candidate_img)
        
        #corr = the mean correlation between the dot products at each pixel location
        dot_prod = (avg_cell_img_x * cell_candidate_img_x) + (avg_cell_img_y * cell_candidate_img_y)
        corr = np.mean(dot_prod.flatten())      
        
        #energy: the mean of the norm of the image gradients at each pixel location
        mag = np.sqrt(cell_candidate_img_x **2 + cell_candidate_img_y **2)
        energy = np.mean((mag * avg_cell_img).flatten())  

        return corr, energy


    def equalize_array_size_by_trimming(self, array1, array2):
        '''PART OF STEP 3. CALCULATE CELL FEATURES
        
        PRIOR DOCSTRING:
        makes array1 and array 2 the same size
        '''
        size0 = min(array1.shape[0], array2.shape[0])
        size1 = min(array1.shape[1], array2.shape[1])
        array1 = self.trim_array_to_size(array1, size0, size1)
        array2 = self.trim_array_to_size(array2, size0, size1)
        return array1, array2    

    def trim_array_to_size(self, array, size0, size2):
        '''PART OF STEP 3. CALCULATE CELL FEATURES
        
        PRIOR DOCSTRING:
        trims an array to size
        '''
        if(array.shape[0]>size0):
            size_difference=int((array.shape[0]-size0)/2)
            array=array[size_difference:size_difference+size0,:]
        if(array.shape[1]>size2):
            size_difference=int((array.shape[1]-size2)/2)
            array=array[:,size_difference:size_difference+size2]
        return array

    def sobel(self, img):
        '''PART OF STEP 3. CALCULATE CELL FEATURES

        PRIOR DOCSTRING:
        Compute the normalized sobel edge magnitudes
        '''
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        _mean = (np.mean(sobel_x) + np.mean(sobel_y))/2.
        _std = np.sqrt((np.var(sobel_x)+np.var(sobel_y))/2)
        sobel_x = (sobel_x - _mean) / _std
        sobel_y = (sobel_y - _mean) / _std
        return sobel_x, sobel_y

    def features_using_center_connected_components(self, cell_candidate_data):   
        '''PART OF STEP 3. CALCULATE CELL FEATURES

        #CONTAINS 4 SUB-METHODS: mask_mean(), append_string_to_every_key(), calc_moments_of_mask(), calc_contrasts_relative_to_mask()
        #POPULATES image1, image3 with each channel image (is this the average image?)
        #POPULATES mask WITH MASK FROM CELL CANDIDATE (is there an average mask?)
        #CALCULATES 'MOMENTS OF MASK' (COULD THIS BE ANY MORE CONFUSING? IS THIS A HEART-TO-HEART?)
        #CALCULATES CONTRASTS RELATIVE TO MASK (WHAT IS THIS?) SOMETHING WITH MASK MEAN BETWEEN THE CHANNELS
        
        PRIOR DOCSTRING: calculated designed features for detection input
        '''
        def mask_mean(mask,image):
            mean_in=np.mean(image[mask==1])
            mean_all=np.mean(image.flatten())
            return (mean_in-mean_all)/(mean_in+mean_all)    # calculate the contrast: mean

        def append_string_to_every_key(dictionary, post_fix): 
            return dict(zip([keyi + post_fix for keyi in dictionary.keys()],dictionary.values()))
        
        def calc_moments_of_mask(mask):   
            # calculate moments (how many) and Hu Moments (7)
            mask = mask.astype(np.float32)
            moments = cv2.moments(mask)
            """
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
            Hu Moments are described in this paper: https://www.researchgate.net/publication/224146066_Analysis_of_Hu's_moment_invariants_on_image_scaling_and_rotation
            """
            
            huMoments = cv2.HuMoments(moments)
            moments = append_string_to_every_key(moments, f'_mask')
            return (moments, {'h%d'%i+f'_mask':huMoments[i,0]  for i in range(7)})
        
        mask = cell_candidate_data['mask']  
        moments_data = calc_moments_of_mask(mask)

        #CALCULATE CONSTRASTS RELATIVE TO MASK
        ch1_contrast = mask_mean(mask, cell_candidate_data['image_CH1'])
        ch3_constrast = mask_mean(mask, cell_candidate_data['image_CH3'])

        return (ch1_contrast, ch3_constrast, moments_data)


    def detect_cell(self, file_key: tuple):
        ''' PART OF STEP 4. DETECT CELLS; PRODUCE CONFIDENCE INTERVALS PER CELL

        PRIOR DOCSTRING:
        30 previously trained models are used to calculate a prediction score for features calculated in step
        The mean and standard deviation of the 30 detectors are then used to make a decision if a candidate is a sure or unsure detection.'''
        
        #CREATES INSTANCE OF CellDetector (loads models, Predictor?)
        #Predictor IS ACTUALL A CLASSIFICATION OF CELL TYPE TO 'sure, unsure, not cell' (VERY CONFUSING) BASED ON 2=sure, -2=no_cell, 0=unsure
        #BASED ON MEAN IS LOWER THAN NEGATIVE STANDARD DEVIATION [-2, NOT A CELL]
        # MEAN IS HIGHER THAN STANDARD DEVIATION [2, CELL DETECTED, SURE]
        # MEAN IS BETWEEN NEGATIVE STANDARD DEVIATION AND POSITIVE STANDARD DEVIATION [0, UNSURE/UNKNOWN]

        #AT LEAST KUI WROTE THIS PART SO I WILL ASK HIM
        #STEP 4-1) CREATE INSTANCE OF CELL DETECTOR

        #STEP 4-2) calculate_and_save_detection_results, INCLUDES get_detection_results AND SAVE TO CSV
        #STEP 4-2-1) get_detection_results()

        #STEP 4-2-1-1) get_combined_features_for_detection() -ALL THIS DOES IS READ A CSV FILE *(section, index, row, column)? WOW!
        #MAYBE DON'T STORE ANIMAL IN CSV SINCE WE JUST DROP THAT COLUMN ANYWAY
        #STORES FEATURES IN features VARIABLE (WHAT IS A FEATURE?)

        #STEP 4-2-1-2) calculate_scores(features) - CALCULATES SCORES, LABELS, MEAN STD FOR EACH FEATURE

        #STEP 4-2-1-3) get_prediction(_mean,_std) - CALCULATES PREDICTION BASED ON MEAN AND STD

        #STEP 4-2-1.4) get_combined_features()
        #READS PRIOR STORED FEATURES FROM CSV FILE (section, index, row, column) OR CREATES IF NOT EXIST

        #create_combined_features

        #STEP 4-2-1-5) ADD EVERYTHING TO A DATAFRAME AND RETURN 'detection_df'

        #positive, negative, unsure
        

        """
        PRIOR CODE BELOW:
        features = self.get_combined_features_for_detection()
        scores,labels,_mean,_std = self.detector.calculate_scores(features)
        predictions=self.detector.get_prediction(_mean,_std)
        detection_df = self.get_combined_features()
        detection_df['mean_score'],detection_df['std_score'] = _mean,_std
        detection_df['label'] = labels
        detection_df['predictions'] = predictions
        detection_df = detection_df[['animal', 'section', 'row', 'col','label', 'mean_score','std_score', 'predictions']]
        """

        #STEP 4-2-2) STORES DATAFRAME AS CSV FILE


    def capture_total_sections(self, input_format: str, INPUT):
        '''PART OF STEP 1. USE DASK TO 'TILE' IMAGES
        '''
        if input_format == 'tif':
            #READ FULL-RESOLUTION TIFF FILES (FOR NOW)
            #INPUT = self.fileLocationManager.get_full_aligned(channel=channel)
            total_sections = len(sorted(os.listdir(INPUT)))
        else:
            '''ALT PROCESSING: READ OME-ZARR DIRECTLY AND STORE REFERENCES TO EACH SECTION
                FOLLOWING CODE FUNCTIONAL AS OF 9-NOV-2023
            '''
            #INPUT = self.fileLocationManager.get_ome_zarr(channel=self.channel)
            #print(f'INPUT: {INPUT}')
            
            # Open the OME-Zarr file
            store = parse_url(INPUT, mode="r").store
            reader = Reader(parse_url(INPUT))
            nodes = list(reader())
            image_node = nodes[0]  # first node is image pixel data
            dask_data = image_node.data
            total_sections = dask_data[0].shape[2]
            #print(f'total sections_OME-Zarr: {total_sections}')
        return total_sections
    

    def load_image(self, file: str):
        return imageio.imread(file)
    

    def subtract_blurred_image(self, image):
        '''PART OF STEP 2. IDENTIFY CELL CANDIDATES
        
        PRIOR DOCSTRING:
        average the image by subtracting gaussian blurred mean
        '''
        image = np.float32(image)

        small = cv2.resize(image, (0, 0), fx=0.05, fy=0.05, interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(small, ksize=(21, 21), sigmaX=10)
        relarge = cv2.resize(blurred, image.T.shape, interpolation=cv2.INTER_AREA)
        difference = image - relarge
        
        return difference
    

    def find_connected_segments(self, image, segmentation_threshold) -> tuple:
        '''PART OF STEP 2. IDENTIFY CELL CANDIDATES
        
        PRIOR DOCSTRING:
        find connected segments (cell candidates)
        '''
        n_segments, segment_masks, segment_stats, segment_location = cv2.connectedComponentsWithStats(np.int8(image > segmentation_threshold))
        segment_location = np.int32(segment_location)
        segment_location = np.flip(segment_location, 1) #why do we need to reverse axis order? Was William coding in a mirror?
        return (n_segments, segment_masks, segment_stats, segment_location)


    def filter_cell_candidates(self, section_number, connected_segments, max_segment_size, cell_radius, x_window, y_window, absolute_coordinates, difference_ch1, difference_ch3):
        '''PART OF STEP 2. IDENTIFY CELL CANDIDATES
        
        PRIOR DOCSTRING:
        creates examples for one tile

        :type tile: _type_
        :return: examples
        :rtype: _type_

        connected_segments: tuple - contains information about cell candidates, including dimensions of bounding boxes around cell candidates
        
        Area is for the object, where pixel values are not zero
        Segments are filtered to remove those that are too large or too small
        '''
        n_segments, segment_masks, segment_stats, segment_location = connected_segments
        # print(f'DEBUG segment_location: {segment_location}')
        #segment_stats, segment_location may not be required if we are passing abosolute coordinates

        #print(f'DEBUG - n_segments, segment_masks, segment_stats, segment_location: {n_segments, segment_masks, segment_stats, segment_location}')
        Examples = []
        for segmenti in range(n_segments):
            _, _, width, height, object_area = segment_stats[segmenti, :]

            if object_area > max_segment_size:
                continue
            segment_row, segment_col = segment_location[segmenti, :]
            
            #print(f'DEBUG - segment_row, segment_col: {segment_row, segment_col}')
            row_start = int(segment_row - cell_radius)
            col_start = int(segment_col - cell_radius)
            if row_start < 0 or col_start < 0:
                continue
            row_end = int(segment_row + cell_radius)
            col_end = int(segment_col + cell_radius)

            # print(f'DEBUG - row_start, row_end: {row_start, row_end}')
            # print(f'DEBUG - col_start, col_end: {col_start, col_end}')
            # print(f'DEBUG - compare row_end with x_window: {row_end, x_window}')
            # print(f'DEBUG - compare col_end with y_window: {col_end, y_window}')
            
            #ROW EVALUATES WITH X-AXIS (WIDTH), COL EVALUATES WITH Y-AXIS (HEIGHT)
            if row_end > x_window or col_end > y_window:
                continue

            segment_mask = segment_masks[row_start:row_end, col_start:col_end]# == segmenti

            candidate = {'animal': self.animal,
                       'section': section_number,
                       'area': object_area,
                    #    'row': absolute_coordinates[0:1],
                    #    'col': absolute_coordinates[2:3],
                    #    'height': height,
                    #    'width': width,
                       'absolute_coordinates_YX': (absolute_coordinates[2]+segment_col, absolute_coordinates[0]+segment_row),
                       'cell_shape_YX': (height, width),
                       'image_CH3': difference_ch3[row_start:row_end, col_start:col_end].T,
                       'image_CH1': difference_ch1[row_start:row_end, col_start:col_end].T,
                       'mask': segment_mask}
            Examples.append(candidate)

        return Examples