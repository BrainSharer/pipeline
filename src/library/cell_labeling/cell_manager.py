import os, sys, glob, json, math
import inspect
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import dask
from dask import delayed
import dask.array as da
from pathlib import Path
import numpy as np
from compress_pickle import dump, load
import tifffile

from library.utilities.cell_utilities import (
    filter_cell_candidates,
    find_connected_segments,
    load_image,
    subtract_blurred_image
)
from library.image_manipulation.histogram_maker import make_histogram_full_RAM


class CellSegmenter():
    '''Handles cell segmentation methods'''

    def start_labels(self):
        '''1. Use dask to create virtual tiles of full-resolution images
                Which mode (dye/virus: Neurotrace/GFP is for which directory) -> rename input
                label_of_interest_channel a.k.a. virus channel (channel 3)
                counterstain_channel a.k.a. dye channel (channel 1) 

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
            print(f"DEBUG: START IMAGE SEGMENTATION")
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} Start")
        
        # TODO: Need to address scenario where >1 dye or virus channels are present [currently only 1 of each is supported]
        counterstain_modes = ['counterstain', 'dye', 'ntb']
        label_of_interest_modes = ['label_of_interest', 'virus', 'ctb']
        for channel_number, channel_data in self.meta_channel_mapping.items():
            if channel_data['mode'] in counterstain_modes:
                self.counterstain_channel = channel_number
                print(f'counterstain channel (ex. dye, ntb) detected: {self.counterstain_channel}')
                self.fileLogger.logevent(f'counterstain channel (ex. dye, ntb) detected: {self.counterstain_channel}')
            elif channel_data['mode'] in label_of_interest_modes: 
                self.label_of_interest_channel = channel_number
                print(f'label_of_interest channel (ex. virus, CTB) detected: {self.label_of_interest_channel}')
                self.fileLogger.logevent(f'label_of_interest channel (ex. virus, ctb) detected: {self.label_of_interest_channel}')
            elif channel_data['mode'] == 'unknown':
                continue
            else:
                msg = "Neuroanatomical_tracing is missing either counterstain (ex. dye, ntb) or label_of_interest (ex. virus, ctb) channels."
                if self.debug:
                    print(msg)
                self.fileLogger.logevent(msg)
                raise ValueError(msg)

        self.input_format = 'tif' #options are 'tif' and 'ome-zarr'
        
        if os.path.exists(self.avg_cell_img_file):
            avg_cell_img = load(self.avg_cell_img_file) #Load average cell image once
        else:
            print(f'Could not find {self.avg_cell_img_file}')
            sys.exit()

        if self.input_format == 'tif':
            counterstain = self.fileLocationManager.get_full_aligned(channel=self.counterstain_channel)
            label_of_interest = self.fileLocationManager.get_full_aligned(channel=self.label_of_interest_channel)
            self.section_count = self.capture_total_sections(self.input_format, counterstain) #Only need single/first channel to get total section count
        else:
            counterstain = Path(self.fileLocationManager.get_neuroglancer(False, channel=self.self.counterstain_channel) + '.zarr')
            label_of_interest = Path(self.fileLocationManager.get_neuroglancer(False, channel=self.label_of_interest_channel) + '.zarr')

            # OME-ZARR Section count may be extracted from meta-data in folder or from meta-data in file [do not use database]
        
        file_keys = []
        if self.process_sections:
            if self.debug:
                print(f'SECTION FILTER APPLIED: {self.process_sections}')
        else:
            self.process_sections = range(self.section_count)
        
        # self.process_sections is a tuple with [lower, upper] range
        if isinstance(self.process_sections, tuple) and len(self.process_sections) == 2:
            start, stop = self.process_sections
            self.process_sections = range(start, stop)
        
        self.pruning_info = {
                        "run_pruning": self.run_pruning,
                        "prune_x_range": self.prune_x_range,
                        "prune_y_range": self.prune_y_range,
                        "prune_area_min": self.prune_amin,
                        "prune_area_max": self.prune_amax,
                        "prune_annotation_ids": self.prune_annotation_ids,
                        "prune_combine_method": self.prune_combine_method
                    }
        
        #SAVE PARAMETERS IN cell_candidates folder
        output_path = Path(self.SCRATCH, 'pipeline_tmp', self.animal, 'cell_candidates_' + str(self.set_id))
        output_path.mkdir(parents=True, exist_ok=True)
        param_file = Path(output_path, 'parameters.json')
        
        params = {
                "animal": self.animal,
                "section_cnt": len(list(self.process_sections)),
                "uuid": self.set_id,
                "segmentation_threshold": self.segmentation_threshold,
                "cell_radius": self.cell_radius,
                "segment_size_min": self.segment_size_min,
                "segment_size_max": self.segment_size_max,
                "sampling": self.sampling,
                "gaussian_blur_standard_deviation_sigmaX": self.gaussian_blur_standard_deviation_sigmaX,
                "gaussian_blur_kernel_size_pixels": list(self.gaussian_blur_kernel_size_pixels),
                'pruning': {
                    "run_pruning": self.run_pruning,
                    "prune_x_range": (self.prune_x_range[0], self.prune_x_range[-1]),
                    "prune_y_range": (self.pruning_info['prune_y_range'][0], self.pruning_info['prune_y_range'][-1]),
                    "prune_area_min": self.pruning_info['prune_area_min'],
                    "prune_area_max": self.pruning_info['prune_area_max'],
                    "prune_annotation_ids": list(self.pruning_info['prune_annotation_ids']),
                    "prune_combine_method": self.pruning_info['prune_combine_method']
                }
            }
        
        with open(param_file, 'w') as f:
            json.dump(params, f, indent=2)

        if self.debug:
            print()
            print('*'*50)
            print('PARAMETERS:')
            for key, value in params.items():
                print(f'\t{key}: ', value)
            print('*'*50)
        else:
            print('*'*50)
            print('TRACKING INFO:')
            print("\tunique set_id:".ljust(20), f"{self.set_id}".ljust(20))
            print('*'*50)

        for section in self.process_sections:
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
                    self.segment_size_min,
                    self.segment_size_max,
                    self.gaussian_blur_standard_deviation_sigmaX,
                    self.gaussian_blur_kernel_size_pixels,
                    self.SCRATCH,
                    self.OUTPUT,
                    avg_cell_img,
                    self.model_file,
                    self.input_format,
                    counterstain,
                    label_of_interest,
                    self.step,
                    self.task,
                    self.set_id,
                    self.pruning_info,
                    self.debug,
                ]
            )

        if self.debug:
            workers=1
            print(f'Running in debug mode with {workers} workers; {len(file_keys)} sections to process, out: {self.SCRATCH}')
        else:
            workers = math.floor(min([self.get_nworkers(), 10])*.5) # max 50% of prev. calcs [dask is ram intensive]
            print(f'running in parallel with {workers} workers; {len(file_keys)} sections to process, out: {self.SCRATCH}')
        
        self.run_commands_concurrently(detect_cells_all_sections, file_keys, workers)
        

    def identify_cell_candidates(self, file_keys: tuple) -> list:
        '''2. Identify cell candidates - PREV: find_examples()
                -dask virtual tiling set to 1x1 due to RAM >100GB
                -if <100GB, may need to tile images

                This single method will be run in parallel for each section-consists of 3 sub-steps:
                A) subtract_blurred_image (average the image by subtracting gaussian blurred mean)
                B) identification of cell candidates based on connected segments
                C) filering cell candidates based on size and shape

        NOTE 1: May also include pruning on x, y range ['prune_x_range', 'prune_y_range']
        NOTE 2: '*_' (unpacking file_keys tuple will discard vars after debug if set); must modify if file_keys is changed
        :return: a list of cell candidates (dictionaries) for each section

        N.B.: If full-resolution, single image is used, generate single histogram while image still in RAM
        '''
        (
            animal,
            section,
            str_section_number,
            segmentation_threshold,
            cell_radius,
            segment_size_min,
            segment_size_max,
            gaussian_blur_standard_deviation_sigmaX,
            gaussian_blur_kernel_size_pixels,
            SCRATCH,
            _, _, _,
            input_format,
            counterstain,
            label_of_interest,
            _,
            task,
            set_id,
            pruning_info,
            debug,
            *_,
        ) = file_keys

        if not os.path.exists(label_of_interest):
            print(f'ERROR: {label_of_interest} not found')
            sys.exit(1)
        if not os.path.exists(counterstain):
            print(f'ERROR: {counterstain} not found')
            sys.exit(1)

        #ALL FINAL PATHS WILL INCORPORATE UNIQUE SET ID (uuid)
        output_path = Path(SCRATCH, 'pipeline_tmp', animal, 'cell_candidates_' + str(set_id))
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = Path(output_path, f'extracted_cells_{str_section_number}.gz')
        
        output_file_diff_path = Path(SCRATCH, 'pipeline_tmp', animal, 'DIFF_'+ str(set_id))
        output_file_diff_path.mkdir(parents=True, exist_ok=True)
        output_file_diff3 = Path(output_file_diff_path, f'{str_section_number}.tif')
        
        #TODO check if candidates already extracted
        if os.path.exists(output_file):
            print(f'Cell candidates already extracted. Using: {output_file}')
            cell_candidates = load(output_file)
            return cell_candidates
        else:
            cell_candidates = []

        # TODO: CLEAN UP - maybe extend dask to more dimensions?
        if input_format == 'tif':#section_number is already string for legacy processing 'tif' (zfill)
            if debug:
                print('*'*50)
                print(f'READING "tif" ALIGNED FORMAT')
            input_file_label_of_interest = Path(label_of_interest, str_section_number + '.tif')
            input_file_counterstain = Path(counterstain, str_section_number + '.tif')

        else:
            store = parse_url(label_of_interest, mode="r").store
            reader = Reader(parse_url(label_of_interest))
            nodes = list(reader())
            image_node = nodes[0]  # first node is image pixel data
            dask_data = image_node.data
            total_sections = dask_data[0].shape[2]

            input_file_label_of_interest = []
            for img in dask_data[0][0][0]:
                input_file_label_of_interest.append(img)

            # store = parse_url(counterstain, mode="r").store #TODO: delete
            reader = Reader(parse_url(counterstain))
            nodes = list(reader())
            image_node = nodes[0]  # first node is image pixel data
            dask_data = image_node.data
            # total_sections = dask_data[0].shape[2] #TODO: delete

            input_file_counterstain = []
            for img in dask_data[0][0][0]:
                input_file_counterstain.append(img)

            # TODO: different processing for ome-zarr
            # see del_img_extract2.py (this folder) for more info

        # Create delayed tasks for loading the images (separate task list per channel)
        delayed_tasks_label_of_interest = [delayed(load_image)(path) for path in [input_file_label_of_interest]]
        delayed_tasks_counterstain = [delayed(load_image)(path) for path in [input_file_counterstain]]

        # Get shape without computing
        org_label_of_interest_img_shape = dask.compute(delayed_tasks_label_of_interest[0].shape) 
        org_counterstain_img_shape = dask.compute(delayed_tasks_counterstain[0].shape) 

        # Shape should be same for both channels (stores as y-axis then x-axis)
        y_dim = org_label_of_interest_img_shape[0][0]
        x_dim = org_label_of_interest_img_shape[0][1]
        counterstain_y_dim = org_counterstain_img_shape[0][0]
        counterstain_x_dim = org_counterstain_img_shape[0][1]

        if debug:
            print(f'[LABEL_OF_INTEREST] {org_label_of_interest_img_shape=}: {y_dim=}, {x_dim=}')
            print(f'[COUNTERSTAIN] {org_counterstain_img_shape=}: {counterstain_y_dim=}, {counterstain_x_dim=}')

        # both channels must same dimensions; DO NOT PROCEED IF NOT TRUE!
        assert (x_dim == counterstain_x_dim and y_dim == counterstain_y_dim), (
            f"Dimension mismatch between virus and dye channels: "
            f"label_of_interest/ctb/virus=({y_dim}, {x_dim}), counterstain/dye=({counterstain_y_dim}, {counterstain_x_dim})"
        )

        # Create a Dask array from the delayed tasks (NOTE: DELAYED); SHAPE IS y-axis then x-axis
        image_stack_label_of_interest = [da.from_delayed(v, shape=(y_dim, x_dim), dtype='uint16') for v in delayed_tasks_label_of_interest]
        image_stack_counterstain = [da.from_delayed(v, shape=(y_dim, x_dim), dtype='uint16') for v in delayed_tasks_counterstain]

        data_label_of_interest = dask.compute(image_stack_label_of_interest[0])[0] #FULL IMAGE
        data_counterstain = dask.compute(image_stack_counterstain[0])[0] #FULL IMAGE

        # Swap x and y axes (read in y-axis, then x-axis but we want x,y)
        data_label_of_interest = np.swapaxes(data_label_of_interest, 1, 0)
        data_counterstain = np.swapaxes(data_counterstain, 1, 0)
    
        # FINAL VERSION BELOW:
        total_virtual_tile_rows = 1
        total_virtual_tile_columns = 1
        x_window = int(math.ceil(x_dim / total_virtual_tile_rows))
        y_window = int(math.ceil(y_dim / total_virtual_tile_columns))

        if debug:
            print(f'dask parameters: {x_window=}, {y_window=}; {total_virtual_tile_rows=}, {total_virtual_tile_columns=}')

        overlap_pixels = self.cell_radius * 1.5
        full_difference_label_of_interest = np.zeros(data_label_of_interest.shape, dtype=np.uint16)
        difference_counterstain = []
        
        for row in range(total_virtual_tile_rows):
            for col in range(total_virtual_tile_columns):
            
                # MODS FOR VIRTUAL TILE OVERLAP (TO CAPTURE CELL AT EDGES)
                if row == 0:
                    x_start = 0
                else:
                    x_start = int(row * x_window - overlap_pixels)
                
                if col == 0:
                    y_start = 0
                else:
                    y_start = int(col * y_window - overlap_pixels)

                x_end = int(x_window * (row + 1))
                y_end = int(y_window * (col + 1))

                # Ensure we don't go beyond image boundaries
                x_start = max(0, x_start)
                y_start = max(0, y_start)
                x_end = min(data_label_of_interest.shape[0], x_end)
                y_end = min(data_label_of_interest.shape[1], y_end)

                image_roi_label_of_interest = data_label_of_interest[x_start:x_end, y_start:y_end] #image_roi IS numpy array
                image_roi_counterstain = data_counterstain[x_start:x_end, y_start:y_end] #image_roi IS numpy array

                absolute_coordinates = (x_start, x_end, y_start, y_end)

                if debug:
                    print('CALCULATING DIFFERENCE FOR LABEL OF INTEREST (A.K.A. VIRUS, CTB, CH3)')
                    print(f'ROI: {x_start=}, {x_end=}, {y_start=}, {y_end=}')

                difference_label_of_interest = subtract_blurred_image(image_roi_label_of_interest, self.gaussian_blur_standard_deviation_sigmaX, self.gaussian_blur_kernel_size_pixels, self.segmentation_make_smaller, debug) #calculate img difference for virus channel (e.g. fluorescence)
                full_difference_label_of_interest[x_start:x_end, y_start:y_end] = difference_label_of_interest

                connected_segments = find_connected_segments(difference_label_of_interest, segmentation_threshold)

                if connected_segments[0] > 2: # found cell candidate (first element of tuple is count)
                    if debug:
                        print(f'FOUND CELL CANDIDATE: COM-{absolute_coordinates=}, {cell_radius=}, {str_section_number=}')
                    if task != 'segment':
                        if debug:
                            print('CALCULATING DIFFERENCE FOR COUNTERSTAIN (A.K.A. DYE, NTB, CH1)')
                        difference_counterstain = subtract_blurred_image(image_roi_counterstain, self.gaussian_blur_standard_deviation_sigmaX, self.gaussian_blur_kernel_size_pixels, self.segmentation_make_smaller, debug)  # Calculate img difference for dye channel (e.g. neurotrace)

                    cell_candidate = filter_cell_candidates(
                        animal,
                        section,
                        connected_segments,
                        segment_size_min,
                        segment_size_max,
                        cell_radius,
                        x_window,
                        y_window,
                        absolute_coordinates,
                        difference_counterstain,
                        difference_label_of_interest,
                        task,
                        pruning_info,
                        debug
                    )

                    # For overlapping regions, we need to filter duplicates
                    # Only keep cells whose centers are in the non-overlapping part of the tile
                    if row > 0 or col > 0:  # Not the first row or column
                        non_overlap_x_start = int(row * x_window)
                        non_overlap_y_start = int(col * y_window)
                        
                        filtered_candidates = []
                        for candidate in cell_candidate:
                            # Get absolute coordinates (note: stored as Y,X in absolute_coordinates_YX)
                            abs_y, abs_x = candidate["absolute_coordinates_YX"]
                            
                            # Check if cell center is in non-overlapping region
                            if (abs_x >= non_overlap_x_start and 
                                abs_y >= non_overlap_y_start):
                                filtered_candidates.append(candidate)
                            elif debug:
                                print(f"Excluding candidate at ({abs_x}, {abs_y}) - outside non-overlap region")
                        cell_candidates.extend(filtered_candidates)
                        
                    else:
                        cell_candidates.extend(cell_candidate)
        
        if len(cell_candidates) > 0:
            if debug:
                print(f'Saving {len(cell_candidates)} cell candidates to {output_file}')
                print(f'Saving label_of_interest (ch3_diff) images to {output_file_diff3}')
            dump(cell_candidates, output_file, compression="gzip", set_default_extension=True)

            full_difference_label_of_interest_corrected = np.swapaxes(full_difference_label_of_interest, 0, 1)
            tifffile.imwrite(str(output_file_diff3), full_difference_label_of_interest_corrected)

            #Make histogram for full-resolution image [while still in RAM]
            output_path_histogram = Path(self.fileLocationManager.www, 'histogram', 'DIFF_' + str(set_id), f'{str_section_number}.png')
            output_path_histogram.parent.mkdir(parents=True, exist_ok=True)
            mask_path = Path(self.fileLocationManager.masks, 'C1', 'thumbnail_masked', f'{str_section_number}.tif')
            make_histogram_full_RAM(full_difference_label_of_interest_corrected, output_path_histogram, mask_path, debug)

        return cell_candidates


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

    # Note: MUST be staticmethod to allow pickling (ProcessPoolExecutor)
    from library.cell_labeling.cell_process import CellMaker #import statement cannot got at top of file!

    animal = file_keys[0]
    debug = file_keys[-1]  # debug is last item

    cell_segmenter = CellMaker(animal) #Instantiate class in staticmethod

    if debug: #Last element of tuple is debug
        print(f"DEBUG: auto_cell_labels - STEP 1 & 2 (Identify cell candidates a.k.a. cell segmentation)")
    
    cell_candidates = cell_segmenter.identify_cell_candidates(file_keys) #STEPS 1 & 2. virtual tiling and cell candidate identification

    if debug:
        print(f"DEBUG: found {len(cell_candidates)} cell candidates on section {file_keys[1]}")

    if len(cell_candidates) > 0: #continue if cell candidates were detected [note: this is list of all cell candidates]
        if debug:
            print(f"DEBUG: create cell features with identified cell candidates (auto_cell_labels - step 3)")
        cell_features = cell_segmenter.calculate_features(file_keys, cell_candidates) #Step 3. calculate cell features
        if debug: 
            print(f'DEBUG: start_labels - STEP 4 (Detect cells [based on features])')
            print(f'Cell features: {len(cell_features)}')
        cell_segmenter.score_and_detect_cell(file_keys, cell_features)