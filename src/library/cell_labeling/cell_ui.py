import os, glob, json
import sys
import re
from datetime import datetime
from pathlib import Path
import inspect
from library.utilities.utilities_process import test_dir, M_UM_SCALE, SCALING_FACTOR, get_scratch_dir, use_scratch_dir, read_image, get_hostname, delete_in_background
from library.image_manipulation.image_manager import ImageManager
from library.image_manipulation.histogram_maker import make_single_histogram_full, make_combined_histogram_full
from library.image_manipulation.neuroglancer_manager import NumpyToNeuroglancer
from library.image_manipulation.precomputed_manager import NgPrecomputedMaker, XY_CHUNK, Z_CHUNK
from cloudvolume import CloudVolume
from taskqueue import LocalTaskQueue

from library.omezarr.builder_init import builder
import igneous.task_creation as tc
from library.utilities.cell_utilities import (
    copy_with_rclone
)
from library.utilities.dask_utilities import closest_divisors_to_target

import dask
from distributed import LocalCluster
from dask.distributed import Client


class Cell_UI():

    def __init__(self):
        self.SCRATCH = get_scratch_dir()


    def ng_prep(self):
        '''
        #WIP 21-AUG-2025
        CREATES PRECOMPUTED FORMAT FROM IMAGE STACK ON SCRATCH, MOVES TO 'CH3_DIFF'
        ALSO INCLUDES HISTOGRAM (SINGLE AND COMBINED)

        Note: called from regular pipeline: self.TASK_NG_PREVIEW
        or called from cell labeling pipeline: self.TASK_SEGMENT
        '''
        if self.debug:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} Start")
        
        if self.task == 'segment' or (self.task == 'ng_preview' and self.set_id): #REGENERATE ENTIRE NG FROM SCRATCH
            ch_name = 'DIFF' + '_' + self.set_id
            progress_dir = Path(self.fileLocationManager.neuroglancer_data, 'progress', ch_name)
            
            INPUT_DIR = Path(self.SCRATCH, 'pipeline_tmp', self.animal, ch_name)

            #MOVE src prep file [from scratch] to final location [if not already moved]
            FINAL_PREP_DIR = Path(self.fileLocationManager.prep, ch_name)

            if not INPUT_DIR.exists() or len([item for item in INPUT_DIR.iterdir()]) == 0:
                print(f"WARNING: Input directory {INPUT_DIR} is empty or does not exist; FILES POSSIBLY ALREADY MOVED")
                INPUT_DIR = FINAL_PREP_DIR

                #FINAL CHECK IF NOT RE-RUN (DIFF DIRECTORY WILL NOT EXIST)
                if not FINAL_PREP_DIR.exists() or len([item for item in FINAL_PREP_DIR.iterdir()]) == 0:
                    print('WARNING: Final preparation directory is empty or does not exist; attempting to auto-detect channels')
                    ch_name = f"C{self.channel}"
                    progress_dir = Path(self.fileLocationManager.neuroglancer_data, 'progress', 'NG_PREVIEW' + '_' + self.set_id)
                    if self.channel:
                        if self.downsample:
                            INPUT_DIR = self.fileLocationManager.get_thumbnail_aligned(self.channel)
                            OUTPUT_DIR = Path(self.fileLocationManager.neuroglancer_data, f'{ch_name}T_aligned')
                            if OUTPUT_DIR.exists() and len([item for item in OUTPUT_DIR.iterdir()]) > 0:
                                print('OUTPUT_DIR ALREADY EXISTS; SKIPPING PRECOMPUTATION')
                                return
                        else:
                            INPUT_DIR = self.fileLocationManager.get_full_aligned(self.channel)
                            print("NOT READY AS OF 3-SEP-2025 - SEE DUANE [cell_ui.py]")
                            sys.exit()
            else:
                copy_with_rclone(INPUT_DIR, FINAL_PREP_DIR)
        else: #e.g. 'ng_preview':
            ch_name = f"C{self.channel}"
            print('GENERATING NEUROGLANCER PREVIEW: CALLED FROM REGULAR PIPELINE')
            progress_dir = Path(self.fileLocationManager.neuroglancer_data, 'progress', 'NG_PREVIEW' + '_' + self.set_id)
            if self.channel:
                print('GENERATING PREVIEW FOR CHANNEL:', self.channel)
                if self.downsample:
                    INPUT_DIR = self.fileLocationManager.get_thumbnail_aligned(self.channel)
                else:
                    INPUT_DIR = self.fileLocationManager.get_full_aligned(self.channel)

        temp_output_path = Path(self.SCRATCH, 'pipeline_tmp', self.animal, ch_name + '_ng')
        temp_output_path_pyramid = Path(self.SCRATCH, 'pipeline_tmp', self.animal, ch_name + '_py')
        temp_output_path.mkdir(parents=True, exist_ok=True)
        temp_output_path_pyramid.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR = Path(self.fileLocationManager.neuroglancer_data, ch_name)

        try:
            delete_in_background(progress_dir)
        except Exception as e:
            print(f"Non-critical Error deleting progress directory: {e}")
        progress_dir.mkdir(parents=True, exist_ok=True)    
        
        if self.debug:
            workers = 1    
        else:
            workers = self.get_nworkers()

        

        print('*'*50)
        print("\tINPUT_DIR:".ljust(20), f"{INPUT_DIR}".ljust(20))
        print("\tTEMP ng Output:".ljust(20), f"{temp_output_path}".ljust(20))
        print("\tTEMP DOWNSAMPLED PRECOMPUTED Output:".ljust(20), f"{temp_output_path_pyramid}".ljust(20))
        print("\tTEMP Progress:".ljust(20), f"{progress_dir}".ljust(20))
        print("\tFINAL Output:".ljust(20), f"{OUTPUT_DIR}".ljust(20))
        print("\tNEW NEUROGLANCER WILL BE CREATED [IF NOT EXIST]".ljust(20))
        print('*'*50)

        image_manager = ImageManager(INPUT_DIR)
        chunks = [image_manager.height//16, image_manager.width//16, 1]
        image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.tif')])

        if not OUTPUT_DIR.exists() or not any(OUTPUT_DIR.iterdir()):
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            ################################################
            # PRECOMPUTED FORMAT CREATION 
            ################################################
            expected_z_length = len(os.listdir(self.fileLocationManager.section_web)) #SHOULD ALWAYS EXIST FOR ALL BRAIN STACKS
            full_volume_size = list(image_manager.volume_size)
            full_volume_size[2] = expected_z_length 

            precompute = NgPrecomputedMaker(self.sqlController)
            scales = precompute.get_scales()
            ng = NumpyToNeuroglancer(
                self.animal,
                None,
                scales,
                layer_type="image", 
                data_type=image_manager.dtype,
                num_channels=1,
                chunk_size=chunks
            )

            # Initialize precomputed volume
            ng.init_precomputed(temp_output_path, full_volume_size)

            file_keys = []
            orientation = self.sqlController.histology.orientation
            section_to_file_index = self.create_section_mapping(image_files)
            
            for section_num in range(expected_z_length):
                is_blank = False
                filepath = None
                
                if section_num in section_to_file_index:
                    file_index = section_to_file_index[section_num]
                    f = image_files[file_index]
                    filepath = Path(INPUT_DIR, f)
                else:
                    is_blank = True
                    print(f"Section {section_num} is missing, creating blank placeholder")
                
                file_keys.append([section_num, filepath, orientation, progress_dir, is_blank, image_manager.height, image_manager.width])

            if self.debug:
                for file_key in file_keys:
                    ng.process_image(file_key)
            else:
                self.run_commands_concurrently(ng.process_image, file_keys, workers)

            #################################################
            # PRECOMPUTED FORMAT PYRAMID (DOWNSAMPLED) START
            #################################################
            chunks = [XY_CHUNK, XY_CHUNK, Z_CHUNK]
            mips = 7
            fill_missing = True
            encoding="raw"
            print('CREATING PYRAMID DOWNSAMPLED PRECOMPUTED FORMAT')

            # Use the same temp_output_path for both initial volume and pyramid
            cloudpath = f"file://{temp_output_path}" #full-resolution (already generated)
            # outpath = f"file://{temp_output_path}"
            outpath = f"file://{temp_output_path_pyramid}"

            # Create task queue
            tq = LocalTaskQueue(parallel=workers)
            tasks = tc.create_image_shard_transfer_tasks(
                cloudpath,
                outpath, 
                mip=0, 
                chunk_size=chunks
                # memory_target=memory_target,
                # encoding=encoding,
                # compress=False
            )
            tq.insert(tasks)
            tq.execute()

            # Create downsampling tasks for each mip level
            for mip in range(0, mips):
                cv = CloudVolume(outpath, mip)
                print(f'Creating sharded downsample tasks at mip={mip}')
                tasks = tc.create_image_shard_downsample_tasks(
                    cv.layer_cloudpath,
                    mip=mip,
                    chunk_size=chunks
                    # memory_target=memory_target,
                    # encoding=encoding,
                    # factor=(2,2,1),  # Explicit downsampling factor
                    # sparse=False,
                    # preserve_chunk_size=True
                )
                tq.insert(tasks)
                tq.execute()

                cv = CloudVolume(outpath, mip=mip+1)
                if not cv.info['scales'][mip+1]:  # Alternative way to check MIP existence
                    raise Exception(f"MIP {mip+1} not properly generated")

            print('Finished all downsampling tasks')
            #################################################
            # PRECOMPUTED FORMAT PYRAMID (DOWNSAMPLED) END
            #################################################
            cv = CloudVolume(cloudpath)
            cv.cache.flush()

            ################################################
            # MODIFY PROVENANCE FILE WITH META-DATA
            ################################################
            prov_path = Path(temp_output_path_pyramid, 'provenance')
            try:
                if prov_path.exists():
                    with open(prov_path, 'r') as f:
                        prov = self.clean_provenance(json.load(f))
                else:
                    prov = {'description': "", 'owners': [], 'processing': [], 'sources': {}}
            except json.JSONDecodeError:
                prov = {'description': "", 'owners': [], 'processing': [], 'sources': {}}
            
            current_processing = {
                'by': 'automated_pipeline@ucsd.edu',
                'date': datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
                'method': {
                    'task': 'PyramidGeneration',
                    'mips': mips,
                    'chunk_size': [int(x) for x in chunks],
                    'encoding': encoding
                }
            }
            prov['processing'].append(current_processing)

            prov.update({
                'description': "CH3_DIFF",
                'sources': {
                    'subject': self.animal,
                    'PRECOMPUTED_SETTINGS': {
                        'INPUT_DIR': str(INPUT_DIR),
                        'volume_size': [int(x) for x in image_manager.volume_size],
                        'dtype': str(image_manager.dtype),
                        'encoding': encoding,
                        'chunk_size': [int(x) for x in chunks]
                    }
                }
            })

            with open(prov_path, 'w') as f:
                json.dump(prov, f, indent=2, ensure_ascii=False)

            #MOVE PRECOMPUTED [ALL MIPS] FILES TO FINAL LOCATION
            # copy_with_rclone(temp_output_path, OUTPUT_DIR)
            copy_with_rclone(temp_output_path_pyramid, OUTPUT_DIR)
        
        #################################################
        # HISTOGRAM CREATION - SINGLE AND COMBINED
        #################################################
        OUTPUT_DIR = Path(self.fileLocationManager.histogram, ch_name)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f'Creating histogram for {OUTPUT_DIR} [IF NOT EXIST]')
        
        if not OUTPUT_DIR.exists() or not any(OUTPUT_DIR.iterdir()):
            file_keys = []
            for i, file in enumerate(image_files):
                filename = str(i).zfill(3) + ".tif"
                input_path = str(Path(INPUT_DIR, filename))
                mask_path = str(Path(self.fileLocationManager.masks, 'C1', 'thumbnail_masked', filename))
                output_path = str(Path(OUTPUT_DIR, f"{i}.png")) 

                if not output_path:
                    file_keys.append(
                        [input_path, mask_path, ch_name, file, output_path]
                    )
            # Process single histograms in parallel
            self.run_commands_concurrently(make_single_histogram_full, file_keys, workers)
            if not Path(OUTPUT_DIR, self.animal + '.png').exists():
                make_combined_histogram_full(image_manager, OUTPUT_DIR, self.animal, Path(mask_path).parent)

        #CLEAN UP TEMP FILES
        temp_folders = [progress_dir, temp_output_path, temp_output_path_pyramid]
        try:
            for folder in temp_folders:
                if self.debug:
                    print(f"DEBUG: Deleting temporary folder: {folder}")
                delete_in_background(folder)
        except Exception as e:
            print(f"Non-critical Error deleting progress directory: {e}")

        self.gen_ng_preview()

    def ng_prep_old(self):
            '''
            #WIP 19-AUG-2025
            CREATES PRECOMPUTED FORMAT FROM IMAGE STACK ON SCRATCH, MOVES TO 'CH3_DIFF'
            ALSO INCLUDES HISTOGRAM (SINGLE AND COMBINED)

            Note: called from regular pipeline: self.TASK_NG_PREVIEW
            or called from cell labeling pipeline: self.TASK_SEGMENT
            '''
            if self.debug:
                current_function_name = inspect.currentframe().f_code.co_name
                print(f"DEBUG: {self.__class__.__name__}::{current_function_name} Start")

            if self.task == 'segment': #REGENERATE ENTIRE NG FROM SCRATCH
                ch_name = 'DIFF' + '_' + self.set_id
                progress_dir = Path(self.fileLocationManager.neuroglancer_data, 'progress', ch_name)
                INPUT_DIR = Path(self.SCRATCH, 'pipeline_tmp', self.animal, ch_name)

                #MOVE src prep file [from scratch] to final location
                FINAL_PREP_DIR = Path(self.fileLocationManager.prep, ch_name)
                copy_with_rclone(INPUT_DIR, FINAL_PREP_DIR)
                
            else: #e.g. 'ng_preview':
                print('GENERATING NEUROGLANCER PREVIEW: CALLED FROM REGULAR PIPELINE')
                progress_dir = Path(self.fileLocationManager.neuroglancer_data, 'progress', 'NG_PREVIEW' + '_' + self.set_id)
                if self.channel:
                    print('GENERATING PREVIEW FOR CHANNEL:', self.channel)
                    ch_name = f"C{self.channel}"
                    if self.downsample:
                        INPUT_DIR = self.fileLocationManager.get_thumbnail_aligned(self.channel)
                    else:
                        INPUT_DIR = self.fileLocationManager.get_full_aligned(self.channel)
            
            temp_output_path = Path(self.SCRATCH, 'pipeline_tmp', self.animal, ch_name + '_ng')
            temp_output_path_pyramid = Path(self.SCRATCH, 'pipeline_tmp', self.animal, ch_name + '_py')
            temp_output_path.mkdir(parents=True, exist_ok=True)
            temp_output_path_pyramid.mkdir(parents=True, exist_ok=True)
            OUTPUT_DIR = Path(self.fileLocationManager.neuroglancer_data, ch_name)

            try:
                delete_in_background(progress_dir)
            except Exception as e:
                print(f"Non-critical Error deleting progress directory: {e}")
            progress_dir.mkdir(parents=True, exist_ok=True)    
            
            if self.debug:
                workers = 1    
            else:
                workers = self.get_nworkers()

            print('*'*50)
            print("\tINPUT_DIR:".ljust(20), f"{INPUT_DIR}".ljust(20))
            print("\tTEMP ng Output:".ljust(20), f"{temp_output_path}".ljust(20))
            print("\tTEMP DOWNSAMPLED PRECOMPUTED Output:".ljust(20), f"{temp_output_path_pyramid}".ljust(20))
            print("\tTEMP Progress:".ljust(20), f"{progress_dir}".ljust(20))
            print("\tFINAL Output:".ljust(20), f"{OUTPUT_DIR}".ljust(20))
            print("\tNEW NEUROGLANCER WILL BE CREATED [IF NOT EXIST]".ljust(20))
            print('*'*50)
            
            #pad before ImageManager?
            image_manager = ImageManager(INPUT_DIR)
            chunks = [image_manager.height//16, image_manager.width//16, 1]
            image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.tif')])
            z_length = len(os.listdir(self.fileLocationManager.section_web)) #SHOULD ALWAYS EXIST FOR ALL BRAIN STACKS

            if not OUTPUT_DIR.exists() or not any(OUTPUT_DIR.iterdir()):
                
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                # Update the volume size to include all sections (including blank spacers)
                # image_manager.volume_size = (
                #     image_manager.volume_size[0],  # x dimension (width)
                #     image_manager.volume_size[1],  # y dimension (height) 
                #     z_length                     # z dimension (now includes all sections)
                # )

                ################################################
                # PRECOMPUTED FORMAT CREATION 
                ################################################
                precompute = NgPrecomputedMaker(self.sqlController)
                scales = precompute.get_scales()
                ng = NumpyToNeuroglancer(
                    self.animal,
                    None,
                    scales,
                    layer_type="image", 
                    data_type=image_manager.dtype,
                    num_channels=1,
                    chunk_size=chunks
                )
                
                # Initialize precomputed volume
                ng.init_precomputed(temp_output_path, image_manager.volume_size)

                file_keys = []
                orientation = self.sqlController.histology.orientation
                is_blank = False

                # if self.task == 'segment' and len(image_files) < z_length: #PARTIAL PROCESS; ADD SPACERS (is_blank)
                #     if self.debug:
                #         print(f'DEBUG: Adding {z_length - len(image_files)} spacers (blank) for non-processed sections')
                #         print(f'\tPROCESSED SECTIONS: {len(image_files)}')
                #         print(f'\tTOTAL SECTIONS: {z_length}')
                #     section_files = {}
                #     for file_path in image_manager.files:
                #         # Extract the section number from filename (e.g., "050.tif" -> 50)
                #         filename = Path(file_path).name
                #         section_num = int(Path(filename).stem)  # Remove .tif extension and convert to int
                #         section_files[section_num] = file_path

                #     complete_files = []
                #     for section_num in range(z_length):  # Assuming sections are 0-indexed
                #         if section_num in section_files:
                #             complete_files.append(section_files[section_num])
                #         else:
                #             complete_files.append(None)  # Blank spacer
                    
                #     # Replace the original files list with the complete one
                #     image_manager.files = complete_files
                    
                for i, f in enumerate(image_manager.files):
                    if f is None:
                        is_blank = True
                        filepath = None
                    else:
                        is_blank = False
                        filepath = Path(INPUT_DIR, f)
                    file_keys.append([i, filepath, orientation, progress_dir, is_blank, image_manager.height, image_manager.width])

                if self.debug:
                    for file_key in file_keys:
                        ng.process_image(file_key)
                else:
                    self.run_commands_concurrently(ng.process_image, file_keys, workers)

                #####################################
                #PATCH ORG. provenance FILE
                # prov_path = Path(temp_output_path, 'provenance')
                
                # if provenance_path.exists():
                #     with open(provenance_path, 'r') as f:
                #         prov = self.clean_provenance(json.load(f))
                # else:
                # prov = {'description': ch_name, 'owners': [], 'processing': [], 'sources': {}}

                # current_processing = {
                #     'by': 'automated_pipeline@ucsd.edu',
                #     'date': datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
                #     'method': {
                #         'task': 'PyramidGeneration',
                #         'mips': '0',
                #         'chunk_size': str([int(x) for x in chunks]),
                #         'encoding': "raw"
                #     }
                # }
                # prov['processing'].append(current_processing)

                # prov.update({
                #     'description': ch_name,
                #     'sources': {
                #         'subject': self.animal,
                #         'PRECOMPUTED_SETTINGS': str({
                #             'INPUT_DIR': str(INPUT_DIR),
                #             'volume_size': str([int(x) for x in image_manager.volume_size]),
                #             'dtype': str(image_manager.dtype),
                #             'encoding': "raw",
                #             'chunk_size': str([int(x) for x in chunks])
                #         })
                #     }
                # })

                # with open(prov_path, 'w') as f:
                #     json.dump(prov, f, indent=2, ensure_ascii=False)
                # proper_provenance = {
                #     "processing": [
                #         {
                #             "processing": f"Created neuroglancer volume for {self.animal}",
                #             "method": "NumpyToNeuroglancer",
                #             "date": datetime.now().isoformat()
                #         }
                #     ]
                # }
                # with open(provenance_path, 'w') as f:
                #     json.dump(proper_provenance, f, indent=2)
                # print(f"Created proper provenance file at {provenance_path}")
                #####################################


                #################################################
                # PRECOMPUTED FORMAT PYRAMID (DOWNSAMPLED) START
                #################################################
                chunks = [XY_CHUNK, XY_CHUNK, Z_CHUNK]
                mips = 7
                encoding="raw"
                print('CREATING PYRAMID DOWNSAMPLED PRECOMPUTED FORMAT')

                # Use the same temp_output_path for both initial volume and pyramid
                cloudpath = f"file://{temp_output_path}" #full-resolution (already generated)
                # outpath = f"file://{temp_output_path}"
                outpath = f"file://{temp_output_path_pyramid}"
                
                # Create task queue
                tq = LocalTaskQueue(parallel=workers)
                # try:
                tasks = tc.create_image_shard_transfer_tasks(
                    cloudpath,
                    outpath, 
                    mip=0, 
                    chunk_size=chunks
                    # memory_target=memory_target,
                    # encoding=encoding,
                    # compress=False
                )
                tq.insert(tasks)
                tq.execute()
                # except Exception as e:
                #     print(f"Error in pyramid creation: {e}")
                    # print("Trying alternative pyramid creation method...")
                    
                    # # Alternative approach using downsample_scales
                    # from cloudvolume import CloudVolume
                    # from igneous import downsample_scales
                    
                    # # Load the source volume
                    # src_vol = CloudVolume(cloudpath, mip=0)
                    
                    # # Create destination volume with same info
                    # dest_vol = CloudVolume(cloudpath, 
                    #                     mip=0, 
                    #                     info=src_vol.info,
                    #                     compress=False)
                    # dest_vol.commit_info()
                    
                    # # Create downsampling tasks
                    # tasks = downsample_scales(dest_vol, 
                    #                         mip=0, 
                    #                         num_mips=mips, 
                    #                         chunk_size=chunks,
                    #                         compress=False)
                    
                    # tq.insert(tasks)
                    # tq.execute()

                # Create downsampling tasks for each mip level
                for mip in range(0, mips):
                    cv = CloudVolume(outpath, mip)
                    print(f'Creating sharded downsample tasks at mip={mip}')
                    tasks = tc.create_image_shard_downsample_tasks(
                        cv.layer_cloudpath,
                        mip=mip,
                        chunk_size=chunks
                        # memory_target=memory_target,
                        # encoding=encoding,
                        # factor=(2,2,1),  # Explicit downsampling factor
                        # sparse=False,
                        # preserve_chunk_size=True
                    )
                    tq.insert(tasks)
                    tq.execute()

                    cv = CloudVolume(outpath, mip=mip+1)
                    if not cv.info['scales'][mip+1]:  # Alternative way to check MIP existence
                        raise Exception(f"MIP {mip+1} not properly generated")

                print('Finished all downsampling tasks')
                #################################################
                # PRECOMPUTED FORMAT PYRAMID (DOWNSAMPLED) END
                #################################################
                cv = CloudVolume(cloudpath)
                cv.cache.flush()

                ################################################
                # MODIFY PROVENANCE FILE WITH META-DATA
                ################################################
                prov_path = Path(temp_output_path_pyramid, 'provenance')
                try:
                    if prov_path.exists():
                        with open(prov_path, 'r') as f:
                            prov = self.clean_provenance(json.load(f))
                    else:
                        prov = {'description': "", 'owners': [], 'processing': [], 'sources': {}}
                except json.JSONDecodeError:
                    prov = {'description': "", 'owners': [], 'processing': [], 'sources': {}}
                
                current_processing = {
                    'by': 'automated_pipeline@ucsd.edu',
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
                    'method': {
                        'task': 'PyramidGeneration',
                        'mips': mips,
                        'chunk_size': [int(x) for x in chunks],
                        'encoding': encoding
                    }
                }
                prov['processing'].append(current_processing)

                prov.update({
                    'description': "CH3_DIFF",
                    'sources': {
                        'subject': self.animal,
                        'PRECOMPUTED_SETTINGS': {
                            'INPUT_DIR': str(INPUT_DIR),
                            'volume_size': [int(x) for x in image_manager.volume_size],
                            'dtype': str(image_manager.dtype),
                            'encoding': encoding,
                            'chunk_size': [int(x) for x in chunks]
                        }
                    }
                })

                with open(prov_path, 'w') as f:
                    json.dump(prov, f, indent=2, ensure_ascii=False)

                #MOVE PRECOMPUTED [ALL MIPS] FILES TO FINAL LOCATION
                # copy_with_rclone(temp_output_path, OUTPUT_DIR)
                copy_with_rclone(temp_output_path_pyramid, OUTPUT_DIR)
            
            # #################################################
            # # HISTOGRAM CREATION - SINGLE AND COMBINED
            # #################################################
            OUTPUT_DIR = Path(self.fileLocationManager.histogram, ch_name)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            print(f'Creating histogram for {OUTPUT_DIR} [IF NOT EXIST]')
            
            if not OUTPUT_DIR.exists() or not any(OUTPUT_DIR.iterdir()):
                file_keys = []
                for i, file in enumerate(image_files):
                    filename = str(i).zfill(3) + ".tif"
                    input_path = str(Path(INPUT_DIR, filename))
                    mask_path = str(Path(self.fileLocationManager.masks, 'C1', 'thumbnail_masked', filename))
                    output_path = str(Path(OUTPUT_DIR, f"{i}.png")) 

                    if not output_path:
                        file_keys.append(
                            [input_path, mask_path, ch_name, file, output_path]
                        )
                # Process single histograms in parallel
                self.run_commands_concurrently(make_single_histogram_full, file_keys, workers)

            #TODO: keep? Song-Mao said overall histogram was not necessary: 19-AUG-2025
            #POSSIBLE INDIVIDUAL SECTION GENERATION IN PRIOR STEP
            # if not Path(OUTPUT_DIR, self.animal + '.png').exists():
            #     mask_path = self.fileLocationManager.get_thumbnail_masked(channel=1) #hard code this to channel 1
            #     if not os.path.exists(mask_path):
            #         print(f"Mask path does not exist {mask_path}; SKIPPING DIFF gen histogram creation")
            #     else:
            #         make_combined_histogram_full(image_manager, OUTPUT_DIR, self.animal, mask_path)

            #CLEAN UP TEMP FILES
            temp_folders = [progress_dir, temp_output_path, temp_output_path_pyramid]
            try:
                for folder in temp_folders:
                    if self.debug:
                        print(f"DEBUG: Deleting temporary folder: {folder}")
                    delete_in_background(folder)
            except Exception as e:
                print(f"Non-critical Error deleting progress directory: {e}")
            
            self.gen_ng_preview()


    def gen_ng_preview(self):
        '''AUTOMATED JSON STATE CREATION FROM AVAIALBLE NG RESOURCES:
           - Neuroglancer-compatible image layers
           - Annotations
           - Metadata
           - db additions to create unique id for actual preview and populating meta information

           -Note: if task==segment, assume desired brainsharer id will only contain that annotation set
        '''

        if self.debug:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} START")
            print(self.task)

        if (self.task == 'segment' or self.task == 'ng_preview') and self.downsample == False:
            self.downsample = False

            #FIND ANNOTATION SETS TO INCLUDE        
            annotations_dir = Path(self.fileLocationManager.neuroglancer_data, 'annotations')
            annotations_dir.mkdir(parents=True, exist_ok=True)

            pattern = re.compile(r'\.precomputed$')
            matching_ml_pos_folders = [
                f for f in annotations_dir.iterdir()
                if f.is_dir() and pattern.search(f.name)
            ]
            # print('INDEXING ANNOTATIONS IN PRECOMPUTED FOLDERS')
            # print([f.name for f in matching_ml_pos_folders])
        elif (self.task == 'segment' or self.task == 'ng_preview') and self.downsample == True:
            #NO ANNOTATIONS WILL EXIST FOR DOWNSAMPLED IMAGES
            matching_ml_pos_folders = []

        # GET CHANNEL NAMES FROM meta-data.json
        meta_data_file = 'meta-data.json'
        meta_store = os.path.join(self.fileLocationManager.prep, meta_data_file)
        with open(meta_store, 'r') as f:
            data = json.load(f)
        tracing_data = data.get("Neuroanatomical_tracing", {})
        
        try:
            if self.channel is not None and self.task != 'segment':
                #ONLY USE PROVIDED CHANNEL
                channel_names = [f"C{self.channel}"]
            else:
                channel_names = [info.get("channel_name") for info in tracing_data.values()]
        except Exception as e:
            print(f"Error determining channels: {e}")

        #IF AVAIABLE, GET XY RESOLUTION FROM meta-data.json, ELSE FROM DATABASE
        xy_resolution_unit = data.get("xy_resolution_unit", {})
        if not xy_resolution_unit:
            print('pulling resolution from DB')
            xy_resolution_unit = self.sqlController.scan_run.resolution
            x_resolution = xy_resolution_unit * 1e-6
            y_resolution  = xy_resolution_unit * 1e-6
        else:
            x_resolution = float(xy_resolution_unit[0])
            y_resolution = float(xy_resolution_unit[2])

        z_resolution = round(self.sqlController.scan_run.zresolution * 1e-6, 7)
        dimensions = {
            "x": [x_resolution, "m"],
            "y": [y_resolution, "m"],
            "z": [z_resolution, "m"]
        }
        
        pattern1 = re.compile(r'^C\d+$')  # starts with 'C' and then only digits
        pattern2 = re.compile(r'^C3_DIFF$')  # specific C3_DIFF pattern 
        if self.downsample == False:
            # CHECK IF NG-COMPATIBLE IMAGE LAYERS WERE GENERATED
            pattern3 = re.compile(r'.*_(re)?aligned$')
            # pattern2 = re.compile(r'.*\.zarr$')
        else:
            pattern3 = re.compile(r'^C1T_(?:aligned|realigned)$')

        pattern_rechunkme = re.compile(r'rechunkme')  
        ng_folders = [
            f for f in glob.glob(os.path.join(self.fileLocationManager.neuroglancer_data, '*'))
            if os.path.isdir(f) and
                not pattern_rechunkme.search(os.path.basename(f)) and 
                (pattern1.match(os.path.basename(f)) or
                # pattern2.match(os.path.basename(f)) or
                pattern2.match(os.path.basename(f)) or
                pattern3.match(os.path.basename(f)))
        ]
        
        #TODO clean up nested if/else
        ng_layers = []
        img_layers = {}
        segmentation_layers = {}
        for channel_name in channel_names:
            ome_zarr_path = os.path.join(self.fileLocationManager.neuroglancer_data, channel_name + ".zarr")
            if self.downsample:
                precomputed_path1 = os.path.join(self.fileLocationManager.neuroglancer_data, channel_name + 'T')
                precomputed_path2 = os.path.join(self.fileLocationManager.neuroglancer_data, channel_name + 'T_aligned')
            else:    
                precomputed_path1 = os.path.join(self.fileLocationManager.neuroglancer_data, channel_name)
                precomputed_path2 = os.path.join(self.fileLocationManager.neuroglancer_data, channel_name + '_aligned')
            if ome_zarr_path in ng_folders:
                folder_name = os.path.basename(ome_zarr_path)
                print(f"FOR {channel_name}, USING OME-ZARR DIR: {ome_zarr_path}")
                img_layers[channel_name] = {'src': ome_zarr_path, 'src_type': 'zarr2', 'folder_name': folder_name}
            elif precomputed_path1 in ng_folders:
                if self.downsample:
                    img_layers[channel_name + 'T'] = {'src': precomputed_path1, 'src_type': 'precomputed', 'folder_name': channel_name + 'T_aligned'}
                else:
                    img_layers[channel_name] = {'src': precomputed_path1, 'src_type': 'precomputed', 'folder_name': channel_name}
                print(f"USING IMG PRECOMPUTED DIR: {precomputed_path1}")
            elif precomputed_path2 in ng_folders:
                if self.downsample:
                    img_layers[channel_name + 'T'] = {'src': precomputed_path2, 'src_type': 'precomputed', 'folder_name': channel_name + 'T_aligned'}
                else:    
                    img_layers[channel_name] = {'src': precomputed_path2, 'src_type': 'precomputed', 'folder_name': channel_name}
                print(f"USING IMG PRECOMPUTED DIR: {precomputed_path2}")
            else:
                print(f"ERROR: NEUROGLANCER DATA NOT FOUND FOR {channel_name}")
                continue
            
            #FOR cell_manager USAGE, ONLY ADD C3_DIFF LAYER - rename
            if self.task == 'segment':
                ng_folders = [f for f in ng_folders if not os.path.basename(f) == 'C3_DIFF.zarr']
                c3_diff_dir = [f for f in ng_folders if os.path.basename(f) == 'C3_DIFF']
                if c3_diff_dir:
                    print(f' FOLDERS: {ng_folders}')
                c3_path = os.path.join(self.fileLocationManager.neuroglancer_data, 'C3_DIFF')
                img_layers['C3_DIFF'] = {'src': c3_path, 'src_type': 'precomputed', 'folder_name': 'C3_DIFF'}
        
        ##################################################################################
        #define initial view field (x,y,z) - center on image and across stack
        if self.downsample:
            check_dir = self.fileLocationManager.get_thumbnail()
        else:
            check_dir = self.fileLocationManager.get_full()
        _, nfiles, max_width, max_height = test_dir(self.animal, check_dir, section_count=0)
        view_field = {"position": [max_width//2, max_height//2, nfiles//2]}

        ##################################################################################

        print(f'GENERATING NG PREVIEW FOR {self.animal} WITH LAYERS: {list(img_layers.keys())}')
        
        ##################################################################################
        #COMPILE IMG SRC FOR LAYERS
        base_url = f"https://imageserv.dk.ucsd.edu/data/{self.animal}/neuroglancer_data/"
        
        if self.downsample:
            desired_order = ['C1T', 'C2T', 'C3T']
        else:
            desired_order = ['C1', 'C2', 'C3_DIFF', 'ML_POS']
        ordered_dict = {key: img_layers[key] for key in desired_order if key in img_layers}
        # print(f'{ordered_dict=}')
        # print(f'{ordered_dict.keys()}')
        channel_numbers = []
        for key in ordered_dict.keys():
            if key.startswith('C') and not key.endswith('_DIFF'):
                # Extract digits after 'C' (split at '_' if present)
                num_part = key[1:].split('_')[0]  # Takes '3' from 'C3_DIFF'
                if num_part.isdigit():  # Ensure it's a valid number
                    channel_numbers.append(int(num_part))
        if channel_numbers:
            largest_number = max(channel_numbers)
            diff_channel_name = f'C{largest_number}_DIFF'
            print(f"Largest channel number: {largest_number}")
        else:
            print("No valid channel numbers found.")
        
        for channel_name, channel_attributes in ordered_dict.items():
            if channel_name.endswith('_DIFF'):
                channel_name = diff_channel_name
            layer = {
                "type": "image",
                "source": f"{channel_attributes['src_type']}://{base_url}{channel_attributes['folder_name']}",
                "tab": "rendering",
                "name": channel_name
            }
            ng_layers.append(layer)
      
        if (self.task == 'segment' or self.task == 'ng_preview') and self.downsample == False: #ADD ANNOTATION LAYERS LAST
            for folder in matching_ml_pos_folders:
                folder_name = folder.name
                if folder_name.endswith('.precomputed'):
                    annotation_name = folder_name.replace('ML_', '').replace('.precomputed', '') + '_candidates'
                    src = 'precomputed://' + base_url + 'annotations' + '/' + str(folder_name)
                    seg_layer = {
                        'source': str(src),
                        'type': 'segmentation',
                        "tab": "source",
                        "segments": [],
                        'name': str(annotation_name)
                    }
                    #ONLY ADD ANNOTATION LAYER OF SPECIFIC set_id
                    if self.set_id in folder_name:
                        ng_layers.append(seg_layer)
                    
        dimensions_json = {"dimensions": dimensions}
        layers_json = {"layers": ng_layers}

        ##################################################################################
        #not sure what this is (but was in template)
        other_stuff = {"crossSectionScale": 121.51041751873494,
                "crossSectionDepth": 46.15384615384615,
                "projectionScale": 131072}

        #add annotation layers
        #TODO: dimensions from primary dimensions

        #see cell_manager.py (extract_predictions) for reverse engineering  [csv -> annotation format]

        annotations_json = {
            "type": "annotation",
            "source": {
                "url": "local://annotations",
                "transform": {
                "outputDimensions": {
                    "x": [
                    x_resolution,
                    "m"
                    ],
                    "y": [
                    y_resolution,
                    "m"
                    ],
                    "z": [
                    z_resolution,
                    "m"
                    ]
                }
                }
            },
            "tab": "source",
            "annotations": [],
            "annotationProperties": [
                {
                "id": "color",
                "description": "color",
                "type": "rgb",
                "default": "#ffff00"
                },
                {
                "id": "visibility",
                "description": "visibility",
                "type": "float32",
                "default": 1,
                "min": 0,
                "max": 1,
                "step": 1
                },
                {
                "id": "opacity",
                "description": "opacity",
                "type": "float32",
                "default": 1,
                "min": 0,
                "max": 1,
                "step": 0.01
                },
                {
                "id": "point_size",
                "description": "point marker size",
                "type": "float32",
                "default": 5,
                "min": 0,
                "max": 10,
                "step": 0.01
                },
                {
                "id": "point_border_width",
                "description": "point marker border width",
                "type": "float32",
                "default": 3,
                "min": 0,
                "max": 5,
                "step": 0.01
                },
                {
                "id": "line_width",
                "description": "line width",
                "type": "float32",
                "default": 1,
                "min": 0,
                "max": 5,
                "step": 0.01
                }
            ],
            "name": "annotation"
            }


        #general settings **TODO: needs edit
        gen_settings = {
            "layout": "4panel",
            "helpPanel": {
                "row": 2
            },
            "settingsPanel": {
                "row": 3
            },
            "userSidePanel": {
                "tab": "User",
                "location": {
                "row": 1
                }
            }}

        combined_json = {**dimensions_json, **view_field, **other_stuff, **layers_json, **annotations_json, **gen_settings}

        combined_json_str = json.dumps(combined_json)
        comments = self.animal + ' auto preview'
        # POPULATE description with processing parameters
        
        
        if self.task == 'segment':
            params = 'parameters.json'
            param_store = Path(self.cell_label_path, params)
            try:
                with open(param_store, 'r') as f:
                    param_data = json.load(f)

                description = str(param_data)
            except:
                description = "No parameters.json file found in cell_labels folder"
        else:
            description = "No parameters.json file found in cell_labels folder"
        active_query = self.sqlController.insert_ng_state(combined_json_str, fk_prep_id=self.animal, comments=comments, description=description, readonly=True, public=False, active=True)
        
        if active_query:
            # print(f"preview state created: {active_query}")
            print(f"\nhttps://brainsharer.org/ng/?id={active_query.id}\n")
        else:
            print("error; no preview state created")

        #TODO: add more targeted link to only expose channels of interest on imageserv
        target_path = str(self.fileLocationManager.www)
        link_path = str(Path('/', 'srv', self.animal))
        self.create_symbolic_link(target_path, link_path)


    def clean_provenance(self, prov_data):
        """Clean and normalize provenance data"""
        # Convert byte strings to regular strings
        if 'processing' in prov_data:
            for entry in prov_data['processing']:
                if isinstance(entry['by'], bytes):
                    entry['by'] = entry['by'].decode('utf-8')
        
        # Remove duplicate processing entries
        if 'processing' in prov_data:
            unique_entries = []
            seen = set()
            for entry in prov_data['processing']:
                entry_str = json.dumps(entry, sort_keys=True)
                if entry_str not in seen:
                    seen.add(entry_str)
                    unique_entries.append(entry)
            prov_data['processing'] = unique_entries
        
        return prov_data
    

    def create_section_mapping(self, image_files):
        """Create a mapping of section numbers to file indices"""
        section_mapping = {}
        
        for i, filename in enumerate(image_files):
            # Extract section number from filename - adjust this pattern based on your naming convention
            # Example: if files are named "section_001.tif", "section_002.tif", etc.
            match = re.search(r'section_(\d+)', filename)
            if match:
                section_num = int(match.group(1))
                section_mapping[section_num] = i
            else:
                # Fallback: assume files are in order starting from section 0
                section_mapping[i] = i
        
        return section_mapping
    

    #TODO: possibly delete
    def omezarr(self):
        if self.debug:
            # current_function_name = inspect.currentframe().f_code.co_name
            # print(f"DEBUG: {self.__class__.__name__}::{current_function_name} START")
            workers = 1    
        else:
            workers = self.get_nworkers()

        INPUT_DIR = Path(self.fileLocationManager.prep, 'CH3_DIFF')
        temp_output_path = Path(SCRATCH, 'pipeline_tmp', self.animal, 'ch3_diff_tmp_zarr')
        temp_output_path.mkdir(parents=True, exist_ok=True)
        progress_dir = Path(SCRATCH, 'pipeline_tmp', self.animal, 'ch3_diff_progress')
        progress_dir.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR = Path(self.fileLocationManager.neuroglancer_data, 'C3_DIFF')
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        use_scratch = use_scratch_dir(INPUT_DIR)
        if use_scratch:
            self.scratch_space = os.path.join(get_scratch_dir(), 'pipeline_tmp', self.animal, 'dask-scratch-space')
            if os.path.exists(self.scratch_space):
                delete_in_background(self.scratch_space)
            os.makedirs(self.scratch_space, exist_ok=True)

        print(f'{INPUT_DIR=}')
        # print(f'TEMP Output: {temp_output_path}')
        print(f'FINAL Output: {OUTPUT_DIR}')

        files = []
        for file in sorted(os.listdir(INPUT_DIR)):
            filepath = os.path.join(INPUT_DIR, file)
            files.append(filepath)

        image_manager = ImageManager(INPUT_DIR)
        target_chunk_size = 3000
        chunk_y = closest_divisors_to_target(image_manager.height, target_chunk_size)
        chunk_x = closest_divisors_to_target(image_manager.width, target_chunk_size)
        concurrent_slice_processing = 2 # Number of slices to process in parallel (but requires more RAM)
        originalChunkSize = [concurrent_slice_processing, image_manager.num_channels, concurrent_slice_processing, chunk_y, chunk_x] # 1796x984

        xy_resolution = self.sqlController.scan_run.resolution
        z_resolution = self.sqlController.scan_run.zresolution

        print(f'Creating OME-Zarr from {INPUT_DIR}')
        storefile = f'CH3_DIFF.zarr'
        scaling_factor = 1
        
        mips = 7
        
        if self.debug:
            print(f'INPUT FOLDER: {INPUT_DIR}')
            print(f'INPUT FILES COUNT: {len(files)}')

        omero = {}
        omero['channels'] = {}
        omero['channels']['color'] = None
        omero['channels']['label'] = None
        omero['channels']['window'] = None
        omero['name'] = self.animal
        omero['rdefs'] = {}
        omero['rdefs']['defaultZ'] = len(files) // 2
        omero_dict = omero

        # storepath = Path(self.fileLocationManager.www, "neuroglancer_data", storefile)
        xy = xy_resolution * scaling_factor
        resolution = (z_resolution, xy, xy)

        omezarr = builder(
            INPUT_DIR,
            str(temp_output_path),
            files,
            resolution,
            originalChunkSize=originalChunkSize,
            tmp_dir=str(self.scratch_space),
            debug=self.debug,
            omero_dict=omero_dict,
            mips=mips,  # Limit pyramid levels to reduce memory
            available_memory=self.available_memory * 0.9  # Leave 10% headroom
        )
        dask.config.set({
            'temporary_directory': str(self.scratch_space),
            'distributed.worker.memory.target': 0.8,  # Target fraction to stay under
            'distributed.worker.memory.spill': 0.9,   # Spill to disk at 90%
            'distributed.worker.memory.pause': 0.95,   # Pause at 95%
            'distributed.worker.memory.terminate': 0.98  # Terminate at 98%
        })
        mem_per_worker = max(2, round((self.available_memory * 0.9) / workers)) # Minimum 2GB per worker
        print(f'Starting omezarr with {omezarr.workers} workers and {omezarr.sim_jobs} sim_jobs with free memory/worker={mem_per_worker}GB')

        mem_per_worker = str(mem_per_worker) + 'GB'
        cluster = LocalCluster(
            n_workers=workers,
            threads_per_worker=1, 
            memory_limit=mem_per_worker,
            local_directory=str(self.scratch_space),
            processes=True # Use processes instead of threads for memory isolation
        )

        with Client(cluster) as client:
            omezarr.write_resolution_0(client)
            for mip in range(1, len(omezarr.pyramidMap)):
                omezarr.write_mips(mip, client)                    

        cluster.close()

        print("Processing complete - transferring to network storage")
        copy_with_rclone(temp_output_path, OUTPUT_DIR)
        omezarr.cleanup()