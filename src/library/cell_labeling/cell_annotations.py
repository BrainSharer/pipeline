import os, sys, glob, json
import re
import inspect
from pathlib import Path
import numpy as np
import polars as pl
from tqdm import tqdm
from library.utilities.utilities_process import M_UM_SCALE, SCALING_FACTOR, delete_in_background

from cloudvolume import CloudVolume
from taskqueue import LocalTaskQueue
import igneous.task_creation as tc


class CellAnnotations():

    def create_annotations(self, task: str = None, meta_data_info: dict = None):
        '''Create annotations for for cell candidates
        Output will be pyramid precomputed format suitable for loading into brainsharer/ng'''

            #final naming scheme: ML_{channel_with_label_of_interest}.precomputed

        if self.debug:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} Start")
            print(f"DEBUG: create annotations set in precomputed format")

        if task:
            print(self.TASK_EXTRACT)
            #GET label_of_interest CHANNEL
            for tracer_id, tracer_data in meta_data_info['Neuroanatomical_tracing'].items():
                if tracer_data['mode'] == 'dye' or tracer_data['mode'] == 'ntb' or tracer_data['mode'] == 'counterstain' or tracer_data['mode'] == 'C1':
                    counterstain_channel = tracer_data['channel_name']
                elif tracer_data['mode'] == 'virus' or tracer_data['mode'] == 'ctb' or tracer_data['mode'] == 'C3':
                    label_of_interest_channel = tracer_data['channel_name']

        labels = ['ML_' + label_of_interest_channel]
        sampling = self.sampling
        
        performance_lab = self.sqlController.histology.FK_lab_id
        xy_resolution = self.sqlController.scan_run.resolution
        z_resolution = self.sqlController.scan_run.zresolution
        w = int(self.sqlController.scan_run.width//SCALING_FACTOR)
        h = int(self.sqlController.scan_run.height//SCALING_FACTOR)
        z_length = len(os.listdir(self.fileLocationManager.section_web)) #SHOULD ALWAYS EXIST FOR ALL BRAIN STACKS

        print(f'DEBUG: {xy_resolution=}, {z_resolution=}, {w=}, {h=}, {z_length=}')
        #READ, CONSOLIDATE PREDICTION FILES, SAMPLE AS NECESSARY
        
        if os.path.exists(self.cell_label_path):
            print(f'Parsing cell labels from {self.cell_label_path}')
        else:
            print(f'ERROR: {self.cell_label_path} not found')
            sys.exit(1)
        
        detection_files = sorted(glob.glob(os.path.join(self.cell_label_path, f'detections_*.csv') ))
        if len(detection_files) == 0:
            print(f'Error: no csv files found in {self.cell_label_path}')
            sys.exit(1)

        #NAMING CONVENTION FOR ANNOTATION FOLDERS
        # e.g. ML_POS_0, ML_POS_1, ML_POS_2
        annotations_dir = Path(self.fileLocationManager.neuroglancer_data, 'annotations') #ref 'drawn_directory
        annotations_dir.mkdir(parents=True, exist_ok=True)
        prefix = labels[0]

        # Regex pattern to match folder names starting with prefix and ending with _{int}, e.g. ML_POS_0, ML_POS_12
        pattern_template = r'^{}(?:_(\d+))?$'  # group 1 captures optional suffix number
        pattern = re.compile(pattern_template.format(re.escape(prefix)))

        max_suffix = -1
        found_any = False

        for folder in annotations_dir.iterdir():
            if folder.is_dir():
                m = pattern.match(folder.name)
                if m:
                    found_any = True
                    if m.group(1) is None:
                        # Folder has prefix but no number suffix
                        max_suffix = max(max_suffix, -1)
                    else:
                        num = int(m.group(1))
                        if num > max_suffix:
                            max_suffix = num

        if not found_any:
            # Directory empty of matching prefix folders
            ann_out_folder_name = prefix
        else:
            # We found matching folders, get next suffix number
            next_number = max_suffix + 1
            ann_out_folder_name = f"{prefix}_{next_number}"

        print(f"New annotations output: {ann_out_folder_name}")

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
                    (pl.col("section") + 0.5).cast(pl.Int32).alias("section") #annotations were offset by 1 (perhaps due to round down)
                    
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
        # if not df.is_empty():
        #     df.write_csv(dfpath)

        # # SAVE JSON FORMAT *SAMPLED DATA
        # annotations_file = str(Path(annotations_dir, labels[0]+'.json'))

        # # Populate points variable after sampling
        # sampled_points = (
        #     df.sort("section")
        #     .select(["x", "y", "section"])
        #     .to_numpy()
        #     .tolist()
        # )
        # with open(annotations_file, 'w') as fh:
        #     json.dump(sampled_points, fh)

        # SAVE PRECOMPUTED [SEGMENTATION] FORMAT *SAMPLED DATA
        shape = (w, h, z_length)
        volume = np.zeros(shape, dtype=np.uint32) #uint32 supports 4.2 billion IDs
        desc = f"Drawing on {self.animal} volume={volume.shape}"
        for row in tqdm(df.iter_rows(named=True), desc=desc):
            x = int(row["x"] // SCALING_FACTOR)
            y = int(row["y"] // SCALING_FACTOR)
            z = int(row["section"])
            
            # Verify coordinates are within bounds
            if 0 <= x < w and 0 <= y < h and 0 <= z < z_length:
                volume[x,y,z] = 1
                # used_ids.add(current_id)
                # current_id += 1
            else:
                print(f"Out of bounds: ({x}, {y}, {z})")

            # Assign unique IDs to each point
            # if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
            #     volume[x, y, z] = current_id  
            #     current_id += 1  # Increment ID for the next point

            #single point annotation
            # if 0 <= x < w and 0 <= y < h and 0 <= z < z_length:
            #     volume[z][y, x] = 1 # Set a single pixel to label value 1


                # Draw a small circle to make the point more visible
                #cv2.circle(volume[z], center=(x, y), radius=1, color=1, thickness=-1)  # label = 1

        out_dir = Path(annotations_dir, ann_out_folder_name + '.precomputed')
        if os.path.exists(out_dir):
            print(f'Removing existing directory {out_dir}')
            delete_in_background(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        
        print(f'Creating precomputed annotations in {out_dir}')
        resolution = int(xy_resolution * 1000 * SCALING_FACTOR)
        scales = (resolution, resolution, int(z_resolution * 1000))
        preferred_chunk_size = (128, 128, 64)
        adjusted_chunk_size = tuple(min(p, s) for p, s in zip(preferred_chunk_size, shape))

        info = CloudVolume.create_new_info(
            num_channels=1,
            layer_type='segmentation', 
            data_type='uint32',
            encoding='raw',
            resolution=scales,
            voxel_offset=(0, 0, 0),
            chunk_size=adjusted_chunk_size,
            volume_size=shape,  # x, y, z
        )

        vol = CloudVolume(f'file://{out_dir}', info=info)
        vol.commit_info()
        vol[:, :, :] = volume

        tq = LocalTaskQueue(parallel=1)
        tasks = tc.create_downsampling_tasks(f'file://{out_dir}', 
                                            mip=0, 
                                            num_mips=3, 
                                            compress=False,
                                            #  encoding='compressed_segmentation', #preserve segmentation IDs
                                            #  sparse=True,
                                            fill_missing=True,
                                            delete_black_uploads=True
                                            )
        tq.insert(tasks)
        tq.execute()

        # MODIFY PROVENANCE FILE WITH META-DATA
        prov_path = Path(out_dir, 'provenance')
        with open(prov_path, 'r') as f:
            prov = json.load(f)
        if self.sampling:
            prov['description'] = f"SEGMENTATION VOL OF ANNOTATIONS - SAMPLING ENABLED (n={self.sampling} of {df.height} TOTAL)"
        else:
            prov['description'] = f"SEGMENTATION VOL OF ANNOTATIONS - (n={df.height})"
        subject = self.animal
        prov['sources'] = {
            "subject": subject, 
            "ML_segmentation": {
                "min_segment_size": self.segment_size_min,
                "max_segment_size": self.segment_size_max,
                "gaussian_blur_standard_deviation_sigmaX": self.gaussian_blur_standard_deviation_sigmaX,
                "gaussian_blur_kernel_size_pixels": self.gaussian_blur_kernel_size_pixels,
                "segmentation_threshold": self.segmentation_threshold,
                "cell_radius": self.cell_radius
            }
        }

        prov['owners'] = [f'PERFORMANCE LAB: {performance_lab}']
        with open(prov_path, 'w') as f:
            json.dump(prov, f, indent=2)