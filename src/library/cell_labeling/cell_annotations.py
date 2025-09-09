import os, sys, glob, json, time
import re
from datetime import datetime
import inspect
from pathlib import Path
import numpy as np
import polars as pl
from tqdm import tqdm
from library.utilities.utilities_process import M_UM_SCALE, SCALING_FACTOR, delete_in_background
from library.image_manipulation.precomputed_manager import NgPrecomputedMaker
from cloudvolume import CloudVolume
from cloudvolume.lib import mkdir, touch
from taskqueue import LocalTaskQueue
import igneous.task_creation as tc
from library.utilities.cell_utilities import (
    copy_with_rclone
)
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import uuid
from cloudvolume.lib import Vec
import cv2

class CellAnnotations():


    def extract_point_annotations(self, task: str = None, meta_data_info: dict = None, segment_ch: int = None):
        '''
        Part of create_annotations method - extracts point annotations from csv files
        '''
        if self.debug:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} Start")

        print(self.TASK_EXTRACT)

        if meta_data_info is None:    
            meta_data_file = 'meta-data.json'
            meta_store = os.path.join(self.fileLocationManager.prep, meta_data_file)
            meta_data_info = {}
            if os.path.isfile(meta_store):
                print(f'Found neuroanatomical tracing info; reading from {meta_store}')
                with open(meta_store) as fp:
                    info = json.load(fp)
                self.meta_channel_mapping = info['Neuroanatomical_tracing']
                meta_data_info['Neuroanatomical_tracing'] = self.meta_channel_mapping

        counterstain_modes = ['counterstain', 'dye', 'ntb', 'C1']
        label_of_interest_modes = ['label_of_interest', 'virus', 'ctb', 'C3']
        for tracer_id, tracer_data in meta_data_info['Neuroanatomical_tracing'].items():
            if tracer_data['mode'] in counterstain_modes:
                counterstain_channel = tracer_data['channel_name']
            elif tracer_data['mode'] in label_of_interest_modes:
                label_of_interest_channel = tracer_data['channel_name']

        if segment_ch:
            print(f"DEBUG: override defaults; create annotations set for channel {segment_ch}")
            label_of_interest_channel = f'C{segment_ch}'

        sampling = self.sampling
        
        xy_resolution = self.sqlController.scan_run.resolution
        z_resolution = self.sqlController.scan_run.zresolution

        #READ, CONSOLIDATE PREDICTION FILES, SAMPLE AS NECESSARY
        if os.path.exists(self.cell_label_path):
            print(f'Parsing cell labels from {self.cell_label_path}')
        else:
            print(f'ERROR: {self.cell_label_path} not found')
            sys.exit(1)

        #ACTUAL FILE DIMENSIONS TAKE PRECEDENCE OVER DATABASE
        param_file = Path(self.cell_label_path, 'parameters.json')
        with open(param_file, 'r') as f:
            params = json.load(f)
        expected_x, expected_y, z_length = params["vol_shape_x_y_z"]
        
        detection_files = sorted(glob.glob(os.path.join(self.cell_label_path, f'detections_*.csv') ))
        if len(detection_files) == 0:
            print(f'Error: no csv files found in {self.cell_label_path}')
            sys.exit(1)
        else:
            print(f'Parsing {len(detection_files)} sections...')

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

        ###############################################
        #ALL ANNOTATION POINTS ARE STORED IN df VARIABLE AT THIS POINT
        ###############################################
        return (df, counterstain_channel, label_of_interest_channel, expected_x, expected_y, z_length)
     

    def create_annotation_layer(self, df_points: pl.DataFrame, expected_shape: tuple, scratch_tmp: str, counterstain_channel: str, label_of_interest_channel: str):
        '''Create and save annotation data in Neuroglancer precomputed format from a Polars DataFrame [for loading in Neuroglancer]

        1. cleans temp directories (progress_dir, temp_output_path, temp_output_path_pyramid)
        2. cleans final output dir (OUTPUT_DIR)
        3. recreates temp and final directories
        4. generates precomputed format from annotations' csv files @ temp locations
        5. moves final MIP precomputed to final dest. folder
        '''

        if self.debug:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} Start")
            print("Populating full resolution volume with vectorized operations...")
        
        # SAVE PRECOMPUTED [SEGMENTATION] FORMAT *SAMPLED DATA
        src_dtype = 'uint8'
        volume = np.zeros(expected_shape, dtype=src_dtype)
        
        # Filter valid coordinates first
        valid_df = df_points.filter(
            (pl.col("x") >= 0) & (pl.col("x") < expected_shape[0]) &
            (pl.col("y") >= 0) & (pl.col("y") < expected_shape[1]) &
            (pl.col("section") >= 0) & (pl.col("section") < expected_shape[2])
        )
        invalid_count = len(df_points) - len(valid_df)
        if invalid_count > 0:
            print(f"Skipping {invalid_count} out-of-bounds points")

        if self.debug:
            print(f'{expected_shape=}, {volume.shape=}')

        #NAMING CONVENTION FOR ANNOTATION FOLDERS
        labels = ['ann_' + self.set_id + '_' + label_of_interest_channel]
        prefix = labels[0]
        ann_out_folder_name = 'ann_' + self.set_id + '_' + label_of_interest_channel
        OUTPUT_DIR = Path(self.fileLocationManager.neuroglancer_data, 'annotations', ann_out_folder_name + '.ann_pre') 

        #TODO: Needs re-worked for retraining models
        # Regex pattern to match folder names starting with prefix and ending with _{int}, e.g. ML_POS_0, ML_POS_12
        found_any = False
        max_suffix = -1
        if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
            pattern_template = r'^{}(?:_(\d+))?$'  # group 1 captures optional suffix number
            pattern = re.compile(pattern_template.format(re.escape(prefix)))
            for folder in OUTPUT_DIR.iterdir():
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

        progress_dir = Path(self.fileLocationManager.neuroglancer_data, 'progress', ann_out_folder_name)
        temp_output_path = Path(self.SCRATCH, 'pipeline_tmp', self.animal, ann_out_folder_name + '_ng')
        xy_resolution = self.sqlController.scan_run.resolution
        z_resolution = self.sqlController.scan_run.zresolution
        preferred_chunk_size = (128, 128, 64)
        adjusted_chunk_size = tuple(min(p, s) for p, s in zip(preferred_chunk_size, expected_shape))

        print('*'*50)
        print(f'DEBUG: {xy_resolution=}, {z_resolution=}, {expected_shape=}')
        print("\tTEMP output:".ljust(20), f"{temp_output_path}".ljust(20))
        print("\tTEMP Progress:".ljust(20), f"{progress_dir}".ljust(20))
        print("\tFINAL Output:".ljust(20), f"{OUTPUT_DIR}".ljust(20))

        #duane test
        csv_src = Path(self.fileLocationManager.neuroglancer_data, 'annotations', 'ann_cf849f697b4b4560accc29255c94252e_C3.csv')
        points = np.genfromtxt(csv_src, delimiter=',', skip_header=1)

        # Define the dataset resolution (voxel size in nm, e.g., [8, 8, 8] for 8nm x 8nm x 8nm)
        # Convert to nm
        xy_resolution_nm = xy_resolution * 1000
        z_resolution_nm = z_resolution * 1000   
        resolution = Vec(xy_resolution_nm, xy_resolution_nm, z_resolution_nm)
        points_nm = points * np.array([xy_resolution_nm, xy_resolution_nm, z_resolution_nm])

        # Specify the output path for the precomputed layer (local or cloud)
        # output_path = 'file:///path/to/precomputed/annotations'  # Local path
        # or output_path = 's3://bucket-name/annotations' for cloud storage
 
        # Create a CloudVolume instance for the annotation layer
        vol = CloudVolume(
            f'file://{str(OUTPUT_DIR)}',
            info={
                "@type": "neuroglancer_annotations_v1",
                "dimensions": {
                    "x": [resolution.x, "nm"],
                    "y": [resolution.y, "nm"],
                    "z": [resolution.z, "nm"]
                },
                "annotation_type": "point",
                "spatial_index": True,  # Enables efficient querying
                "scales": [{
                    "resolution": [resolution.x, resolution.y, resolution.z],
                    "size": [57600, 32000, 467],  # Match expected_shape
                    "voxel_offset": [0, 0, 0],      # Assuming origin at (0,0,0)
                    "key": f"{int(resolution.x)}_{int(resolution.y)}_{int(resolution.z)}",  # Unique key for the scale
                    "encoding": "raw"  # Required for annotation layers
                }]
            }
        )
        vol.commit_info()

        # Add annotations
        annotations = [
            {
                "point": point.tolist(),  # [x, y, z] in nm
                "id": str(i),             # Unique ID for each annotation
                "type": "point"
            } for i, point in enumerate(points_nm)
        ]
        

        # Write annotations (handle potential version differences)
        try:
            vol.add_annotations(annotations)
        except AttributeError:
            # Fallback for older cloudvolume versions
            print("add_annotations not available, using manual annotation writing")
            annotation_data = {
                "@type": "neuroglancer_annotations_v1",
                "annotations": annotations
            }
            annotation_file = OUTPUT_DIR / "annotations"
            annotation_file.parent.mkdir(parents=True, exist_ok=True)
            with open(annotation_file, 'w') as f:
                json.dump(annotation_data, f)
            # Update spatial index manually if needed
            vol.commit_spatial_index()

        print(f"Annotations written to {OUTPUT_DIR}")








        ##ED code

        # Create directory structure
        # spatial_dir = Path(OUTPUT_DIR, 'predictions0', 'spatial0')
        # spatial_dir.mkdir(parents=True, exist_ok=True)
        # info_dir = Path(OUTPUT_DIR, 'predictions0')
        # info_dir.mkdir(parents=True, exist_ok=True)


        # point_filename = Path(spatial_dir, '0_0_0')  # Annotations file
        # info_filename = Path(info_dir, 'info')      # Info file

        # # Validate required columns
        # required_cols = ['x', 'y', 'section']
        # if not all(col in valid_df.columns for col in required_cols):
        #     raise ValueError("DataFrame must contain 'x', 'y', and 'section' columns")

        # # Validate numeric columns and handle nulls
        # for col in required_cols:
        #     if not valid_df[col].dtype.is_numeric():
        #         raise ValueError(f"Column '{col}' must be numeric, got {valid_df[col].dtype}")
        #     if valid_df[col].is_null().any():
        #         valid_df = valid_df.with_columns(pl.col(col).fill_null(0))

        # # Create annotations
        # annotations = []
        # for i, row in enumerate(valid_df.iter_rows(named=True)):
        #     annotation = {
        #         "point": [float(row['x']), float(row['y']), float(row['section'])],
        #         "type": "point",
        #         "annotation_type": "point",
        #         "id": str(i),
        #         "description": f"Point {i}"
        #     }
            
        #     # Add additional properties from DataFrame columns
        #     for col_name in valid_df.columns:
        #         if col_name not in ['x', 'y', 'section']:
        #             annotation[col_name] = row[col_name]
            
        #     annotations.append(annotation)

        # # Save annotations JSON
        # with open(point_filename, 'w') as outfile:
        #     json.dump({
        #         "annotations": annotations,
        #         "relationships": []
        #     }, outfile, indent=2)

        # # Create info structure
        # info = {
        #     "@type": "neuroglancer_annotations_v1",
        #     "annotation_type": "POINT",
        #     "by_id": {"key": "by_id"},
        #     "dimensions": {
        #         "x": [float(xy_resolution * 1e-6), "m"],
        #         "y": [float(xy_resolution * 1e-6), "m"],
        #         "z": [float(z_resolution * 1e-6), "m"]
        #     },
        #     "lower_bound": [0, 0, 0],
        #     "upper_bound": [expected_shape[0], expected_shape[1], expected_shape[2]],
        #     "properties": [],
        #     "relationships": [],
        #     "spatial": [{
        #         "key": "spatial0",
        #         "chunk_size": [int(x) for x in adjusted_chunk_size],
        #         "grid_shape": [1, 1, 1]
        #     }]
        # }

        # # Save info JSON
        # with open(info_filename, 'w') as infofile:
        #     json.dump(info, infofile, indent=2)

        # print(f"Saved {len(annotations)} JSON annotations to {point_filename}")
        # print(f"Saved info file to {info_filename}")

        # point_filename = Path(spatial_dir, '0_0_0.gz')
        # info_filename = Path(info_dir, 'info')

        # required_cols = ['x', 'y', 'section']
        # if not all(col in valid_df.columns for col in required_cols):
        #     raise ValueError("DataFrame must contain 'x', 'y', and 'section' columns")
        
        # for col in required_cols:
        #         if not valid_df[col].dtype.is_numeric():
        #             raise ValueError(f"Column '{col}' must be numeric, got {valid_df[col].dtype}")
        #         if valid_df[col].is_null().any():
        #             valid_df = valid_df.with_columns(pl.col(col).fill_null(0))

        # points = valid_df.select(["x", "y", "section"]).to_numpy()
        # # Write to binary file
        # with open(point_filename, 'wb') as outfile:
        #     # Write number of points (64-bit unsigned integer)
        #     buf = struct.pack('<Q', len(points))
            
        #     # Write points (three 32-bit floats per point)
        #     pt_buf = b''.join(struct.pack('<3f', float(x), float(y), float(z)) for x, y, z in points)
        #     buf += pt_buf
            
        #     # Write IDs (64-bit unsigned integers)
        #     id_buf = struct.pack('<%sQ' % len(points), *range(len(points)))
        #     buf += id_buf
            
        #     # Compress and write
        #     bufout = gzip.compress(buf)
        #     outfile.write(bufout)

        # chunk_size = adjusted_chunk_size
        # info = {
        #     "@type": "neuroglancer_annotations_v1",
        #     "type": "points",  # Different key for binary format
        #     "by_id": {"key": "by_id"},
        #     "dimensions": {
        #         "x": [xy_resolution * 1e-6, "m"],
        #         "y": [xy_resolution * 1e-6, "m"],
        #         "z": [z_resolution * 1e-6, "m"]
        #     },
        #     "lower_bound": [0, 0, 0],
        #     "upper_bound": chunk_size,
        #     "properties": [],
        #     "relationships": [],
        #     "spatial": [{
        #         "key": "spatial0",
        #         "chunk_size": chunk_size,
        #         "grid_shape": [1, 1, 1]
        #     }]
        # }

        # with open(info_filename, 'w') as infofile:
        #     json.dump(info, infofile, indent=2)

        ##ED code


######################################################################################


        # # Convert DataFrame to annotations
        # required_cols = ['x', 'y', 'section']
        # if not all(col in valid_df.columns for col in required_cols):
        #     raise ValueError("DataFrame must contain 'x', 'y', and 'section' columns")

        # # Create a list of dictionaries for annotations using vectorized operations
        # annotations = (
        #     valid_df
        #     .with_row_count(name="row_id")  # Add row index for ID generation
        #     .select([
        #         pl.lit("point").alias("type"),  # Explicitly set to "point"
        #         pl.col("row_id").add(1).cast(pl.Utf8).alias("id"),
        #         pl.struct(["x", "y", "section"]).cast(pl.List(pl.Int64)).alias("point"),
        #         # Create properties struct with dynamic columns
        #         pl.struct(
        #             pl.col("*").exclude(["x", "y", "section", "row_id"]).cast(pl.Utf8).fill_null(""),
        #             color=pl.lit("#FF0000"),
        #             description=pl.concat_str([pl.lit("Point "), pl.col("row_id").add(1)])
        #         ).alias("properties")
        #     ])
        #     .to_dicts()  # Convert to list of dictionaries
        # )

        # # Create spatial chunk data
        # spatial_data = {
        #     "annotations": annotations,
        #     "relationships": []
        # }
        
        # # Save spatial chunk (using single chunk for simplicity)
        # spatial_file = Path(spatial_dir, "0_0_0")
        # with open(spatial_file, 'w') as f:
        #     json.dump(spatial_data, f, indent=2)

        # info_properties = [
        #                     {
        #                         "id": "color",
        #                         "type": "rgb",
        #                         "description": "Color of the annotation"
        #                     },
        #                     {
        #                         "id": "description",
        #                         "type": "text",
        #                         "description": "Description of the annotation"
        #                     }
        #                 ]

        # # Create info file
        # info_data = {
        #     "@type": "neuroglancer_annotations_v1",
        #     "annotation_type": "point",
        #     "dimensions": {
        #         "x": [xy_resolution, "m"],
        #         "y": [xy_resolution, "m"],
        #         "z": [z_resolution, "m"]
        #     },
        #     "lower_bound": [0, 0, 0],
        #     "upper_bound": expected_shape,
        #     "by_id": {
        #         "key": "by_id"
        #     },
        #     "properties": info_properties,
        #     "spatial": [
        #         {
        #             "key": "spatial0",
        #             "chunk_size": adjusted_chunk_size,
        #             "grid_shape": [
        #                 max(1, expected_shape[0] // adjusted_chunk_size[0]),
        #                 max(1, expected_shape[1] // adjusted_chunk_size[1]),
        #                 max(1, expected_shape[2] // adjusted_chunk_size[2])
        #             ]
        #         }
        #     ]
        # }

        # info_file = Path(OUTPUT_DIR, "info")
        # with open(info_file, 'w') as f:
        #     json.dump(info_data, f, indent=2)
        
        # print(f"Created {len(annotations)} annotations in {OUTPUT_DIR}")

        sys.exit()
        return len(annotations)
        

            # Assign unique IDs to each point
            # if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
            #     volume[x, y, z] = current_id  
            #     current_id += 1  # Increment ID for the next point

            #single point annotation
            # if 0 <= x < w and 0 <= y < h and 0 <= z < z_length:
            #     volume[z][y, x] = 1 # Set a single pixel to label value 1

            # Draw a small circle to make the point more visible
            #cv2.circle(volume[z], center=(x, y), radius=1, color=1, thickness=-1)  # label = 1

        encoding="raw"
        precompute = NgPrecomputedMaker(self.sqlController)
        scales = precompute.get_scales()
        max_workers = self.get_nworkers()
        preferred_chunk_size = (128, 128, 64)
        adjusted_chunk_size = tuple(min(p, s) for p, s in zip(preferred_chunk_size, shape))
        ################################################
        # ANNOTATION LAYER CREATION
        ################################################
        create_annotation_format(valid_df, encoding, temp_output_path, shape, src_dtype, scales, max_workers, adjusted_chunk_size)
        print('done')
        sys.exit()
        ################################################
        # PRECOMPUTED FORMAT CREATION - SEGMENTATION
        ################################################
        

        print(f'Creating precomputed annotations in {temp_output_path}')
        print(f"{shape=}")
        print(f"{scales=}")
        print(f"{adjusted_chunk_size=}")

        info = CloudVolume.create_new_info(
            num_channels=1,
            layer_type='segmentation', 
            data_type=src_dtype, 
            encoding=encoding,
            resolution=scales,
            voxel_offset=(0, 0, 0),
            chunk_size=adjusted_chunk_size,
            volume_size=shape,  # x, y, z
        )

        vol = CloudVolume(f'file://{temp_output_path}', progress=True, info=info, parallel=True, non_aligned_writes=False, provenance={})
        vol.commit_info()
        # vol[:, :, :] = volume #THIS IS SINGLE CORE! maybe good for debug mode?

        #########################################
        #create Ng from polars directly
        #########################################
        
        chunk_x = (valid_df['x'] // adjusted_chunk_size[0]).cast(pl.UInt32)
        chunk_y = (valid_df['y'] // adjusted_chunk_size[1]).cast(pl.UInt32)
        chunk_z = (valid_df['section'] // adjusted_chunk_size[2]).cast(pl.UInt32)

        # Add as temporary columns
        df_chunks = valid_df.with_columns([
            chunk_x.alias('cx'),
            chunk_y.alias('cy'),
            chunk_z.alias('cz')
        ])

        # Group points by chunk
        grouped = (
            df_chunks
            .group_by(['cx', 'cy', 'cz'])
            .agg(
                x_list=pl.col('x'),
                y_list=pl.col('y'),
                z_list=pl.col('section')
            )
        )

        print(f"Total non-empty chunks: {grouped.height}")
        chunks_list = grouped.to_dicts()

        args_list = [
            (chunk_dict, temp_output_path, adjusted_chunk_size, shape)
            for chunk_dict in chunks_list
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for _ in tqdm(executor.map(write_chunk, args_list),
                        total=len(chunks_list),
                        desc="Writing chunks"):
                pass    
       
        ################################################
        # CREATE/MODIFY PROVENANCE FILE WITH META-DATA
        ################################################
        mips = 3
        try:
            with open(prov_path, 'r') as f:
                prov = json.load(f)
        except Exception as e:
            prov = {}
        prov_path = Path(temp_output_path, 'provenance')
        if self.sampling:
            prov['description'] = f"SEGMENTATION VOL OF ANNOTATIONS - SAMPLING ENABLED (n={self.sampling} of {df.height} TOTAL)"
        else:
            prov['description'] = f"SEGMENTATION VOL OF ANNOTATIONS - (n={df.height})"
        prov['sources'] = [f"subject={self.animal}, ML_segmentation task"]

        current_processing = {
            'method': {
                'task': 'PyramidGeneration',
                'mips': mips,
                'chunk_size': [int(x) for x in adjusted_chunk_size],
                'encoding': encoding,
                'ML_segmentation': {  # put full segmentation info here
                    "min_segment_size": self.segment_size_min,
                    "max_segment_size": self.segment_size_max,
                    "gaussian_blur_standard_deviation_sigmaX": self.gaussian_blur_standard_deviation_sigmaX,
                    "gaussian_blur_kernel_size_pixels": self.gaussian_blur_kernel_size_pixels,
                    "segmentation_threshold": self.segmentation_threshold,
                    "cell_radius": self.cell_radius
                }
                }
            }

        prov['processing'] = [current_processing]
        prov['owners'] = [f'PERFORMANCE LAB: {performance_lab}']
        with open(prov_path, 'w') as f:
            json.dump(prov, f, indent=2)

        #################################################
        # PRECOMPUTED FORMAT PYRAMID (DOWNSAMPLED) IN-PLACE
        #################################################
        cloudpath = f"file://{temp_output_path}" #full-resolution (already generated)
        # outpath = f"file://{temp_output_path_pyramid}"

        tq = LocalTaskQueue(parallel=1)
        tasks = tc.create_downsampling_tasks(
                                            layer_path=cloudpath, 
                                            mip=0, 
                                            num_mips=mips, 
                                            compress=True,
                                            encoding=encoding,
                                            sparse=True,
                                            fill_missing=True,
                                            delete_black_uploads=True,
                                            chunk_size=adjusted_chunk_size
                                            )
        tq.insert(tasks)
        tq.execute()

        #MOVE PRECOMPUTED [ALL MIPS] FILES TO FINAL LOCATION
        copy_with_rclone(temp_output_path, OUTPUT_DIR)

 
    def create_segmentation_layer(self, df_points: pl.DataFrame, expected_shape: tuple, scratch_tmp: str, counterstain_channel: str, label_of_interest_channel: str):
        '''Create annotations for for cell candidates
        Note: output will be annotation layer compatible with neuroglancer


        1. cleans temp directories (progress_dir, temp_output_path, temp_output_path_pyramid)
        2. cleans final output dir (OUTPUT_DIR)
        3. recreates temp and final directories
        4. generates precomputed format from annotations' csv files @ temp locations
        5. moves final MIP precomputed to final dest. folder
        '''

        if self.debug:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} Start")
            print("Populating full resolution volume with vectorized operations...")

        # labels = ['ann_' + self.set_id + '_' + label_of_interest_channel]
        # prefix = labels[0]
        # sampling = self.sampling
        
        # xy_resolution = self.sqlController.scan_run.resolution
        # z_resolution = self.sqlController.scan_run.zresolution

        # #READ, CONSOLIDATE PREDICTION FILES, SAMPLE AS NECESSARY
        # if os.path.exists(self.cell_label_path):
        #     print(f'Parsing cell labels from {self.cell_label_path}')
        # else:
        #     print(f'ERROR: {self.cell_label_path} not found')
        #     sys.exit(1)

        # #ACTUAL FILE DIMENSIONS TAKE PRECEDENCE OVER DATABASE
        # param_file = Path(self.cell_label_path, 'parameters.json')
        # with open(param_file, 'r') as f:
        #     params = json.load(f)
        # expected_x, expected_y, z_length = params["vol_shape_x_y_z"]
        
        # detection_files = sorted(glob.glob(os.path.join(self.cell_label_path, f'detections_*.csv') ))
        # if len(detection_files) == 0:
        #     print(f'Error: no csv files found in {self.cell_label_path}')
        #     sys.exit(1)
        # else:
        #     print(f'Parsing {len(detection_files)} sections...')

        # ann_out_folder_name = 'ann_' + self.set_id + '_' + label_of_interest_channel
        # OUTPUT_DIR = Path(self.fileLocationManager.neuroglancer_data, 'annotations', ann_out_folder_name + '.pre') 

        # #TODO: Needs re-worked for retraining models
        # # Regex pattern to match folder names starting with prefix and ending with _{int}, e.g. ML_POS_0, ML_POS_12
        # found_any = False
        # max_suffix = -1
        # if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        #     pattern_template = r'^{}(?:_(\d+))?$'  # group 1 captures optional suffix number
        #     pattern = re.compile(pattern_template.format(re.escape(prefix)))
        #     for folder in OUTPUT_DIR.iterdir():
        #         if folder.is_dir():
        #             m = pattern.match(folder.name)
        #             if m:
        #                 found_any = True
        #                 if m.group(1) is None:
        #                     # Folder has prefix but no number suffix
        #                     max_suffix = max(max_suffix, -1)
        #                 else:
        #                     num = int(m.group(1))
        #                     if num > max_suffix:
        #                         max_suffix = num

        # if not found_any:
        #     # Directory empty of matching prefix folders
        #     ann_out_folder_name = prefix
        # else:
        #     # We found matching folders, get next suffix number
        #     next_number = max_suffix + 1
        #     ann_out_folder_name = f"{prefix}_{next_number}"

        # #NAMING CONVENTION FOR ANNOTATION FOLDERS
        # progress_dir = Path(self.fileLocationManager.neuroglancer_data, 'progress', ann_out_folder_name)
        # temp_output_path = Path(self.SCRATCH, 'pipeline_tmp', self.animal, ann_out_folder_name + '_ng')

        # #CLEAN UP TEMP & FINAL DESTNATION FOLDERS - ON HOLD
        # temp_folders = [progress_dir, temp_output_path, OUTPUT_DIR]
        # try:
        #     for folder in temp_folders:
        #         if self.debug:
        #             print(f"DEBUG: Deleting temporary folder: {folder}")
        #         delete_in_background(folder)
        # except Exception as e:
        #     print(f"Non-critical Error deleting progress directory: {e}")

        #NAMING CONVENTION FOR ANNOTATION FOLDERS
        labels = ['ann_' + self.set_id + '_' + label_of_interest_channel]
        prefix = labels[0]
        ann_out_folder_name = 'ann_' + self.set_id + '_' + label_of_interest_channel
        OUTPUT_DIR = Path(self.fileLocationManager.neuroglancer_data, 'annotations', ann_out_folder_name + '.seg_pre') 

        #TODO: Needs re-worked for retraining models
        # Regex pattern to match folder names starting with prefix and ending with _{int}, e.g. ML_POS_0, ML_POS_12
        found_any = False
        max_suffix = -1
        if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
            pattern_template = r'^{}(?:_(\d+))?$'  # group 1 captures optional suffix number
            pattern = re.compile(pattern_template.format(re.escape(prefix)))
            for folder in OUTPUT_DIR.iterdir():
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

        progress_dir = Path(self.fileLocationManager.neuroglancer_data, 'progress', ann_out_folder_name)
        temp_output_path = Path(self.SCRATCH, 'pipeline_tmp', self.animal, ann_out_folder_name + '_ng')
        performance_lab = self.sqlController.histology.FK_lab_id
        xy_resolution = self.sqlController.scan_run.resolution
        z_resolution = self.sqlController.scan_run.zresolution
        preferred_chunk_size = (128, 128, 64)
        adjusted_chunk_size = tuple(min(p, s) for p, s in zip(preferred_chunk_size, expected_shape))

        print('*'*50)
        print(f'DEBUG: {xy_resolution=}, {z_resolution=}, {expected_shape=}')
        print("\tTEMP ng Output:".ljust(20), f"{temp_output_path}".ljust(20))
        print("\tTEMP Progress:".ljust(20), f"{progress_dir}".ljust(20))
        print("\tFINAL Output:".ljust(20), f"{OUTPUT_DIR}".ljust(20))

        # dfs = []
        # for file_path in detection_files:
        #     # Read CSV with Polars (much faster than pandas)
        #     try:
        #         df = pl.read_csv(file_path)
        #         if df.is_empty():
        #             continue
                    
        #         # Filter and process in one go
        #         filtered = df.filter(
        #         (pl.col("predictions").cast(pl.Float32) > 0))
                
        #         if filtered.is_empty():
        #             continue
                    
        #         # Process all rows at once
        #         x_vals = (filtered["col"].cast(pl.Float32) / M_UM_SCALE * xy_resolution).to_list()
        #         y_vals = (filtered["row"].cast(pl.Float32) / M_UM_SCALE * xy_resolution).to_list()
        #         sections = ((filtered["section"].cast(pl.Float32) + 0.5) * z_resolution / M_UM_SCALE).to_list()
                
        #         # Append to dataframe data
        #         dfs.append(filtered.select([
        #             pl.col("col").alias("x"),
        #             pl.col("row").alias("y"),
        #             (pl.col("section") + 0.5).cast(pl.Int32).alias("section") #annotations were offset by 1 (perhaps due to round down)
                    
        #         ]))
            
        #     except Exception as e:
        #         print(f"Error processing {file_path}: {e}")
        #         continue

        # # Concatenate all dataframes at once
        # final_df = pl.concat(dfs) if dfs else pl.DataFrame()

        # #random sampling for model training, if selected
        # # Choose between full dataset or sampled subset
        # if self.sampling:
        #     sampled_df = final_df.sample(n=sampling, seed=42) if final_df.height > sampling else final_df
        #     df = sampled_df
        #     print(f"Sampling enabled - randomly sampled neuron count: {len(df)}")
        # else:
        #     df = final_df
        #     print(f"No sampling - using all {df.height} points")
        
        # if len(df) == 0:
        #     print('No neurons found')
        #     sys.exit()

        # ###############################################
        # #ALL ANNOTATION POINTS ARE STORED IN df VARIABLE AT THIS POINT
        # ###############################################

        ###############################################
        # SAVE OUTPUTS
        ###############################################
        # SAVE CSV FORMAT (x,y,z POINTS IN CSV) *SAMPLED DATA
        # csv_dump = str(Path(self.fileLocationManager.neuroglancer_data, 'annotations', ann_out_folder_name)) + '.csv'
        # if not df.is_empty():
        #     df.write_csv(csv_dump)

        # # SAVE JSON FORMAT *SAMPLED DATA
        # annotations_file = str(Path(self.fileLocationManager.neuroglancer_data, 'annotations', ann_out_folder_name)) +'.json'

        # # # Populate points variable after sampling
        # sampled_points = (
        #     df.sort("section")
        #     .select(["x", "y", "section"])
        #     .to_numpy()
        #     .tolist()
        # )
        # with open(annotations_file, 'w') as fh:
        #     json.dump(sampled_points, fh)

        # SAVE PRECOMPUTED [SEGMENTATION] FORMAT *SAMPLED DATA
        src_dtype = 'uint8'
        volume = np.zeros(expected_shape, dtype=src_dtype)

        # Filter valid coordinates first
        valid_df = df_points.filter(
            (pl.col("x") >= 0) & (pl.col("x") < expected_shape[0]) &
            (pl.col("y") >= 0) & (pl.col("y") < expected_shape[1]) &
            (pl.col("section") >= 0) & (pl.col("section") < expected_shape[2])
        )
        invalid_count = len(df_points) - len(valid_df)
        if invalid_count > 0:
            print(f"Skipping {invalid_count} out-of-bounds points")

        if self.debug:
            print(f'{expected_shape=}, {volume.shape=}')

        # Draw circles for each valid point
        for row in valid_df.iter_rows(named=True):
            x = int(row['x'])
            y = int(row['y'])
            z = int(row['section'])  # This is your Z coordinate
            
            # Draw circle on the specific Z-slice
            # Note: volume[z] gives us the 2D slice at depth z
            cv2.circle(volume[z], center=(x, y), radius=2, color=1, thickness=-1)  # radius=2, filled circle
        print(f"Drew {len(valid_df)} circles in volume")

            # Assign unique IDs to each point
            # if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
            #     volume[x, y, z] = current_id  
            #     current_id += 1  # Increment ID for the next point

            #single point annotation
            # if 0 <= x < w and 0 <= y < h and 0 <= z < z_length:
            #     volume[z][y, x] = 1 # Set a single pixel to label value 1

        # Draw a small circle to make the point more visible
        # cv2.circle(volume[z], center=(x, y), radius=1, color=1, thickness=-1)  # label = 1
        
        # num_annotations = create_neuroglancer_annotations(
        #     valid_df=valid_df,
        #     output_dir=temp_output_path,
        #     expected_x=expected_x,
        #     expected_y=expected_y, 
        #     z_length=z_length
        # )
        # print(f"Successfully created {num_annotations} annotations in {temp_output_path}")

        encoding="raw"
        precompute = NgPrecomputedMaker(self.sqlController)
        scales = precompute.get_scales()
        max_workers = self.get_nworkers()

        # ################################################
        # # PRECOMPUTED FORMAT CREATION - SEGMENTATION
        # ################################################
        

        # print(f'Creating precomputed annotations in {temp_output_path}')
        # print(f"{shape=}")
        # print(f"{scales=}")
        # print(f"{adjusted_chunk_size=}")

        info = CloudVolume.create_new_info(
            num_channels=1,
            layer_type='segmentation', 
            data_type=src_dtype, 
            encoding=encoding,
            resolution=scales,
            voxel_offset=(0, 0, 0),
            chunk_size=adjusted_chunk_size,
            volume_size=volume.shape,  # x, y, z
        )

        vol = CloudVolume(f'file://{temp_output_path}', progress=True, info=info, parallel=True, non_aligned_writes=False, provenance={})
        vol.commit_info()
        # # vol[:, :, :] = volume #THIS IS SINGLE CORE! maybe good for debug mode?

        # #########################################
        # #create Ng from polars directly (multi-core)
        # #########################################
        chunk_x = (valid_df['x'] // adjusted_chunk_size[0]).cast(pl.UInt32)
        chunk_y = (valid_df['y'] // adjusted_chunk_size[1]).cast(pl.UInt32)
        chunk_z = (valid_df['section'] // adjusted_chunk_size[2]).cast(pl.UInt32)

        # Add as temporary columns
        df_chunks = valid_df.with_columns([
            chunk_x.alias('cx'),
            chunk_y.alias('cy'),
            chunk_z.alias('cz')
        ])

        # Group points by chunk
        grouped = (
            df_chunks
            .group_by(['cx', 'cy', 'cz'])
            .agg(
                x_list=pl.col('x'),
                y_list=pl.col('y'),
                z_list=pl.col('section')
            )
        )

        print(f"Total non-empty chunks: {grouped.height}")
        chunks_list = grouped.to_dicts()

        args_list = [
            (chunk_dict, temp_output_path, adjusted_chunk_size, volume.shape)
            for chunk_dict in chunks_list
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for _ in tqdm(executor.map(write_chunk, args_list),
                        total=len(chunks_list),
                        desc="Writing chunks"):
                pass    
       
        ################################################
        # CREATE/MODIFY PROVENANCE FILE WITH META-DATA
        ################################################
        mips = 3
        try:
            with open(prov_path, 'r') as f:
                prov = json.load(f)
        except Exception as e:
            prov = {}
        prov_path = Path(temp_output_path, 'provenance')
        if self.sampling:
            prov['description'] = f"SEGMENTATION VOL OF ANNOTATIONS - SAMPLING ENABLED (n={self.sampling} of {valid_df.height} TOTAL)"
        else:
            prov['description'] = f"SEGMENTATION VOL OF ANNOTATIONS - (n={valid_df.height})"
        prov['sources'] = [f"subject={self.animal}, ML_segmentation task"]

        current_processing = {
            'method': {
                'task': 'PyramidGeneration',
                'mips': mips,
                'chunk_size': [int(x) for x in adjusted_chunk_size],
                'encoding': encoding,
                'ML_segmentation': {  # put full segmentation info here
                    "min_segment_size": self.segment_size_min,
                    "max_segment_size": self.segment_size_max,
                    "gaussian_blur_standard_deviation_sigmaX": self.gaussian_blur_standard_deviation_sigmaX,
                    "gaussian_blur_kernel_size_pixels": self.gaussian_blur_kernel_size_pixels,
                    "segmentation_threshold": self.segmentation_threshold,
                    "cell_radius": self.cell_radius
                }
                }
            }

        prov['processing'] = [current_processing]
        prov['owners'] = [f'PERFORMANCE LAB: {performance_lab}']
        with open(prov_path, 'w') as f:
            json.dump(prov, f, indent=2)

        #################################################
        # PRECOMPUTED FORMAT PYRAMID (DOWNSAMPLED) IN-PLACE
        #################################################
        cloudpath = f"file://{temp_output_path}" #full-resolution (already generated)

        tq = LocalTaskQueue(parallel=max_workers)
        tasks = tc.create_downsampling_tasks(
                                            layer_path=cloudpath, 
                                            mip=0, 
                                            num_mips=mips, 
                                            compress=True,
                                            encoding=encoding,
                                            sparse=True,
                                            fill_missing=True,
                                            delete_black_uploads=True,
                                            chunk_size=adjusted_chunk_size
                                            )
        tq.insert(tasks)
        tq.execute()
        
        #MOVE PRECOMPUTED [ALL MIPS] FILES TO FINAL LOCATION
        copy_with_rclone(temp_output_path, OUTPUT_DIR)


def write_chunk(args):
    '''
    Writes a chunk of the volume to a CloudVolume. (individual function is single core but tasks called with ProcessPoolExecutor)
    '''
    try:
        # Unpack the tuple of arguments
        chunk_dict, vol_path, chunk_size, shape = args

        # Extract chunk indices
        cx, cy, cz = chunk_dict['cx'], chunk_dict['cy'], chunk_dict['cz']

        # Compute voxel bounds
        x0, y0, z0 = cx * chunk_size[0], cy * chunk_size[1], cz * chunk_size[2]
        x1, y1, z1 = min(x0 + chunk_size[0], shape[0]), min(y0 + chunk_size[1], shape[1]), min(z0 + chunk_size[2], shape[2])

        # Extract coordinates from lists in chunk_dict
        x = np.array(chunk_dict['x_list'], dtype=np.uint32) - x0
        y = np.array(chunk_dict['y_list'], dtype=np.uint32) - y0
        z = np.array(chunk_dict['z_list'], dtype=np.uint32) - z0

        # Create slab
        slab = np.zeros((x1 - x0, y1 - y0, z1 - z0), dtype=np.uint8)
        slab[x, y, z] = 1

        # Write to CloudVolume
        vol = CloudVolume(f'file://{vol_path}', progress=False, parallel=False)
        vol[x0:x1, y0:y1, z0:z1] = slab
    except Exception as e:
        print(f"Error processing chunk ({cx}, {cy}, {cz}): {e}")
        raise


def create_annotation_format(valid_df: pl.DataFrame, encoding, temp_output_path: str, shape: tuple, src_dtype: str, scales: list, max_workers: int, adjusted_chunk_size) -> list[dict]:
    """
    Create and save annotation data in Neuroglancer precomputed format from a Polars DataFrame.

    Args:
        valid_df (pl.DataFrame): Polars DataFrame with columns 'id', 'name', 'type', 'x', 'y', 'z'.
        temp_output_path (str): Directory path to save the precomputed layer (e.g., '/path/to/C1T_aligned').
        shape (tuple): Volume dimensions (x, y, z).
        src_dtype (str): Data type for the layer (e.g., 'uint32').
        scales (list): Resolution of the volume in nanometers [x, y, z].

    Returns:
        list[dict]: List of annotation dictionaries in Neuroglancer format.

    Raises:
        ValueError: If required columns are missing or volume name is invalid.
    """
    if 'section' in valid_df.columns and 'z' not in valid_df.columns:
        valid_df = valid_df.rename({"section": "z"})
    if 'id' not in valid_df.columns:
        valid_df = valid_df.with_columns(pl.lit(None).cast(pl.Utf8).alias('id'))
    if 'name' not in valid_df.columns:
        valid_df = valid_df.with_columns(pl.lit("point").alias('name'))
    if 'type' not in valid_df.columns:
        valid_df = valid_df.with_columns(pl.lit("point").alias('type'))

    # Validate required columns
    required_columns = {'id', 'name', 'type', 'x', 'y', 'z'}
    if not required_columns.issubset(valid_df.columns):
        missing = required_columns - set(valid_df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Convert Polars DataFrame to annotation format
    annotations = (
        valid_df
        .select([
            pl.col('id').cast(pl.Utf8),
            pl.col('name'),
            pl.col('type'),
            pl.col('x').cast(pl.Float32),
            pl.col('y').cast(pl.Float32),
            pl.col('z').cast(pl.Float32)
        ])
        .with_columns(
            pl.struct(['x', 'y', 'z']).alias('point')
        )
        .select([
            pl.col('id'),
            pl.col('name'),
            pl.col('type'),
            pl.col('point')
        ])
        .to_dicts()
    )

    # Reformat to match Neuroglancer annotation schema
    annotations = [
        {
            'id': row['id'] if row['id'] else str(uuid.uuid4()),  # Fallback to UUID if id is null
            'type': row['type'],  # e.g., 'point'
            'point': [row['point']['x'], row['point']['y'], row['point']['z']],  # Neuroglancer expects list
            'props': {'name': row['name']}  # Store name as a property
        }
        for row in annotations
    ]

    # Validate volume name
    volume_name = Path(temp_output_path).name
    # if not re.match(r'^C\d+T_(?:aligned|realigned)$', volume_name):
    #     raise ValueError(f"Volume name {volume_name} does not match expected pattern")

    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type='annotation',
        data_type=src_dtype,
        encoding=encoding,
        resolution=scales,
        voxel_offset=(0, 0, 0),
        chunk_size=adjusted_chunk_size,
        volume_size=shape,
    )
    info['@type'] = 'neuroglancer_annotations_v1'
    info['annotations'] = [{
        "name": "points",
        "type": "point",
        "properties": [
            {"name": "name", "type": "string"}
        ]
    }]

    # Initialize CloudVolume and commit info
    vol = CloudVolume(f'file://{temp_output_path}', progress=True, info=info, parallel=True, non_aligned_writes=False, provenance={})
    vol.commit_info()


    # Write annotations to disk
    def write_annotation_chunk(chunk, chunk_path):
        """Write a chunk of annotations to a JSON file."""
        os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
        with open(chunk_path, 'w') as f:
            json.dump(chunk, f, indent=2)

     # Group annotations by chunk for spatial indexing
    annotations_dir = Path(temp_output_path) / 'annotations' / 'spatial0'
    chunked_annotations = {}
    for ann in annotations:
        # Compute chunk coordinates
        x, y, z = ann['point']
        cx = int(x // adjusted_chunk_size[0])
        cy = int(y // adjusted_chunk_size[1])
        cz = int(z // adjusted_chunk_size[2])
        chunk_key = f"{cx}_{cy}_{cz}"
        if chunk_key not in chunked_annotations:
            chunked_annotations[chunk_key] = []
        chunked_annotations[chunk_key].append(ann)

    os.makedirs(annotations_dir, exist_ok=True)
    # Write chunks in parallel
    for key, chunk in chunked_annotations.items():
        chunk_path = annotations_dir / f"{key}.json"
        with open(chunk_path, 'w') as f:
            json.dump(chunk, f, indent=2)

    # # Optional: Create downsampled pyramid
    # if self.downsample:
    #     self.fileLogger.info("Creating downsampled pyramid for visualization...")
    #     tq = LocalTaskQueue(parallel=max_workers)
    #     tasks = taskqueue.create_downsampling_tasks(
    #         layer_path=f"file://{temp_output_path}",
    #         mip=0,
    #         num_mips=4,
    #         compress=True,
    #         encoding=encoding,
    #         sparse=True,
    #         fill_missing=True,
    #         delete_black_uploads=True,
    #         chunk_size=adjusted_chunk_size
    #     )
    #     tq.insert(tasks)
    #     tq.execute()
    #     self.fileLogger.info(f"Created pyramid for {temp_output_path}")

    return annotations


def create_neuroglancer_annotations(valid_df, output_dir, expected_x, expected_y, z_length):
    """
    Convert Polars DataFrame to Neuroglancer precomputed annotations
    
    Args:
        valid_df: Filtered Polars DataFrame with x, y, section columns
        output_dir: Directory to save annotation files
        expected_x, expected_y, z_length: Dimensions for coordinate validation
    """
    
    # Create directory structure
    annotations_dir = Path(output_dir)
    spatial_dir = annotations_dir / "spatial0"
    spatial_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert DataFrame to annotations
    annotations = []
    
    for row in valid_df.iter_rows(named=True):
        annotation = {
            "point": [
                int(row['x']),  # x coordinate
                int(row['y']),  # y coordinate  
                int(row['section'])  # z coordinate (section)
            ],
            "type": "point",
            "id": str(len(annotations) + 1),
            "properties": {
                "color": "#FF0000",  # Default red, you can customize
                "description": f"Point {len(annotations) + 1}"
            }
        }
        
        # Add any additional properties from your DataFrame
        for col in valid_df.columns:
            if col not in ['x', 'y', 'section'] and row[col] is not None:
                annotation['properties'][col] = str(row[col])
        
        annotations.append(annotation)
    
    # Create spatial chunk data
    spatial_data = {
        "annotations": annotations,
        "relationships": []
    }
    
    # Save spatial chunk (using single chunk for simplicity)
    spatial_file = spatial_dir / "0_0_0"
    with open(spatial_file, 'w') as f:
        json.dump(spatial_data, f)
    
    # Create info file
    info_data = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": {
            "x": [1e-9, "m"],  # Adjust based on your data resolution
            "y": [1e-9, "m"],
            "z": [1e-9, "m"]
        },
        "by_id": {
            "key": "by_id"
        },
        "properties": [
            {
                "id": "color",
                "type": "rgb",
                "description": "Color of the annotation"
            },
            {
                "id": "description", 
                "type": "text",
                "description": "Description of the annotation"
            }
        ],
        "spatial": [
            {
                "key": "spatial0",
                "chunk_size": [expected_x, expected_y, z_length]  # Single chunk covering entire volume
            }
        ]
    }
    
    # Add dynamic properties from DataFrame columns
    for col in valid_df.columns:
        if col not in ['x', 'y', 'section']:
            info_data["properties"].append({
                "id": col,
                "type": "text",
                "description": f"Column {col} from DataFrame"
            })
    
    # Save info file
    info_file = annotations_dir / "info"
    with open(info_file, 'w') as f:
        json.dump(info_data, f)
    
    print(f"Created {len(annotations)} annotations in {output_dir}")
    return len(annotations)