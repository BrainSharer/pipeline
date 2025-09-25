import os, sys, json
from cloudvolume import CloudVolume
from taskqueue.taskqueue import LocalTaskQueue
import igneous.task_creation as tc

from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import psutil
import gc
from library.utilities.cell_utilities import (
    copy_with_rclone
)

from library.image_manipulation.neuroglancer_manager import NumpyToNeuroglancer
from library.utilities.dask_utilities import closest_divisors_to_target
from library.utilities.utilities_process import test_dir, read_image
from library.image_manipulation.image_manager import ImageManager
XY_CHUNK = 64
Z_CHUNK = 64


class NgPrecomputedMaker:
    """Class to convert a tiff image stack to the precomputed
    neuroglancer format code from Seung lab
    """

    def __init__(self, sqlController, *args, **kwargs):
        self.sqlController = sqlController
        # Initialize other attributes as needed
        self.input = kwargs.get('input', None)
        self.output = kwargs.get('output', None)
        self.animal = kwargs.get('animal', None)
        self.section_count = kwargs.get('section_count', None)
        self.downsample = kwargs.get('downsample', False)
        self.scaling_factor = kwargs.get('scaling_factor', 1.0)
        self.rechunkme_path = kwargs.get('rechunkme_path', None)
        self.progress_dir = kwargs.get('progress_dir', None)
        self.fileLogger = kwargs.get('fileLogger', None)
        self.debug = kwargs.get('debug', False)

    def get_scales(self):
        """returns the scanning resolution for a given animal.  
        The scan resolution and sectioning thickness are retrived from the database.
        The resolution in the database is stored as micrometers (microns -um). But
        neuroglancer wants nanometers so we multipy by 1000

        :returns: list of converstion factors from pixel to micron for x,y and z
        """
        db_resolution = self.sqlController.scan_run.resolution
        zresolution = self.sqlController.scan_run.zresolution
        resolution = db_resolution * 1000
        if self.downsample:
          if zresolution < 20:
                zresolution *=  self.scaling_factor

          resolution *=  self.scaling_factor         

        resolution = int(resolution)
        zresolution = int(zresolution * 1000)
        scales = (resolution, resolution, zresolution)
        return scales


    def create_neuroglancer(self):
        """create the Seung lab cloud volume format from the image stack
        For a large isotropic data set, Allen uses chunks = [128,128,128]
        self.input and self.output are defined in the pipeline_process
        Using chunks=[4172, 2552, 1] for neuroglancer rechunkme transfer step
        Finished Creating Neuroglancer data.
        neuroglancer took 1549.0 seconds.
        14142608	/data/pipeline_tmp/CTB016/C1_rechunkme_aligned/
        ############
        Using chunks=[66752, 40832, 1] for neuroglancer rechunkme transfer step
        Finished Creating Neuroglancer data.
        neuroglancer took 1349.48 seconds.
        15001680	/data/pipeline_tmp/CTB016/C1_rechunkme_aligned/
        ####
        Using chunks=[1856, 2086, 1] for neuroglancer rechunkme transfer step
        neuroglancer took 1579.27 seconds.
        """
        image_manager = ImageManager(self.input)

        if self.downsample:
            self.xy_chunk = int(XY_CHUNK//2)
            chunks = [self.xy_chunk, self.xy_chunk, 1]
        else:
            target_chunk = 2048
            chunk_x = closest_divisors_to_target(image_manager.width, target=target_chunk)
            chunk_y = closest_divisors_to_target(image_manager.height, target=target_chunk)
            chunks = [chunk_x, chunk_y, 1] # 1796x984x1, xyz

        chunks = [image_manager.width, image_manager.height, 1] # 1796x984x1, xyz

        #test_dir(self.animal, self.input, self.section_count, self.downsample, same_size=True)
        scales = self.get_scales()
        print(f'scales={scales} scaling_factor={self.scaling_factor} downsample={self.downsample}')
        num_channels = image_manager.num_channels
        print(image_manager.width, image_manager.height, image_manager.len_files, image_manager.shape)
        print(f'volume_size={image_manager.volume_size} ndim={image_manager.ndim} dtype={image_manager.dtype} num_channels={num_channels} and size={image_manager.size}')
        print(f'Using chunks={chunks} for neuroglancer rechunkme transfer step')
        ng = NumpyToNeuroglancer(
            self.animal,
            None,
            scales,
            "image",
            image_manager.dtype,
            num_channels=num_channels,
            chunk_size=chunks,
        )
        
        ng.init_precomputed(self.rechunkme_path, image_manager.volume_size)
        file_keys = []
        orientation = self.sqlController.histology.orientation
        for i, f in enumerate(image_manager.files):
            filepath = os.path.join(self.input, f)
            file_keys.append([i, filepath, orientation, self.progress_dir, False, 0, 0]) #added is_blank, height, width

        workers = self.get_nworkers()
        if self.debug:
            workers = 1

        if self.debug:
            for file_key in file_keys:
                ng.process_image(file_key=file_key)
        else:
            self.run_commands_concurrently(ng.process_image, file_keys, workers)
        ng.precomputed_vol.cache.flush()


    def create_downsamples(self):
        """Downsamples the neuroglancer cloudvolume this step is needed to make the files viewable in neuroglancer
        """

        image_manager = ImageManager(self.input)

        chunks = [XY_CHUNK, XY_CHUNK, Z_CHUNK]
        if self.downsample:
            chunks = [self.xy_chunk, self.xy_chunk, self.xy_chunk]

        if not self.downsample and self.section_count < 100:
            z_chunk = int(XY_CHUNK)//2
            chunks = [XY_CHUNK, XY_CHUNK, z_chunk]

        if self.downsample or image_manager.size < 100000000:
            mips = 4
        else:
            mips = 7

        if os.path.exists(self.output):
            print(f"DIR {self.output} already exists. Downsampling has already been performed.")
            return
        outpath = f"file://{self.output}"
        if not os.path.exists(self.rechunkme_path):
            print(f"DIR {self.rechunkme_path} does not exist, exiting.")
            sys.exit()
        cloudpath = f"file://{self.rechunkme_path}"
        self.fileLogger.logevent(f"Input DIR: {self.rechunkme_path}")
        self.fileLogger.logevent(f"Output DIR: {self.output}")
        workers =self.get_nworkers()

        tq = LocalTaskQueue(parallel=workers)

        if image_manager.num_channels > 2:
            print(f'Creating non-sharded transfer tasks with chunks={chunks} and section count={self.section_count}')
            tasks = tc.create_transfer_tasks(cloudpath, dest_layer_path=outpath, max_mips=mips, chunk_size=chunks, mip=0, skip_downsamples=True)
        else:
            print(f'Creating sharded transfer tasks with chunks={chunks} and section count={self.section_count}')
            tasks = tc.create_image_shard_transfer_tasks(cloudpath, outpath, mip=0, chunk_size=chunks)

        tq.insert(tasks)
        tq.execute()
        print('Finished transfer tasks')

        for mip in range(0, mips):
            cv = CloudVolume(outpath, mip)
            if image_manager.num_channels > 2:
                print(f'Creating downsample tasks at mip={mip}')
                tasks = tc.create_downsampling_tasks(cv.layer_cloudpath, mip=mip, num_mips=1, compress=True)
            else:
                print(f'Creating sharded downsample tasks at mip={mip}')
                tasks = tc.create_image_shard_downsample_tasks(cv.layer_cloudpath, mip=mip, chunk_size=chunks)
            tq.insert(tasks)
            tq.execute()


    def create_precomputed(self, input, temp_output_path, OUTPUT_DIR, progress_dir, max_memory_gb: int = 100):
        """
        REVISED NEUROGLANCER METHOD - COMBINES 'rechunkme' (MIP=0) WITH DOWNSAMPLING (MIP>0)
        
        create the Seung lab cloud volume format from the image stack
        For a large isotropic data set, Allen uses chunks = [128,128,128]
        self.input and self.output are defined in the pipeline_process
        """
        performance_lab = self.sqlController.histology.FK_lab_id
        image_manager = ImageManager(input)
        shape = image_manager.volume_size
        src_dtype = image_manager.dtype
        num_channels = image_manager.num_channels
        if self.downsample or image_manager.size < 100000000:
            mips = 4
        else:
            mips = 7

        if self.downsample:
            self.xy_chunk = int(XY_CHUNK//2)
            chunks = [self.xy_chunk, self.xy_chunk, 1]
        else:
            chunks = [image_manager.height//16, image_manager.width//16, 1] # 1796x984

        adjusted_chunk_size = chunks
        test_dir(self.animal, self.input, self.section_count, self.downsample, same_size=True)

        encoding="raw"
        scales = self.get_scales()
        max_workers = self.get_nworkers()
        if self.debug:
            max_workers = 1

        print(f'Creating precomputed annotations in {temp_output_path}')
        print(f"{shape=}")
        print(f"{scales=}")
        
        info = CloudVolume.create_new_info(
            num_channels=1,
            layer_type='image', 
            data_type=src_dtype, 
            encoding=encoding,
            resolution=scales,
            voxel_offset=(0, 0, 0),
            chunk_size=adjusted_chunk_size,
            volume_size=shape  # x, y, z
        )

        vol = CloudVolume(f'file://{temp_output_path}', progress=True, info=info, parallel=True, non_aligned_writes=False, provenance={})
        vol.commit_info()

        # Generate and write volume data in parallel
        optimal_chunk_size = self.calculate_optimal_chunk_size(
            image_manager, max_memory_gb, max_workers
        )
        
        print(f"Using chunk size: {optimal_chunk_size} slices per chunk")
        self.generate_volume_parallel(vol, image_manager, max_workers, optimal_chunk_size, shape, input, progress_dir)
        file_name = 'mip0_complete'
        Path(progress_dir, file_name).touch()

        ################################################
        # CREATE/MODIFY PROVENANCE FILE WITH META-DATA
        ################################################
        prov_path = Path(temp_output_path, 'provenance')
        try:
            with open(prov_path, 'r') as f:
                prov = json.load(f)
        except Exception as e:
            prov = {}
        prov['description'] = f"IMAGE VOL"
        prov['sources'] = [f"subject={self.animal}, neuroglancer task"]

        current_processing = {
            'method': {
                'task': 'PyramidGeneration',
                'mips': mips,
                'chunk_size': [int(x) for x in adjusted_chunk_size],
                'encoding': encoding
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
        
        # Process each MIP level sequentially (NOT WORKING @ 17-SEP-2025)
        tq = LocalTaskQueue(parallel=1)  # Use 1 core to avoid memory spikes
        max_mip = len(vol.available_mips)

        # Process mips in smaller batches (2-3 at a time)
        batch_size = 2  # Adjust based on your memory constraints
        for start_mip in range(0, mips - 1, batch_size):
            end_mip = min(start_mip + batch_size, mips - 1)
            num_mips_in_batch = end_mip - start_mip

            if start_mip >= max_mip:
                break
            else:
                print(f"Processing downsampling from MIP {start_mip} to MIP {end_mip}...")
            
            tasks = tc.create_downsampling_tasks(
                layer_path=cloudpath, 
                mip=start_mip, 
                num_mips=num_mips_in_batch,  # Process fewer mips at a time
                compress=True,
                encoding=encoding,
                sparse=True,
                fill_missing=True,
                delete_black_uploads=True,
                chunk_size=adjusted_chunk_size,
            )
            tq.insert(tasks)
            tq.execute()
            
            print(f"Completed MIP levels {start_mip} to {end_mip}")
            file_name = f'{end_mip}_complete'
            Path(progress_dir, file_name).touch()

        # tq = LocalTaskQueue(parallel=1) #ONLY USE 1 CORE DUE TO MEMORY SPIKE ISSUE
        # tasks = tc.create_downsampling_tasks(
        #                                     layer_path=cloudpath, 
        #                                     mip=0, 
        #                                     num_mips=mips, 
        #                                     compress=True,
        #                                     encoding=encoding,
        #                                     sparse=True,
        #                                     fill_missing=True,
        #                                     delete_black_uploads=True,
        #                                     chunk_size=adjusted_chunk_size,
        #                                     )
        # tq.insert(tasks)
        # tq.execute()
        
        

        #MOVE PRECOMPUTED [ALL MIPS] FILES TO FINAL LOCATION
        copy_with_rclone(temp_output_path, OUTPUT_DIR)


    def calculate_optimal_chunk_size(self, image_manager, max_memory_gb, max_workers):
        """
        Calculate optimal chunk size based on available memory and image size
        """
        # Get memory info
        available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        safe_memory_gb = min(available_memory_gb * 0.7, max_memory_gb)
        
        # Estimate memory per slice
        slice_memory_bytes = (image_manager.height * image_manager.width * 
                            image_manager.num_channels * np.dtype(image_manager.dtype).itemsize)
        slice_memory_gb = slice_memory_bytes / (1024 ** 3)
        
        # Calculate max slices that fit in memory
        max_slices = int(safe_memory_gb / slice_memory_gb)
        
        # Consider worker parallelism
        chunk_size = max(1, max_slices // max_workers)
        
        # Limit chunk size for responsiveness
        chunk_size = min(chunk_size, 20)
        
        return chunk_size


    def generate_volume_parallel(self, vol, image_manager, max_workers, optimal_chunk_size, shape, input, progress_dir):
        """
        Generate volume in parallel with dual progress bars
        """
        total_slices = shape[2]
        chunk_size = optimal_chunk_size
        total_chunks = (total_slices + chunk_size - 1) // chunk_size
        
        print(f"Total slices: {total_slices}, Chunk size: {chunk_size}, Total chunks: {total_chunks}")
        
        # Main progress bar for chunks
        with tqdm(total=total_slices, desc="Overall progress", unit="slice") as pbar:
            for start_z in range(0, total_slices, chunk_size):
                end_z = min(start_z + chunk_size, total_slices)
                current_chunk_size = end_z - start_z
                
                print(f"\nProcessing chunk {start_z//chunk_size + 1}/{total_chunks} (slices {start_z}-{end_z-1})")
                
                self.process_chunk_with_memory_management(
                    vol, image_manager, start_z, end_z, max_workers, input, progress_dir
                )
                
                # Update overall progress
                pbar.update(current_chunk_size)
                
                # Show memory usage
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                print(f"Memory usage: {memory_usage:.1f} MB")


    def print_memory_usage(self, stage):
        """
        Print current memory usage
        """
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 ** 2)
        print(f"Memory usage {stage}: {memory_mb:.1f} MB")


    def process_chunk_with_memory_management(self, vol, image_manager, start_z, end_z, max_workers, input, progress_dir):
        """
        Process a chunk of slices with careful memory management
        """
        chunk_size = end_z - start_z
        
        # Pre-allocate memory for the entire chunk
        chunk_shape = (image_manager.num_channels, image_manager.height, 
                    image_manager.width, chunk_size)
        chunk_data = np.zeros(chunk_shape, dtype=image_manager.dtype)
        
        # Load slices into pre-allocated array
        self.load_slices_into_preallocated(start_z, end_z, chunk_data, max_workers, input, progress_dir)
        
        # Write the entire chunk to CloudVolume [with transpose]
        # chunk_data shape: [Channels, Height, Width, Z_slices]
        # neuroglancer shape: [Height, Width, Z_slices, Channels]
        transposed_chunk = chunk_data.transpose(1, 2, 3, 0)  # [H, W, Z, C]

        # Swap height and width dimensions if needed
        if transposed_chunk.shape[0] != vol.shape[0] or transposed_chunk.shape[1] != vol.shape[1]:
            transposed_chunk = transposed_chunk.transpose(1, 0, 2, 3)  # Swap H and W
        
        vol[:, :, start_z:end_z] = transposed_chunk
        
        # Explicitly free memory
        del chunk_data


    def load_slices_into_preallocated(self, start_z, end_z, chunk_data, max_workers, input, progress_dir):
        """
        Load slices directly into pre-allocated memory using threads
        """
        z_indices = list(range(start_z, end_z))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for i, z in enumerate(z_indices):
                future = executor.submit(
                    self.load_slice_to_position, z, chunk_data, i, input, progress_dir
                )
                futures.append(future)
            
            # Wait for all loads to complete
            for future in tqdm(as_completed(futures), total=len(futures), 
                            desc="Loading slices"):
                future.result()


    def load_slice_to_position(self, z, chunk_data, position, input, progress_dir):
        """
        Load a single slice into a specific position in the pre-allocated array
        Note: 'slice' is synonymous with single 2D image in the stack
        """
        try:
            file_path = Path(input, f"{z:03d}.tif")
            slice_data = read_image(file_path)

            # Validate shapes match expected chunk dimensions
            expected_height, expected_width = chunk_data.shape[1:3]
            
            if slice_data.shape[0] != expected_height or slice_data.shape[1] != expected_width:
                print(f"Shape mismatch for {file_path}:")
                print(f"  Expected: ({expected_height}, {expected_width})")
                print(f"  Got: {slice_data.shape}")
                # Optionally resize or skip
                return False
            
            # Handle different image formats
            if slice_data.ndim == 2:  # Grayscale → add channel dimension
                if chunk_data.shape[0] == 1:
                    chunk_data[0, :, :, position] = slice_data
                else:
                    print(f"Channel mismatch: grayscale image but chunk expects {chunk_data.shape[0]} channels")
                    return False
                    
            elif slice_data.ndim == 3:  # Multi-channel
                num_slice_channels = slice_data.shape[2]
                num_chunk_channels = chunk_data.shape[0]
                
                if num_slice_channels == num_chunk_channels:
                    chunk_data[:, :, :, position] = slice_data.transpose(2, 0, 1)
                else:
                    # Handle channel count mismatch
                    num_channels = min(num_slice_channels, num_chunk_channels)
                    chunk_data[:num_channels, :, :, position] = slice_data.transpose(2, 0, 1)[:num_channels]
                    print(f"Channel count adjusted: {num_slice_channels} → {num_channels}")
            
            progress_marker = Path(progress_dir, f"section_{z:06d}_add")
            progress_marker.touch(exist_ok=True)    

            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return False