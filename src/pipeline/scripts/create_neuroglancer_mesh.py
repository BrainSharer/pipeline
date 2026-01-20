"""
Creates a 3D Mesh
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys
from PIL import Image
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = None
from taskqueue.taskqueue import LocalTaskQueue
import igneous.task_creation as tc
from cloudvolume import CloudVolume
import numpy as np
from pathlib import Path
from timeit import default_timer as timer

import dask.array as da
from dask import delayed
from dask.diagnostics import ProgressBar
import tifffile as tiff
from scipy.ndimage import label
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_objects

"""
use:
kill -s SIGUSR1 <pid> 
This will give you a stacktrace of the running process and you can see where it hangs.
"""

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.controller.sql_controller import SqlController
from library.image_manipulation.filelocation_manager import FileLocationManager
from library.image_manipulation.neuroglancer_manager import NumpyToNeuroglancer, MESHDTYPE
from library.utilities.utilities_process import get_cpus


class MeshPipeline():

    def __init__(self, animal, scale, mip, debug):

        self.animal = animal
        self.scale = scale
        self.mip = mip
        self.debug = debug
        self.sqlController = SqlController(animal)
        self.fileLocationManager = FileLocationManager(animal)
        if self.scale > 1:
            self.mips = [0, 1, 2]
        else:
            self.mips = [0, 1, 2, 3, 4, 5]
        self.max_simplification_error = 40
        xy = self.sqlController.scan_run.resolution * 1000
        z = self.sqlController.scan_run.zresolution * 1000
        xy *=  self.scale
        z *= self.scale
        self.scales = (int(xy), int(xy), int(z))
        self.chunk = 64
        self.chunks = (self.chunk, self.chunk, 1)
        self.volume_size = 0
        self.dtype = np.uint8
        self.encoding = 'raw'
        self.ng = NumpyToNeuroglancer(self.animal, None, self.scales, layer_type='segmentation', 
            data_type=self.dtype, chunk_size=self.chunks)

        # dirs
        self.mesh_dir = os.path.join(self.fileLocationManager.neuroglancer_data, f'mesh_{scale}')
        self.layer_path = f'file://{self.mesh_dir}'
        self.mesh_input_dir = os.path.join(self.fileLocationManager.neuroglancer_data, f'mesh_input_{self.scale}')
        self.mesh_path = f'mesh_mip_{self.mip}_err_{self.max_simplification_error}'

        self.progress_dir = os.path.join(self.fileLocationManager.neuroglancer_data, 'progress', f'mesh_{self.scale}')
        self.input = os.path.join(self.fileLocationManager.prep, 'C1', 'full_aligned')
        self.output = os.path.join(self.fileLocationManager.prep, 'C1', f'downsampled_{self.scale}')
        _, self.cpus = get_cpus()
        
        # setup
        self.get_stack_info()

    def get_stack_info(self):
        files = sorted(os.listdir(self.input))
        len_files = len(files)
        midpoint = len_files // 2
        infile = os.path.join(self.input, files[midpoint])
        midim = Image.open(infile)
        midfile = np.array(midim)
        del midim
        #midfile = midfile.astype(self.dtype)
        ids, counts = np.unique(midfile, return_counts=True)
        print(f'Midfile: {infile}, dtype={midfile.dtype}, shape={midfile.shape}, ids={ids}, counts={counts}')
        height, width = midfile.shape
        self.volume_size = (width//self.scale, height//self.scale, len_files // self.scale) # neuroglancer is width, height
        self.files = files
        self.midpoint = midpoint
        self.midfile = midfile
        self.ids = ids
        self.counts = counts
        self.xs = self.scales[0]
        self.ys = self.scales[1]
        self.zs = self.scales[2]
        scale_dir = "_".join([str(self.xs), str(self.ys), str(self.zs)])
        self.transfered_path = os.path.join(self.mesh_dir, scale_dir)

    def create_volume(self):
        """creating with all x=518, y=394, z = 665"""
        len_files = len(self.files)
        if os.path.exists(self.output) and len(os.listdir(self.output)) > 0:
            print(f'Directory {self.output} already exists and is not empty, skipping creation of downsampled volume')
            return

        os.makedirs(self.output, exist_ok=True)
        print(f'Creating downsampled volume from {self.input}')
        print(f'\tat {self.output}')
        print(f'\twith scale={self.scale}')
        

        index = 0
        for i in tqdm(range(0, len_files, self.scale), desc="Creating downsampled volume"):
            if index == len_files // self.scale:
                print(f'breaking at index={index}')
                break
            infile = os.path.join(self.input, self.files[i]) 
            outfile = os.path.join(self.output, self.files[i])
            if infile.endswith('.tif') is False:
                print(f'Skipping non-tiff file: {infile}')
                continue
            im = Image.open(infile)           
            width, height = im.size
            im = im.resize((width//self.scale, height//self.scale))
            farr = np.array(im).astype('bool')
            tiff.imwrite(outfile, farr)
        print(f'Created downsampled volume at {self.output}')

    def process_distance(self):
        def block_distance_transform(block):
            return distance_transform_edt(block)
             
        def radius_to_labels(radius):
            labels = np.zeros_like(radius, dtype=self.dtype)
            labels[(radius >= 1) & (radius < 3)] = 1
            labels[(radius >= 3) & (radius < 6)] = 2
            labels[(radius >= 6) & (radius < 12)] = 3
            labels[(radius >= 12) & (radius < 25)] = 4
            labels[radius >= 25] = 5

            return labels

        def cleanup(block):
            out = np.zeros_like(block)
            for l in np.unique(block):
                if l == 0:
                    continue
                mask = block == l
                mask = remove_small_objects(mask)
                out[mask] = l
            return out

        def load_tiff_stack_dask(tiff_dir, chunk_shape):
            filenames = [os.path.join(tiff_dir, f) for f in os.listdir(tiff_dir) if f.endswith('.tif')]
            filenames.sort() # sort to ensure the correct order in the stack         
            if not filenames:
                raise ValueError("No TIFF files found in the directory.")

            # Read one file to get metadata (this runs immediately)
            sample_image = tiff.imread(filenames[0])
            image_shape = sample_image.shape
            image_dtype = sample_image.dtype
            del sample_image # Free memory from the sample 
            # Wrap the imread function with dask.delayed
            lazy_imread = delayed(tiff.imread)
            # Create a list of delayed objects, one for each image file
            lazy_arrays = [lazy_imread(image) for image in filenames]
            # For each delayed object, create a Dask array with the known shape and dtype
            dask_arrays = [da.from_delayed(delayed_reader, shape=image_shape, dtype=image_dtype) for delayed_reader in tqdm(lazy_arrays)]
            # Stack along a new dimension (e.g., time or z-axis)
            image_stack = da.stack(dask_arrays, axis=0)
            image_stack = image_stack.rechunk(chunk_shape)
            # dask array is in z,y,x but neuroglancer wants x,y,z
            image_stack = da.swapaxes(image_stack, 0, 2)
            print(f"Dask array created with shape: {image_stack.shape}, dtype: {image_stack.dtype}")
            return image_stack

        def write_block(block, block_info=None):
            """
            block is in shape z,y,x
            """
            if block_info and 'array-location' in block_info[None]:
                global_slices = block_info[None]['array-location']        
                # Extract the start coordinates (z, y, x)
                z0, y0, x0 = [x[0] for x in global_slices]
                z1, y1, x1 = [x[1] for x in global_slices]
                
                try:
                    vol[z0:z1, y0:y1, x0:x1] = block
                except Exception as e:
                    print(f'Error writing block at z:{z0}-{z1}, y:{y0}-{y1}, x:{x0}-{x1}: {e}')
                    exit(1)

        self.create_volume()
        TIFF_DIR = self.output
        CHUNK_SHAPE = (self.chunk, self.chunk, self.chunk)
        # Isotropic voxel size (µm)
        VOXEL_SIZE_UM = self.scale  # change to 2.0, 4.0, etc. if needed
        print(f'Processing distance transform on volume in {TIFF_DIR} with scale={self.scale} um')
        # Chunk size (optimize for memory)
        # Minimum connected component size (voxels)
        binary = load_tiff_stack_dask(TIFF_DIR, CHUNK_SHAPE)
        binary = binary > 0
        print(f'Binary volume shape={binary.shape}, dtype={binary.dtype} chunks={binary.chunksize}') 
        with ProgressBar():        
            distance = da.map_blocks(
                block_distance_transform,
                binary,
                dtype=np.float32
            )
        print(f'Distance volume shape={distance.shape}, dtype={distance.dtype} chunks={distance.chunksize}')
        radius_um = distance * VOXEL_SIZE_UM
        print(f'Radius volume shape={radius_um.shape}, dtype={radius_um.dtype} chunks={radius_um.chunksize}')
        with ProgressBar():        
            labels = da.map_blocks(
                radius_to_labels,
                radius_um,
                dtype=self.dtype
            )
        print(f'Labels volume shape={labels.shape}, dtype={labels.dtype} chunks={labels.chunksize}')
        with ProgressBar():        
            labels = da.map_blocks(cleanup, labels, dtype=self.dtype)
        ids = da.unique(labels, return_counts=False)
        with ProgressBar():
            self.ids = ids.compute()
        print(f'Cleaned Labels volume shape={labels.shape}, dtype={labels.dtype} chunks={labels.chunksize}')
        # volume now is at z,y,x
        print(f'Label IDs: {self.ids}')

        info = CloudVolume.create_new_info(
            num_channels=1,
            layer_type='segmentation',  # 'image' or 'segmentation'
            data_type=self.dtype,  #
            encoding=self.encoding,  # other options: 'jpeg', 'compressed_segmentation', 'raw' (req. uint32 or uint64)
            resolution=[VOXEL_SIZE_UM*1000]*3,  # Size of X,Y,Z pixels in nanometers,
            voxel_offset=[0,0,0],  # values X,Y,Z values in voxels
            chunk_size=CHUNK_SHAPE,  # rechunk of image X,Y,Z in voxels
            volume_size=[labels.shape[0], labels.shape[1], labels.shape[2]],  # z,x,y
        )
        vol = CloudVolume(self.layer_path, info=info, compress=True, progress=False)
        vol.commit_info()
        with ProgressBar():
            labels.map_blocks(write_block, dtype=self.dtype).compute()
        print(f'Wrote labels volume to {self.layer_path}')

        tasks = tc.create_downsampling_tasks(
            self.layer_path,
            num_mips=2,
            compress=True,
            chunk_size=[self.chunk, self.chunk, self.chunk]
        )
        tq = LocalTaskQueue(parallel=self.cpus)
        tq.insert(tasks)
        tq.execute()

        print('Finished downsampling tasks')
        cloud_volume = CloudVolume(self.layer_path, 0)
        cloud_volume.info['segment_properties'] = 'names'
        cloud_volume.commit_info()

        ids = self.ids.tolist()
        ids = [int(i) for i in ids if i > 0]
        segment_properties = {str(id): str(id) for id in ids}

        segment_properties_path = os.path.join(cloud_volume.layerpath.replace('file://', ''), 'names')
        os.makedirs(segment_properties_path, exist_ok=True)
        info = {
            "@type": "neuroglancer_segment_properties",
            "inline": {
                "ids": [str(number) for number, _ in segment_properties.items()],
                "properties": [{
                    "id": "label",
                    "type": "label",
                    "values": [str(label) for _, label in segment_properties.items()]
                }]
            }
        }
        with open(os.path.join(segment_properties_path, 'info'), 'w') as file:
            json.dump(info, file, indent=2)
        print(f'Wrote segment properties to {segment_properties_path}')
        return





        LOD = 5
        print(f'Creating sharded multires task with LOD={LOD} from {self.mesh_path}')
        tasks = tc.create_unsharded_multires_mesh_tasks(self.layer_path, num_lod=LOD, magnitude=2, mesh_dir=self.mesh_path)
        tq.insert(tasks)    
        tq.execute()


    def process_volume_distance(self):
        #len_files = len(self.files)
        self.create_volume()
        files = sorted(os.listdir(self.output))
        file_list = []
        for file in tqdm(files):
            infile = os.path.join(self.output, file) 
            im = Image.open(infile)           
            farr = np.array(im).astype(bool)
            file_list.append(farr)
            
        vol = np.stack(file_list, axis = 0)
        vol = np.swapaxes(vol, 0, 2)
        print(f'Volume shape={vol.shape}, dtype={vol.dtype}, min={vol.min()}, max={vol.max()}')
        self.volume_size = vol.shape
        cc, n = label(vol)

        cc = cc > 1

        sizes = np.bincount(cc.ravel())
        print(f'sizes len={len(sizes)} min={sizes.min()} max={sizes.max()} mean={sizes.mean()}')

        return
        labels = np.zeros_like(cc, dtype=np.self.dtype)

        for i in range(1, n + 1):
            if sizes[i] < 1e4:
                labels[cc == i] = 1
            elif sizes[i] < 1e6:
                labels[cc == i] = 2
            else:
                labels[cc == i] = 3        
        
        self.ids, counts = np.unique(labels, return_counts=True)
        print(f'Labels volume shape={labels.shape}, dtype={labels.dtype}, min={labels.min()}, max={labels.max()}')
        print(f'Label IDs: {self.ids}, counts: {counts}')
        self.ng.init_precomputed(self.mesh_input_dir, self.volume_size)
        self.ng.precomputed_vol[:, :, :] = labels
        self.process_transfer()
        self.process_mesh()


    def process_stack(self):
        # make dirs
        os.makedirs(self.mesh_input_dir, exist_ok=True)
        os.makedirs(self.progress_dir, exist_ok=True)
        len_files = len(self.files)
        if limit > 0:
            _start = self.midpoint - limit
            _end = self.midpoint + limit
            files = self.files[_start:_end]
            len_files = len(files)

        print(f'\nMidfile: dtype={self.midfile.dtype}, shape={self.midfile.shape}, ids={self.ids}, counts={self.counts}')
        print(f'Scaling factor={scale}, volume size={self.volume_size} with dtype={self.dtype}, scales={self.scales}')
        print(f'Initial chunks at {self.chunks} and chunks for downsampling=({self.chunk},{self.chunk},{self.chunk})\n')

        self.ng.init_precomputed(self.mesh_input_dir, self.volume_size, encoding=self.encoding)

        file_keys = []
        index = 0
        for i in range(0, len_files, self.scale):
            if index == len_files // self.scale:
                print(f'breaking at index={index}')
                break
            infile = os.path.join(self.input, self.files[i])            
            file_keys.append([index, infile, (self.volume_size[1], self.volume_size[0]), self.progress_dir, self.scale])
            index += 1

        
        print(f'Working on {len(file_keys)} files with {self.cpus} cpus')
        with ProcessPoolExecutor(max_workers=self.cpus) as executor:
            executor.map(self.ng.process_image_mesh, sorted(file_keys), chunksize=1)
            executor.shutdown(wait=True)



    def process_transfer(self):
        ###### start cloudvolume tasks #####
        # This calls the igneous create_transfer_tasks
        # the input dir is now read and the rechunks are created in the final dir
        self.ng.init_precomputed(self.mesh_input_dir, self.volume_size, encoding=self.encoding)
        # reset chunks to much smaller size for better neuroglancer experience
        chunks = [self.chunk, self.chunk, self.chunk]
        tq = LocalTaskQueue(parallel=self.cpus)
        os.makedirs(self.mesh_dir, exist_ok=True)
        if not os.path.exists(self.transfered_path):
            tasks = tc.create_transfer_tasks(
                self.ng.precomputed_vol.layer_cloudpath,
                self.layer_path,
                mip=0,
                max_mips=0,
                chunk_size=chunks
            )
            print(f'Creating transfer tasks in {self.transfered_path} with chunks={chunks}')
            tq.insert(tasks)
            tq.execute()
        else:
            print(f'Already created transfer tasks in {self.transfered_path} with chunks={chunks}')
        return

    def downsample_transfer(self):

        tq = LocalTaskQueue(parallel=self.cpus)
        chunks = [self.chunk, self.chunk, self.chunk]
        factors = [2,2,2]
        for mip in self.mips:
            tasks = tc.create_downsampling_tasks(
                self.layer_path, mip=mip,
                num_mips=1, factor=factors, chunk_size=chunks)
            tq.insert(tasks)
            tq.execute()


    def process_mesh(self):
        print(f'Creating mesh at mip={self.mip} with max_simplification_error={self.max_simplification_error}')
        print(f'Using mesh output dir: {self.mesh_dir}')
        print(f'Using layer path: {self.layer_path}')
        print(f'Using volume size: {self.volume_size}')
        print(f' using {self.cpus} CPUs')        

        tq = LocalTaskQueue(parallel=self.cpus)

        if not os.path.exists(str(self.layer_path).replace('file://', '')):
            print('You need to run previous tasks first')
            print(f'Missing {self.layer_path}')
            sys.exit()


        tasks = tc.create_meshing_tasks(self.layer_path, mip=self.mip, 
                                        max_simplification_error=self.max_simplification_error,
                                        compress=True, mesh_dir=self.mesh_path, progress=True, sharded=False)
        tq.insert(tasks)
        tq.execute()


        print('Creating meshing manifest tasks')
        tasks = tc.create_mesh_manifest_tasks(self.layer_path, mesh_dir=self.mesh_path) # The second phase of creating mesh
        tq.insert(tasks)
        tq.execute()

        #update mesh info
        cloud_volume = CloudVolume(self.layer_path, self.mip)
        cloud_volume.info['mesh'] = self.mesh_path
        cloud_volume.commit_info()


        segment_properties_path = os.path.join(cloud_volume.layerpath.replace('file://', ''), 'names')
        if os.path.exists(segment_properties_path):
            print(f'Segment properties already exist at {segment_properties_path}, skipping creation')
            return
        
        ids = [255]
        ids = [int(i) for i in ids if i > 0]
        segment_properties = {str(id): str(id) for id in ids}

        os.makedirs(segment_properties_path, exist_ok=True)
        info = {
            "@type": "neuroglancer_segment_properties",
            "inline": {
                "ids": [str(number) for number, _ in segment_properties.items()],
                "properties": [{
                    "id": "label",
                    "type": "label",
                    "values": [str(label) for _, label in segment_properties.items()]
                }]
            }
        }
        with open(os.path.join(segment_properties_path, 'info'), 'w') as file:
            json.dump(info, file, indent=2)
        print(f'Wrote segment properties to {segment_properties_path}')
        return





    def process_multires_mesh(self):
        """
        """
        self.ng.init_precomputed(self.mesh_input_dir, self.volume_size, encoding=self.encoding)
        tq = LocalTaskQueue(parallel=1)

        # Now do the mesh creation
        #if not os.path.exists(self.transfered_path):
        #    print('You need to run previous tasks first')
        #    print(f'Missing {self.transfered_path}')
        #    sys.exit()

        # LOD=0, resolution stays the same
        # LOD=10, resolution shows different detail
        LOD = 5
        print(f'Creating sharded multires task with LOD={LOD} from {self.mesh_path}')
        #tasks = tc.create_sharded_multires_mesh_tasks(self.layer_path, num_lod=LOD)
        tasks = tc.create_unsharded_multires_mesh_tasks(self.layer_path, num_lod=LOD, magnitude=2, mesh_dir=self.mesh_path, min_chunk_size=[self.chunk, self.chunk, self.chunk])
        tq.insert(tasks)    
        tq.execute()

    def process_skeleton(self):
        ##### skeleton
        print('Creating skeletons')
        tasks = tc.create_skeletonizing_tasks(self.layer_path, mip=0)
        tq = LocalTaskQueue(parallel=self.cpus)
        tq.insert(tasks)
        tasks = tc.create_unsharded_skeleton_merge_tasks(self.layer_path)
        tq.insert(tasks)
        tq.execute()

    def check_status(self):
        print(f'Checking mesh creation status for animal={self.animal} at scale={scale}')
        print(f'IDs in volume: {self.ids}')

        dothis = ""
        section_count = len(self.files) // self.scale
        if os.path.exists(self.progress_dir):
            processed_count = len(os.listdir(self.progress_dir))
            if section_count != processed_count and section_count > 0:
                dothis = f"File count={section_count} does not equal processed file count={processed_count}\n"

        mesh_path = os.path.join(self.mesh_dir, self.mesh_path)
        directories = {
            self.progress_dir: "\nRun stack",
            self.transfered_path: "\nRun transfer",
            mesh_path: "\nRun mesh",
        }

        for directory, message in directories.items():
            if not os.path.exists(directory) or len(os.listdir(directory)) == 0:
                directory = directory.split("/")
                directory = directory[-2:]
                dothis += f'{message} as {"/".join(directory)} is missing.'

        if len(dothis) > 0:
            print(dothis)

        result1 = os.path.join(mesh_path, '1')
        result2 = os.path.join(mesh_path, '1.index')
        result3 = os.path.join(mesh_path, '0.shard')

        if (os.path.exists(result1) and os.path.exists(result2)) or os.path.exists(result3):
            print(f'Mesh creation is complete for animal={self.animal} at scale={scale}')


    def run_all(self):
        self.process_stack()
        self.process_transfer()
        self.downsample_transfer()
        self.process_mesh()
        print('Finished running all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=True)
    parser.add_argument('--limit', help='Enter the # of files to test', required=False, default=0)
    parser.add_argument('--scale', help='Enter an integer that will be the denominator', required=False, default=1)
    parser.add_argument('--mip', help='Enter the mesh mip level', required=False, default=0)
    parser.add_argument("--skeleton", help="Create skeletons", required=False, default=False)
    parser.add_argument('--debug', help='debug', required=False, default=False)
    parser.add_argument('--shard', help='shard', required=False, default=False)
    parser.add_argument(
        "--task",
        help="Enter the task you want to perform: stack -> transfer -> mesh -> multi",
        required=False,
        default="status",
        type=str,
    )

    args = parser.parse_args()
    animal = args.animal
    limit = int(args.limit)
    scale = int(args.scale)
    mip = int(args.mip)
    skeleton = bool({"true": True, "false": False}[str(args.skeleton).lower()])
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    task = str(args.task).strip().lower()
    
    pipeline = MeshPipeline(animal, scale, mip, debug=debug)

    function_mapping = {
        "create_volume": pipeline.create_volume,
        "distance": pipeline.process_distance,
        "distance2": pipeline.process_volume_distance,
        "stack": pipeline.process_stack,
        "transfer": pipeline.process_transfer,
        "downsample": pipeline.downsample_transfer,
        "mesh": pipeline.process_mesh,
        "multi": pipeline.process_multires_mesh,
        "skeleton": pipeline.process_skeleton,
        "status": pipeline.check_status,
        "all": pipeline.run_all
    }

    if task in function_mapping:
        start_time = timer()
        function_mapping[task]()
        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        print(f"{task} took {total_elapsed_time} seconds")

    else:
        print(f"{task} is not a correct task. Choose one of these:")
        for key in function_mapping.keys():
            print(f"\t{key}")


