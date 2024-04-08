"""
Creates a 3D Mesh
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from taskqueue.taskqueue import LocalTaskQueue, TaskQueue
import igneous.task_creation as tc
from cloudvolume import CloudVolume
import numpy as np
# np.seterr(all=None, divide=None, over=None, under=None, invalid=None)
np.seterr(all="ignore")
from pathlib import Path
from timeit import default_timer as timer

import faulthandler
import signal
faulthandler.register(signal.SIGUSR1.value)
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

    def __init__(self, animal, scaling_factor, debug):

        self.animal = animal
        self.scaling_factor = scaling_factor
        self.debug = debug
        self.sqlController = SqlController(animal)
        self.fileLocationManager = FileLocationManager(animal)
        self.mips = [0,1,2]
        self.mesh_mip = 1
        self.max_simplification_error = 50
        xy = self.sqlController.scan_run.resolution * 1000
        z = self.sqlController.scan_run.zresolution * 1000
        xy *=  self.scaling_factor
        z *= self.scaling_factor
        self.scales = (int(xy), int(xy), int(z))
        # start with big size chunks to cut down on the number of files created
        self.chunk = 512
        self.chunks = (self.chunk, self.chunk, 1)
        self.volume_size = 0
        self.ng = NumpyToNeuroglancer(self.animal, None, self.scales, layer_type='segmentation', 
            data_type=MESHDTYPE, chunk_size=self.chunks)

        # dirs
        self.mesh_dir = os.path.join(self.fileLocationManager.neuroglancer_data, f'mesh_{scaling_factor}')
        self.layer_path = f'file://{self.mesh_dir}'
        self.mesh_input_dir = os.path.join(self.fileLocationManager.neuroglancer_data, f'mesh_input_{self.scaling_factor}')
        self.progress_dir = os.path.join(self.fileLocationManager.neuroglancer_data, 'progress', f'mesh_{self.scaling_factor}')
        self.input = os.path.join(self.fileLocationManager.prep, 'C1', 'full')
        # make dirs
        os.makedirs(self.mesh_input_dir, exist_ok=True)
        os.makedirs(self.progress_dir, exist_ok=True)
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
        midfile = midfile.astype(MESHDTYPE)
        ids, counts = np.unique(midfile, return_counts=True)
        self.ids = ids.tolist()
        height, width = midfile.shape
        self.volume_size = (width//self.scaling_factor, height//self.scaling_factor, len_files // self.scaling_factor) # neuroglancer is width, height
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

    def process_stack(self):
        len_files = len(self.files)
        if limit > 0:
            _start = self.midpoint - limit
            _end = self.midpoint + limit
            files = files[_start:_end]
            len_files = len(files)

        print(f'\nMidfile: dtype={self.midfile.dtype}, shape={self.midfile.shape}, ids={self.ids}, counts={self.counts}')
        print(f'Scaling factor={scaling_factor}, volume size={self.volume_size} with dtype={MESHDTYPE}, scales={self.scales}')
        print(f'Initial chunks at {self.chunks} and chunks for downsampling=({self.chunk},{self.chunk},{self.chunk})\n')

        self.ng.init_precomputed(self.mesh_input_dir, self.volume_size)

        file_keys = []
        index = 0
        for i in range(0, len_files, scaling_factor):
            if index == len_files // scaling_factor:
                print(f'breaking at index={index}')
                break
            infile = os.path.join(self.input, self.files[i])            
            file_keys.append([index, infile, (self.volume_size[1], self.volume_size[0]), self.progress_dir, self.scaling_factor])
            index += 1

        _, cpus = get_cpus()
        print(f'Working on {len(file_keys)} files with {cpus} cpus')
        with ProcessPoolExecutor(max_workers=cpus) as executor:
            executor.map(self.ng.process_image_mesh, sorted(file_keys), chunksize=1)
            executor.shutdown(wait=True)

    def process_transfer(self):
        ###### start cloudvolume tasks #####
        # This calls the igneous create_transfer_tasks
        # the input dir is now read and the rechunks are created in the final dir
        _, cpus = get_cpus()
        self.ng.init_precomputed(self.mesh_input_dir, self.volume_size)
        # reset chunks to much smaller size for better neuroglancer experience
        chunks = [64, 64, 64]
        tq = LocalTaskQueue(parallel=cpus)
        os.makedirs(self.mesh_dir, exist_ok=True)
        if not os.path.exists(self.transfered_path):
            tasks = tc.create_image_shard_transfer_tasks(self.ng.precomputed_vol.layer_cloudpath, 
                                                            self.layer_path, mip=0, 
                                                            chunk_size=chunks)

            print(f'Creating transfer tasks in {self.transfered_path} with shards and chunks={chunks}')
            tq.insert(tasks)
            tq.execute()
        else:
            print(f'Already created transfer tasks in {self.transfered_path} with shards and chunks={chunks}')

        factors = [2,2,1]
        for mip in self.mips:
            xm,ym,zm = [ self.xs * (factors[0] ** (mip+1)), self.ys * (factors[1] ** (mip+1)), self.zs * (factors[2] ** (mip+1))]
            downsampled_path = os.path.join(self.mesh_dir, "_".join([str(xm), str(ym), str(zm)]))
            print(downsampled_path)
            if not os.path.exists(downsampled_path):
                print(f'Creating (rechunking) at mip={mip} with shards, chunks={chunks}, and factors={factors} in {downsampled_path}')

                tasks = tc.create_image_shard_downsample_tasks(self.layer_path, mip=mip, factor=factors, chunk_size=chunks)

                tq.insert(tasks)
                tq.execute()
            else:
                print(f'Already created (rechunking) at mip={mip} with shards, chunks={chunks} and factors={factors} in {downsampled_path}')

    def process_mesh(self):
        _, cpus = get_cpus()
        self.ng.init_precomputed(self.mesh_input_dir, self.volume_size)
        tq = LocalTaskQueue(parallel=cpus)

        if not os.path.exists(self.transfered_path):
            print('You need to run previous tasks first')
            print(f'Missing {self.transfered_path}')
            sys.exit()

        ##### add segment properties
        cloudpath = CloudVolume(self.layer_path, self.mesh_mip)
        downsample_path = cloudpath.meta.info['scales'][self.mesh_mip]['key']
        print(f'Creating mesh from {downsample_path}', end=" ")
        segment_properties = {str(id): str(id) for id in self.ids}

        print('and creating segment properties', end=" ")
        self.ng.add_segment_properties(cloudpath, segment_properties)
        ##### first mesh task, create meshing tasks
        #####ng.add_segmentation_mesh(cloudpath.layer_cloudpath, mip=0)
        # shape is important! the default is 448 and for some reason that prevents the 0.shard from being created at certain scales.
        # removing shape results in no 0.shard being created!!!
        # at scale=5, shape=128 did not work but 128*2 did
        # larger shape results in less files

        s = int(64*1)
        shape = [s, s, s]
        print(f'and mesh with shape={shape} at mip={self.mesh_mip} without shards')
        tasks = tc.create_meshing_tasks(self.layer_path, mip=self.mesh_mip, 
                                        shape=shape, 
                                        compress=True, 
                                        sharded=False,
                                        max_simplification_error=50) # The first phase of creating mesh
        tq.insert(tasks)
        tq.execute()

        # for apache to serve shards, this command: curl -I --head --header "Range: bytes=50-60" https://activebrainatlas.ucsd.edu/index.html
        # must return HTTP/1.1 206 Partial Content
        # a magnitude < 3 is more suitable for local mesh creation. Bigger values are for horizontal scaling in the cloud.

        print(f'Creating meshing manifest tasks with {cpus} CPUs')
        tasks = tc.create_mesh_manifest_tasks(self.layer_path) # The second phase of creating mesh
        tq.insert(tasks)
        tq.execute()

        #self.ng.precomputed_vol.commit_info()
        #self.ng.precomputed_vol.commit_provenance()


    def process_multires_mesh(self):
        _, cpus = get_cpus()
        self.ng.init_precomputed(self.mesh_input_dir, self.volume_size)
        tq = LocalTaskQueue(parallel=cpus)

        # Now do the mesh creation
        if not os.path.exists(self.transfered_path):
            print('You need to run previous tasks first')
            print(f'Missing {self.transfered_path}')
            sys.exit()

        LOD = 10
        print(f'Creating unsharded multires task with LOD={LOD}')

        tasks = tc.create_unsharded_multires_mesh_tasks(self.layer_path, num_lod=LOD)
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
        dothis = ""
        section_count = len(self.files) // self.scaling_factor
        processed_count = len(os.listdir(self.progress_dir))
        if section_count != processed_count and section_count > 0:
            dothis = "File count does not equal processed file count\n"
        mesh_path = os.path.join(self.mesh_dir, f'mesh_mip_{self.mesh_mip}_err_{self.max_simplification_error}')


        directories = {
            self.progress_dir: f"\nRun stack",
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

        if os.path.exists(result1) and os.path.exists(result2):
            print(f'Mesh creation is complete for animal={self.animal} at scale={scaling_factor}')
        else:
            print('Mesh is not complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=True)
    parser.add_argument('--limit', help='Enter the # of files to test', required=False, default=0)
    parser.add_argument('--scaling_factor', help='Enter an integer that will be the denominator', required=False, default=1)
    parser.add_argument("--skeleton", help="Create skeletons", required=False, default=False)
    parser.add_argument("--debug", help="debug", required=False, default=False)
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
    scaling_factor = int(args.scaling_factor)
    skeleton = bool({"true": True, "false": False}[str(args.skeleton).lower()])
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    task = str(args.task).strip().lower()
    
    pipeline = MeshPipeline(animal, scaling_factor, debug=debug)

    function_mapping = {
        "stack": pipeline.process_stack,
        "transfer": pipeline.process_transfer,
        "mesh": pipeline.process_mesh,
        "multi": pipeline.process_multires_mesh,
        "skeleton": pipeline.process_skeleton,
        "status": pipeline.check_status,
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
