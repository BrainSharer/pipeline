"""
Creates a 3D Mesh
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from taskqueue.taskqueue import LocalTaskQueue
import igneous.task_creation as tc
from cloudvolume import CloudVolume
import shutil
import numpy as np
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
from pathlib import Path

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
from library.image_manipulation.neuroglancer_manager import NumpyToNeuroglancer
from library.utilities.utilities_process import get_cpus, get_hostname
DTYPE = np.uint64

def create_mesh(animal, limit, scaling_factor, skeleton, debug):
    if scaling_factor > 5:
        chunkXY = 64
    else:
        chunkXY = 128
    # chunkZ = chunkXY // 2
    chunkZ = chunkXY
    chunks = (chunkXY, chunkXY, 1)
    sqlController = SqlController(animal)
    fileLocationManager = FileLocationManager(animal)
    xy = sqlController.scan_run.resolution * 1000
    z = sqlController.scan_run.zresolution * 1000
    INPUT = os.path.join(fileLocationManager.prep, 'CH1', 'full')
    MESH_INPUT_DIR = os.path.join(fileLocationManager.neuroglancer_data, f'mesh_input_{scaling_factor}')
    MESH_DIR = os.path.join(fileLocationManager.neuroglancer_data, f'mesh_{scaling_factor}')
    PROGRESS_DIR = os.path.join(fileLocationManager.neuroglancer_data, 'progress', f'mesh_{scaling_factor}')
    
    xy *=  scaling_factor
    z *= scaling_factor

    scales = (int(xy), int(xy), int(z))
    if 'godzilla' in get_hostname():
        print(f'Cleaning {MESH_DIR}')
        if os.path.exists(MESH_DIR):
            shutil.rmtree(MESH_DIR)

    files = sorted(os.listdir(INPUT))

    os.makedirs(MESH_INPUT_DIR, exist_ok=True)
    os.makedirs(MESH_DIR, exist_ok=True)
    os.makedirs(PROGRESS_DIR, exist_ok=True)

    len_files = len(files)
    midpoint = len_files // 2
    infile = os.path.join(INPUT, files[midpoint])
    midim = Image.open(infile)
    midfile = np.array(midim)
    del midim
    midfile = midfile.astype(np.uint8)
    midfile[midfile > 0] = 255
    ids, counts = np.unique(midfile, return_counts=True)

    if limit > 0:
        _start = midpoint - limit
        _end = midpoint + limit
        files = files[_start:_end]
        len_files = len(files)
    
    height, width = midfile.shape
    volume_size = (width//scaling_factor, height//scaling_factor, len_files // scaling_factor) # neuroglancer is width, height
    print(f'\nMidfile: {infile} dtype={DTYPE}, shape={midfile.shape}, ids={ids}, counts={counts}')
    print(f'Scaling factor={scaling_factor}, volume size={volume_size} with dtype={data_type}, scales={scales}')
    print(f'Initial chunks at {chunks} and chunks for downsampling=({chunkXY},{chunkXY},{chunkZ})\n')
    
    
    ng = NumpyToNeuroglancer(animal, None, scales, layer_type='segmentation', 
        data_type=DTYPE, chunk_size=chunks)

    ng.init_precomputed(MESH_INPUT_DIR, volume_size)

    file_keys = []
    index = 0
    for i in range(0, len_files, scaling_factor):
        if index == len_files // scaling_factor:
            print(f'breaking at index={index}')
            break
        infile = os.path.join(INPUT, files[i])            
        file_keys.append([index, infile, (volume_size[1], volume_size[0]), PROGRESS_DIR, scaling_factor])
        index += 1


    _, cpus = get_cpus()
    print(f'Working on {len(file_keys)} files with {cpus} cpus')
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        executor.map(ng.process_image_mesh, sorted(file_keys), chunksize=1)
        executor.shutdown(wait=True)
    
    
    chunks = (chunkXY, chunkXY, chunkZ)
    # This calls the igneous create_transfer_tasks
    ng.add_rechunking(MESH_DIR, chunks=chunks, mip=0, skip_downsamples=True)

    tq = LocalTaskQueue(parallel=cpus)
    cloudpath2 = f'file://{MESH_DIR}'
    ng.add_downsampled_volumes(chunk_size = chunks, num_mips = 1)

    ##### add segment properties
    cv2 = CloudVolume(cloudpath2, 0)
    segment_properties = {str(id): str(id) for id in ids}
    ng.add_segment_properties(cv2, segment_properties)

    ##### first mesh task, create meshing tasks
    ng.add_segmentation_mesh(cv2.layer_cloudpath, mip=0)

    ##### skeleton
    if skeleton:
        print('Creating skeletons')
        tasks = tc.create_skeletonizing_tasks(cv2.layer_cloudpath, mip=0)
        tq.insert(tasks)
        tasks = tc.create_unsharded_skeleton_merge_tasks(cv2.layer_cloudpath)
        tq.insert(tasks)
        tq.execute()
    

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=True)
    parser.add_argument('--limit', help='Enter the # of files to test', required=False, default=0)
    parser.add_argument('--scaling_factor', help='Enter an integer that will be the denominator', required=False, default=1)
    parser.add_argument("--skeleton", help="Create skeletons", required=False, default=False)
    parser.add_argument("--debug", help="debug", required=False, default=False)
    args = parser.parse_args()
    animal = args.animal
    limit = int(args.limit)
    scaling_factor = int(args.scaling_factor)
    skeleton = bool({"true": True, "false": False}[str(args.skeleton).lower()])
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    
    create_mesh(animal, limit, scaling_factor, skeleton, debug)

