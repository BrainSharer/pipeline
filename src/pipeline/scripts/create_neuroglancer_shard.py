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
#np.seterr(all=None, divide=None, over=None, under=None, invalid=None)
np.seterr(all="ignore")
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
from library.image_manipulation.neuroglancer_manager import NumpyToNeuroglancer, calculate_chunks
from library.utilities.utilities_process import get_cpus, get_hostname

def create_shard(animal, debug):
    scaling_factor = 32
    sqlController = SqlController(animal)
    fileLocationManager = FileLocationManager(animal)
    xy = sqlController.scan_run.resolution * 1000
    z = sqlController.scan_run.zresolution * 1000
    INPUT = os.path.join(fileLocationManager.prep, 'CH1', 'thumbnail_aligned')
    RECHUNKME = os.path.join(fileLocationManager.neuroglancer_data, 'rechunkme')
    OUTPUT = os.path.join(fileLocationManager.neuroglancer_data, 'shard')
    PROGRESS_DIR = os.path.join(fileLocationManager.neuroglancer_data, 'progress', 'shard')
    
    xy *=  scaling_factor
    z

    scales = (int(xy), int(xy), int(z))
    if 'godzilla' in get_hostname():
        print(f'Cleaning {OUTPUT}')
        if os.path.exists(OUTPUT):
            shutil.rmtree(OUTPUT)

    files = sorted(os.listdir(INPUT))

    os.makedirs(PROGRESS_DIR, exist_ok=True)
    os.makedirs(RECHUNKME, exist_ok=True)

    len_files = len(files)
    midpoint = len_files // 2
    infile = os.path.join(INPUT, files[midpoint])
    midim = Image.open(infile)
    midfile = np.array(midim)
    del midim

    chunk = 64
    chunks = (chunk, chunk, 1)
    height, width = midfile.shape
    volume_size = (width, height, len_files) # neuroglancer is width, height
    print(f'\nMidfile: {infile} dtype={midfile.dtype}, shape={midfile.shape}')
    print(f'Scaling factor={scaling_factor}, volume size={volume_size} with dtype={midfile.dtype}, scales={scales}')
    print(f'Initial chunks at {chunks} and chunks for downsampling=({chunk},{chunk},{chunk})\n')
    ng = NumpyToNeuroglancer(animal, None, scales, layer_type='image', 
        data_type=midfile.dtype, num_channels=1, chunk_size=chunks)


    ng.init_precomputed(RECHUNKME, volume_size)
    
    file_keys = []
    # index, infile, orientation, progress_dir
    for index, file in enumerate(files):
        infile = os.path.join(INPUT, file)
        file_keys.append([index, infile, None, PROGRESS_DIR, False, 0, 0]) #added is_blank, height, width

    _, cpus = get_cpus()
    print(f'Working on {len(file_keys)} files with {cpus} cpus')
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        executor.map(ng.process_image, sorted(file_keys), chunksize=1)
        executor.shutdown(wait=True)

    
    ###### start cloudvolume tasks #####
    # This calls the igneous create_transfer_tasks
    # the input dir is now read and the rechunks are created in the final dir
    _, cpus = get_cpus()
    chunks = [chunk, chunk, chunk]
    tq = LocalTaskQueue(parallel=cpus)
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT, exist_ok=True)
        layer_path = f'file://{OUTPUT}'
        tasks = tc.create_image_shard_transfer_tasks(ng.precomputed_vol.layer_cloudpath, layer_path, mip=0, chunk_size=chunks)
        print(f'Creating transfer tasks (rechunking) with shards and chunks={chunks}')
        tq.insert(tasks)
        tq.execute()

    print(f'Creating downsamplings tasks (rechunking) with shards and with chunks={chunks}')
    for mip in [0, 1]:
        tasks = tc.create_image_shard_downsample_tasks(layer_path, mip=mip, chunk_size=chunks)
        tq.insert(tasks)
        tq.execute()

    

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=True)
    parser.add_argument("--debug", help="debug", required=False, default=False)
    args = parser.parse_args()
    animal = args.animal
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    
    create_shard(animal, debug)

