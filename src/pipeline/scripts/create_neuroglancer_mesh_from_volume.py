"""
Creates a 3D Mesh
"""
import argparse
import os
import sys
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from cloudvolume import CloudVolume
import shutil
import numpy as np
from pathlib import Path
import pandas as pd


PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.controller.sql_controller import SqlController

# from library.controller.sql_controller import SqlController
from library.image_manipulation.neuroglancer_manager import NumpyToNeuroglancer
from library.utilities.utilities_process import SCALING_FACTOR, get_hostname, read_image
DTYPE = np.uint64

def get_ids_from_csv(csvfile, ids):
    df = pd.read_csv(csvfile)
    print(df.head())
    for id in ids:
        pass 


def create_mesh(animal, filepath, csvfile=None):
    chunks = (64, 64, 64)
    sqlController = SqlController(animal)
    xy = sqlController.scan_run.resolution * 1000
    z = sqlController.scan_run.zresolution * 1000
    outpath = os.path.basename(filepath)
    outpath = outpath.split('.')[0]
    MESH_DIR = os.path.join('/var/www/brainsharer/structures', outpath)
    
    scales = (int(xy), int(xy), int(z))
    print(f'scales={scales}')
    
    if 'mothra' in get_hostname():
        print(f'Cleaning {MESH_DIR}')
        if os.path.exists(MESH_DIR):
            shutil.rmtree(MESH_DIR)


    os.makedirs(MESH_DIR, exist_ok=True)

    if not os.path.exists(filepath):
        print(f'File {filepath} does not exist')
        sys.exit(1)
    volume = read_image(filepath)
    
    ids, counts = np.unique(volume, return_counts=True)

    data_type = volume.dtype
    
    print()
    print(f'Volume: {filepath} dtype={data_type}, shape={volume.shape}')
    print(f'Initial chunks at {chunks} and chunks for downsampling={chunks} and scales with {scales}')
    print(f'Creating in {outpath}')
    print(f'IDS={len(ids)}')
    #print(f'counts={counts}')
    
    ng = NumpyToNeuroglancer(animal, volume, scales, layer_type='segmentation', 
        data_type=data_type, chunk_size=chunks)

    ng.init_volume(MESH_DIR)
    
    # This calls the igneous create_transfer_tasks
    #ng.add_rechunking(MESH_DIR, chunks=chunks, mip=0, skip_downsamples=True)

    #tq = LocalTaskQueue(parallel=4)
    cloudpath2 = f'file://{MESH_DIR}'
    #ng.add_downsampled_volumes(chunk_size = chunks, num_mips = 1)

    ##### add segment properties
    print('Adding segment properties')
    cv2 = CloudVolume(cloudpath2, 0)
    segment_properties = {str(id): str(id) for id in ids}
    ng.add_segment_properties(cv2, segment_properties)

    ##### first mesh task, create meshing tasks
    print(f'Creating meshing tasks on volume from {cloudpath2}')
    ##### first mesh task, create meshing tasks
    ng.add_segmentation_mesh(cv2.layer_cloudpath, mip=0)

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=True)
    parser.add_argument('--filepath', help='Enter the name of the volume file path', required=True)
    parser.add_argument('--csvfile', help='Enter the path of the csv file', required=False)
    args = parser.parse_args()
    animal = args.animal
    filepath = args.filepath
    csvfile = args.csvfile
    
    create_mesh(animal, filepath, csvfile)

