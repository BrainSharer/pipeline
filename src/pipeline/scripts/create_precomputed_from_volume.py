"""
Creates a 3D Mesh
"""
import numpy as np
import argparse
import os
import sys
from PIL import Image

import SimpleITK as sitk
Image.MAX_IMAGE_PIXELS = None
from pathlib import Path
from taskqueue import LocalTaskQueue
import igneous.task_creation as tc
import nrrd
import json
import shutil
import os
import glob
import numpy as np
import tifffile as tiff

from cloudvolume import CloudVolume
import igneous.task_creation as tc

from taskqueue.taskqueue import LocalTaskQueue

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.image_manipulation.filelocation_manager import FileLocationManager
from library.image_manipulation.neuroglancer_manager import NumpyToNeuroglancer
from library.utilities.utilities_process import read_image
from library.controller.sql_controller import SqlController

OUT_VOL = "file:///net/birdstore/Active_Atlas_Data/data_root/pipeline_data/Allen/www/neuroglancer_data/sagittal_annotation"

if os.path.exists(OUT_VOL.replace("file://", "")):
    print(f"Removing and creating output directory: {OUT_VOL}")
    shutil.rmtree(OUT_VOL.replace("file://", ""))
os.makedirs(OUT_VOL.replace("file://", ""))

um2nanom = 1000.0
VOXEL_SIZE = (int(10 * um2nanom), int(10 * um2nanom), int(10 * um2nanom))  # in nanometers
CHUNK_SIZE = (64, 64, 64)
DOWNSAMPLE_LEVELS = 3
NUM_MIPS = 5               # resolution pyramid depth
SEG_DTYPE = np.uint32

def fetch_volume(volume_file):
    fileLocationManager = FileLocationManager(animal)
    volumepath = os.path.join(fileLocationManager.prep, volume_file)
    if not os.path.exists(volumepath):
        print(f'{volumepath} does not exist, exiting.')
        sys.exit()
    else:
        print(f'Got volume from {volumepath}')
        
    outpath = os.path.basename(volumepath)
    outpath = outpath.split('.')[0]
    ext = outpath.split('.')[0]

    if ext == 'nrrd':
        volume, _ = nrrd.read(volumepath)
    elif ext == 'nii':
        image = sitk.ReadImage(volumepath)
        volume = sitk.GetArrayFromImage(image)
    else:
        volume = read_image(volumepath)
    volume = np.swapaxes(volume, 0, 2)
    return volume


def create_precomputed(animal, volume_file, scaling_factor):
    chunk = 64
    chunks = (chunk, chunk, chunk)
    fileLocationManager = FileLocationManager(animal)
    sqlController = SqlController(animal)
    fileLocationManager = FileLocationManager(animal)
    xy = sqlController.scan_run.resolution * 1000
    z = sqlController.scan_run.zresolution * 1000
    scales = (int(xy*scaling_factor), int(xy*scaling_factor), int(z))
    print(f'scales={scales}')
    volumepath = os.path.join(fileLocationManager.prep, volume_file)
    if not os.path.exists(volumepath):
        print(f'{volumepath} does not exist, exiting.')
        sys.exit()
        
    outpath = os.path.basename(volume_file)
    outpath = outpath.split('.')[0]
    ext = outpath.split('.')[0]
    IMAGE_OUTPUT = os.path.join(fileLocationManager.neuroglancer_data, f'{outpath}')

    os.makedirs(IMAGE_OUTPUT, exist_ok=True)

    if ext == 'nrrd':
        volume, _ = nrrd.read(volume_file)
    elif ext == 'nii':
        image = sitk.ReadImage(volumepath)
        volume = sitk.GetArrayFromImage(image)
    else:
        volume = read_image(volumepath)
    volume = np.swapaxes(volume, 0, 2)
    num_channels = 1
    volume_size = volume.shape
    #volume = normalize16(volume)
    print(f'volume shape={volume.shape} dtype={volume.dtype}')

    ng = NumpyToNeuroglancer(
        animal,
        None,
        scales,
        "image",
        volume.dtype,
        num_channels=num_channels,
        chunk_size=chunks,
    )

    ng.init_precomputed(IMAGE_OUTPUT, volume_size)
    ng.precomputed_vol[:, :, :] = volume
    ng.precomputed_vol.cache.flush()
    tq = LocalTaskQueue(parallel=4)
    cloudpath = f"file://{IMAGE_OUTPUT}"
    tasks = tc.create_downsampling_tasks(cloudpath, num_mips=5)
    tq.insert(tasks)
    tq.execute()
    print("Done!")



def volume2mesh(animal, volume_file, scaling_factor):
    chunk = 64
    chunks = (chunk, chunk, chunk)
    fileLocationManager = FileLocationManager(animal)
    sqlController = SqlController(animal)
    fileLocationManager = FileLocationManager(animal)
    xy = sqlController.scan_run.resolution * 1000
    z = sqlController.scan_run.zresolution * 1000
    scales = (int(xy*scaling_factor), int(xy*scaling_factor), int(z))
    print(f'scales={scales}')
    volumepath = os.path.join(fileLocationManager.prep, volume_file)
    if not os.path.exists(volumepath):
        print(f'{volumepath} does not exist, exiting.')
        sys.exit()
        
    outpath = os.path.basename(volume_file)
    outpath = outpath.split('.')[0]
    ext = outpath.split('.')[0]
    IMAGE_OUTPUT = os.path.join(fileLocationManager.neuroglancer_data, f'{outpath}')

    os.makedirs(IMAGE_OUTPUT, exist_ok=True)

    if ext == 'nrrd':
        volume, _ = nrrd.read(volume_file)
    elif ext == 'nii':
        image = sitk.ReadImage(volumepath)
        volume = sitk.GetArrayFromImage(image)
    else:
        volume = read_image(volumepath)
    volume = np.swapaxes(volume, 0, 2)
    num_channels = 1
    volume_size = volume.shape
    #volume = normalize16(volume)
    print(f'volume shape={volume.shape} dtype={volume.dtype}')

    ng = NumpyToNeuroglancer(
        animal,
        None,
        scales,
        "image",
        volume.dtype,
        num_channels=num_channels,
        chunk_size=chunks,
    )

    ng.init_precomputed(IMAGE_OUTPUT, volume_size)
    ng.precomputed_vol[:, :, :] = volume
    ng.precomputed_vol.cache.flush()
    tq = LocalTaskQueue(parallel=4)
    cloudpath = f"file://{IMAGE_OUTPUT}"
    tasks = tc.create_downsampling_tasks(cloudpath, num_mips=5)
    tq.insert(tasks)
    tq.execute()
    print("Done!")

# -----------------------------
# CREATE NEUROGLANCER VOLUME
# -----------------------------
def create_volume(volume_shape):

    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type="segmentation",  # 'image' or 'segmentation'
        data_type=SEG_DTYPE,  #
        encoding='raw',  # other options: 'jpeg', 'compressed_segmentation' (req. uint32 or uint64)
        resolution=VOXEL_SIZE,  # Size of X,Y,Z pixels in nanometers,
        voxel_offset = [0,0,0],  # values X,Y,Z values in voxels
        chunk_size = CHUNK_SIZE,  # rechunk of image X,Y,Z in voxels
        volume_size = volume_shape,  # X,Y,Z size in voxels
    )
    vol = CloudVolume(OUT_VOL, info=info, compress=False, progress=True)
    vol.commit_info()
    print('Create cloudvolume')
    return vol

# -----------------------------
# DOWNSAMPLE SEGMENTATION PYRAMID
# -----------------------------
def build_mips():
    tq = LocalTaskQueue(parallel=8)

    tasks = tc.create_downsampling_tasks(
        layer_path=OUT_VOL,
        num_mips=NUM_MIPS
    )

    tq.insert(tasks)
    tq.execute()
    print('Built mips')


# -----------------------------
# BUILD MULTI-RES MESH PYRAMID
# -----------------------------
def build_meshes(ids):
    print(f'Building meshes with {len(ids)} ids.')
    tq = LocalTaskQueue(parallel=8)

    tasks = tc.create_meshing_tasks(
        layer_path=OUT_VOL,
        mip=0,
        max_simplification_error=40,  # microns, tune for vessels
        mesh_dir="meshes"
    )
    tq.insert(tasks)
    tq.execute()

    tasks = tc.create_mesh_manifest_tasks(OUT_VOL, mesh_dir="meshes") # Second Pass
    tq.insert(tasks)    
    tq.execute()

    tasks = tc.create_unsharded_multires_mesh_tasks(OUT_VOL, num_lod=3, mesh_dir="meshes")
    tq.insert(tasks)    
    tq.execute()

    cloud_volume = CloudVolume(OUT_VOL, 0)
    cloud_volume.info['segment_properties'] = 'names'
    cloud_volume.commit_info()
    segment_properties = {str(k): str(v) for k,v in ids.items()}


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

def create_ids():
    datapath = "allen_ids.json"
    if not os.path.exists(datapath):
        print(f'Cannot fine the allen ids')
        exit(0)
    
    with open(datapath, "r") as filename:
        allen_ids = json.load(filename)

    return allen_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=True, type=str)
    parser.add_argument('--volume', help='Enter the name of the volume file', required=True, type=str)
    parser.add_argument('--scaling_factor', help='Enter the scaling factor', required=False, default=1.0,  type=float)
    args = parser.parse_args()
    animal = args.animal
    volume = args.volume
    scaling_factor = args.scaling_factor
    ids = create_ids()
    
    #create_precomputed(animal, volume, scaling_factor)
    volume = fetch_volume(volume)
    vol = create_volume(volume.shape)
    vol[:,:,:] = volume

    build_mips()
    build_meshes(ids)

    print("✔ Neuroglancer multi-resolution segmentation mesh complete")


