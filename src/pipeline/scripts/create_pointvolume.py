"""
This takes the coordinates and packs them into a binary file,
see https://github.com/google/neuroglancer/issues/227
Create a dir on birdstore called points
put the info file under points/info
create the binary file and put in points/spatial0/0_0_0
"""
import argparse
import json
import os
import struct
import sys
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import gzip
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox, Vec

from taskqueue import LocalTaskQueue
import igneous.task_creation as tc



PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())


from library.image_manipulation.filelocation_manager import FileLocationManager, data_path
from library.controller.sql_controller import SqlController
from library.image_manipulation.neuroglancer_manager import NumpyToNeuroglancer

def create_points(animal, scaling_factor=32, debug=False):
    fileLocationManager = FileLocationManager(animal)
    sqlController = SqlController(animal)
    coordinates = []
    session_id = 8061    
    fileLocationManager = FileLocationManager(animal)
    sqlController = SqlController(animal)
    w = sqlController.scan_run.width//scaling_factor
    h = sqlController.scan_run.height//scaling_factor
    z_length = 440

    polygons = sqlController.get_annotation_volume(session_id, scaling_factor=1)
    xyresolution = sqlController.scan_run.resolution / scaling_factor
    zresolution = sqlController.scan_run.zresolution
    print(f'{xyresolution=}, {zresolution=}')

    path = os.path.join(fileLocationManager.neuroglancer_data, 'predictions0')
    if os.path.exists(path):
        print(f'Removing existing directory {path}')
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    
    coordinates = []
    for section, points  in polygons.items():
        converted_points = [(p[0]/xyresolution, p[1]/xyresolution) for p in points]
        z = int(round(section / zresolution))
        if debug:
            print(f'{z=} {converted_points[0]=}')
        for point in converted_points:
            x, y = point
            coordinates.append([x, y, z])
    points = np.array(coordinates)

    print(f'length of coordinates: {len(coordinates)} {points.shape=} mean={np.mean(points, axis=0)}')
    
    spatial_dir = os.path.join(fileLocationManager.neuroglancer_data, 'predictions0', 'spatial0')
    info_dir = os.path.join(fileLocationManager.neuroglancer_data, 'predictions0')
    if os.path.exists(info_dir):
        print(f'Removing existing directory {info_dir}')
        shutil.rmtree(info_dir)
    os.makedirs(spatial_dir, exist_ok=True)
    point_filename = os.path.join(spatial_dir, '0_0_0.gz')
    info_filename = os.path.join(info_dir, 'info')
    """
    with open(point_filename, 'wb') as outfile:
        buf = struct.pack('<Q', len(coordinates))
        pt_buf = b''.join(struct.pack('<3f', x, y, z) for (x, y, z) in coordinates)
        buf += pt_buf
        id_buf = struct.pack('<%sQ' % len(coordinates), *range(len(coordinates)))
        #struct.pack('<4f4BH2B', x, y, z, cell_size, r, g, b, a, cell_type, 0, 0)
        buf += id_buf
        bufout = gzip.compress(buf)
        outfile.write(bufout)
#        struct.pack('<4f4BH2B', x, y, z, cell_size, r, g, b, a, cell_type, 0, 0)
    """
    with open(point_filename,'wb') as outfile:
        total_count=len(coordinates) # coordinates is a list of tuples (x,y,z) 
        buf = struct.pack('<Q',total_count)
        for (x,y,z) in coordinates:
            pt_buf = struct.pack('<3f',x,y,z)
            buf+=pt_buf
        # write the ids at the end of the buffer as increasing integers 
        id_buf = struct.pack('<%sQ' % len(coordinates), *range(len(coordinates)))
        buf+=id_buf
        bufout = gzip.compress(buf)
        outfile.write(bufout)

    info = {}
    spatial = {}
    properties = {}
    spatial["chunk_size"] = (w,h, z_length)
    spatial["grid_shape"] = [1, 1, 1]
    spatial["key"] = "spatial0"
    spatial["limit"] = 10000
    properties["id"] = "joe"
    properties["type"] = "uint16"

    info["@type"] = "neuroglancer_annotations_v1"
    info["annotation_type"] = "POINT"
    info["by_id"] = {"key":"spatial0"}
    info["dimensions"] = {"x":[str(xyresolution*1000),"um"],
                        "y":[str(xyresolution*1000),"um"],
                        "z":[str(int(zresolution)),"um"]}
    info["lower_bound"] = [0,0,0]
    info["upper_bound"] = (w,h,z)
    info["properties"] = [properties]
    info["relationships"] = []
    info["spatial"] = [spatial]    

    with open(info_filename, 'w') as infofile:
        json.dump(info, infofile, indent=2)
        print('Info')
        print(info)
        print(info_filename)


def create_cloud_volume(animal):
    session_id = 8061    
    scaling_factor = 32
    fileLocationManager = FileLocationManager(animal)
    sqlController = SqlController(animal)
    polygons = sqlController.get_annotation_volume(session_id, scaling_factor=1)
    xyresolution = sqlController.scan_run.resolution / scaling_factor
    zresolution = sqlController.scan_run.zresolution

    w = sqlController.scan_run.width//32
    h = sqlController.scan_run.height//32
    path = os.path.join(fileLocationManager.neuroglancer_data, 'predictions0')
    if os.path.exists(path):
        print(f'Removing existing directory {path}')
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    
    shape = (w, h, 440)  # Shape of your volume
    coordinates = []
    for section, points  in polygons.items():
        converted_points = [(p[0]*xyresolution, p[1]*xyresolution) for p in points]
        section = int(round(section / zresolution))
        for point in converted_points:
            x, y = point
            coordinates.append([x, y, section])
    print(f'length of coordinates: {len(coordinates)}')
    points = np.array(coordinates)
    
    volume = np.zeros(shape, dtype=np.uint8)
    print(f'volume shape: {volume.shape}')
    for x, y, z in points:
        volume[int(z), int(y-1), int(x-1)] = 255
        volume[int(z), int(y), int(x-1)] = 255
        volume[int(z), int(y+1), int(x-1)] = 255
        volume[int(z), int(y), int(x)] = 255
        volume[int(z), int(y+1), int(x)] = 255
        volume[int(z), int(y+1), int(x)] = 255
    resolution = (10000, 10000, 20000)  # Microns per voxel
    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type='image',  # or 'segmentation' if you're using labels
        data_type=np.uint8,   # or 'uint32' for segmentation
        encoding='raw',
        resolution=resolution,
        voxel_offset=(0, 0, 0),
        chunk_size=(32,32,32),
        #volume_size=volume.shape[::-1],  # x,y,z
        volume_size=shape,  # x,y,z
    )
    tq = LocalTaskQueue(parallel=1)

    vol = CloudVolume(f'file://{path}', info=info)
    vol.commit_info()
    vol[:,:,:] = volume
    tasks = tc.create_downsampling_tasks(f'file://{path}', mip=0, num_mips=1, compress=True)
    tq.insert(tasks)
    tq.execute()




def create_pointsXXX(animal, layer, session_id, debug=False):

    fileLocationManager = FileLocationManager(animal)
    sqlController = SqlController(animal)
    # Get src points data
    arr = np.array([[1000,2000,300],[1100,2000,300],[1200,2000,300]])
    scan_run = sqlController.scan_run
    sections = sqlController.get_section_count(animal)
    width = scan_run.width
    height = scan_run.height
    xyresolution = scan_run.resolution
    zresolution = scan_run.zresolution
    scales = np.array([xyresolution, xyresolution, zresolution])
    arr = arr / scales

    df = pd.DataFrame(arr, columns = ['x','y','z'])
    df['Layer'] = [layer for l in range(arr.shape[0])]
    df.sort_values(by=['z','x','y'], inplace=True)
    records = df[['x', 'y', 'z']].to_records(index=False)
    coordinates = list(records)
    layer = str(layer).replace(' ','_')
    OUTPUT_DIR = os.path.join(fileLocationManager.neuroglancer_data, layer)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    infofile_path = os.path.join(data_path, 'atlas_data', 'points', 'info')
    outfile = os.path.join(OUTPUT_DIR, 'info')
    with open(infofile_path, 'r+') as rf:
        data = json.load(rf)
        data['upper_bound'] = [width, height, sections]  # <--- add `id` value.
        rf.seek(0)  # <--- should reset file position to the beginning.
    with open(outfile, 'w') as wf:
        json.dump(data, wf, indent=4)

    spatial_dir = os.path.join(OUTPUT_DIR, 'spatial0')
    os.makedirs(spatial_dir)
    total_count = len(coordinates)  # coordinates is a list of tuples (x,y,z)

    filename = os.path.join(spatial_dir, '0_0_0.gz')
    with open(filename, 'wb') as outfile:
        buf = struct.pack('<Q', total_count)
        pt_buf = b''.join(struct.pack('<3f', x, y, z) for (x, y, z) in coordinates)
        buf += pt_buf
        id_buf = struct.pack('<%sQ' % len(coordinates), *range(len(coordinates)))
        buf += id_buf
        bufout = gzip.compress(buf)
        outfile.write(bufout)

    print(f"wrote {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=True)
    parser.add_argument('--session_id', help='Session ID', required=False, default=1)
    parser.add_argument('--scaling_factor', help='scaling factor', required=True, default=32)
    parser.add_argument('--layer', help='layer', required=False, default='test_layer')
    parser.add_argument('--debug', help='print info', required=False, default='true')

    args = parser.parse_args()
    animal = args.animal
    layer = args.layer
    session_id = int(args.session_id)
    scaling_factor = int(args.scaling_factor)
    debug = bool({'true': True, 'false': False}[args.debug.lower()])
    create_points(animal, scaling_factor, debug)
    #create_cloud_volume(animal)

