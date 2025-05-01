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


PIPELINE_ROOT = Path('./').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())
print(PIPELINE_ROOT)
sys.exit()

from Controllers.MarkedCellController import MarkedCellController
from Controllers.SqlController import SqlController
from lib.FileLocationManager import FileLocationManager, DATA_PATH


def create_points(animal, layer, session_id, debug=False):
    fileLocationManager = FileLocationManager(animal)
    sqlController = SqlController(animal)
    # Get src points data
    cellController = MarkedCellController()
    arr = cellController.get_cells_from_sessioni(session_id)
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
    if debug:
        print(arr.dtype, arr.shape)
        print(width, height, sections, xyresolution, zresolution)
        print(df.head())
        print(df.info())
    else:
        records = df[['x', 'y', 'z']].to_records(index=False)
        coordinates = list(records)
        layer = str(layer).replace(' ','_')
        OUTPUT_DIR = os.path.join(fileLocationManager.neuroglancer_data, layer)
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        info = os.path.join(DATA_PATH, 'atlas_data', 'points', 'info')
        outfile = os.path.join(OUTPUT_DIR, 'info')
        with open(info, 'r+') as rf:
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

        """
        with open(filename,'wb') as outfile:
            buf = struct.pack('<Q',total_count)
            for (x,y,z) in coordinates:
                pt_buf = struct.pack('<3f',x,y,z)
                buf+=pt_buf
            id_buf = struct.pack('<%sQ' % len(coordinates), *range(len(coordinates)))
            buf+=id_buf
            outfile.write(buf)
        """
        print(f"wrote {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=True)
    parser.add_argument('--session_id', help='Session ID', required=True)
    parser.add_argument('--layer', help='layer', required=False, default='test_layer')
    parser.add_argument('--debug', help='print info', required=False, default='true')

    args = parser.parse_args()
    animal = args.animal
    layer = args.layer
    session_id = int(args.session_id)
    debug = bool({'true': True, 'false': False}[args.debug.lower()])
    create_points(animal, layer, session_id, debug)

