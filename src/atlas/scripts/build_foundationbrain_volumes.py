"""
William this is the 2nd script to look at

This gets the CSV data of the 3 foundation brains' annotations.
These annotations were done by Lauren, Beth, Yuncong and Harvey
(i'm not positive about this)
The annotations are full scale vertices.
"""
import argparse
import json
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import center_of_mass
from pathlib import Path

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.controller.sql_controller import SqlController
from settings import data_path as DATA_PATH, atlas as ATLAS


def save_volume_origin(animal, structure, volume, xyz_offsets):
    x, y, z = xyz_offsets

    volume = np.swapaxes(volume, 0, 2)
    volume = np.rot90(volume, axes=(0,1))
    volume = np.flip(volume, axis=0)

    OUTPUT_DIR = os.path.join(DATA_PATH, 'atlas_data', animal)
    volume_filepath = os.path.join(OUTPUT_DIR, 'structure', f'{structure}.npy')
    os.makedirs(os.path.join(OUTPUT_DIR, 'structure'), exist_ok=True)
    np.save(volume_filepath, volume)
    origin_filepath = os.path.join(OUTPUT_DIR, 'origin', f'{structure}.txt')
    os.makedirs(os.path.join(OUTPUT_DIR, 'origin'), exist_ok=True)
    np.savetxt(origin_filepath, (x,y,z))


def create_volumes_and_origins(animal, debug):
    #sqlController = SqlController(animal)
    jsonpath = os.path.join(DATA_PATH, 'atlas_data', animal,  'aligned_padded_structures.json')
    if not os.path.exists(jsonpath):
        print(f'{jsonpath} does not exist')
        sys.exit()
    with open(jsonpath) as f:
        aligned_dict = json.load(f)
    structures = list(aligned_dict.keys())
    for structure in structures:
        onestructure = aligned_dict[structure]
        mins = []
        maxs = []

        for section_num, points in onestructure.items():
            arr_tmp = np.array(points)
            min_tmp = np.min(arr_tmp, axis=0)
            max_tmp = np.max(arr_tmp, axis=0)
            mins.append(min_tmp)
            maxs.append(max_tmp)

        min_xy = np.min(mins, axis=0)
        max_xy = np.max(maxs, axis=0)
        max_x = max_xy[0]
        max_y = max_xy[1]
        sections = [int(i) for i in onestructure.keys()]
        min_x = min_xy[0]
        min_y = min_xy[1]
        min_z = min(sections)
        xlength = max_x - min_x
        ylength = max_y - min_y
        PADDED_SIZE = (int(ylength), int(xlength))
        volume = []
        for section, points in sorted(onestructure.items()):
            vertices = np.array(points) - np.array((min_x, min_y))
            volume_slice = np.zeros(PADDED_SIZE, dtype=np.uint8)
            points = (vertices).astype(np.int32)
            #color = sqlController.get_structure_color_rgb(structure)
            volume_slice = cv2.polylines(volume_slice, [points], isClosed=True, 
                                         color=1, thickness=1)
            volume_slice = cv2.fillPoly(volume_slice, pts=[points], color=1)

            volume.append(volume_slice)
        
        # should this be a boolean?
        volume = np.array(volume).astype(np.bool_)
        if 'SC' in structure:
            print(f'{animal=} {structure=} {min_x=} {min_y=} {min_z=}')
        if not debug:
            # we want the real center of mass in the DB
            #sqlController.add_layer_data(abbreviation=structure, animal=animal, 
            #                         layer='COM', x=comx, y=comy, section=comz, 
            #                         person_id=222222, input_type_id=1)
            save_volume_origin(animal, structure, volume, (min_x, min_y, min_z))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=False)
    parser.add_argument('--debug', help='Enter debug True|False', required=False, default='true')
    args = parser.parse_args()
    animal = args.animal
    debug = bool({'true': True, 'false': False}[str(args.debug).lower()])
    if animal is None:
        animals = ['MD585', 'MD589', 'MD594']
    else:
        animals = [animal]

    for animal in animals:
        create_volumes(animal, debug)
