"""
William, this is the first program to look at.

This gets the CSV data of the 3 foundation brains' annotations.
These annotations were done by Lauren, Beth, Yuncong and Harvey
(i'm not positive about this)
The annotations are full scale vertices.
MD585 section 161,182,223,231,253 annotations are too far south
MD585 off by 100,60,60,80,60
MD589 section 297 too far north
MD594 all good
"""
import argparse
from collections import defaultdict
import os
import sys
import numpy as np
import pandas as pd
import ast
import json
from tqdm import tqdm
from pathlib import Path
from scipy.interpolate import splprep, splev

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.utilities.utilities_contour import get_contours_from_annotations
from library.controller.sql_controller import SqlController
from library.controller.structure_com_controller import StructureCOMController
from library.image_manipulation.filelocation_manager import FileLocationManager
from foundation_contour_aligner import parse_elastix, create_downsampled_transforms
from settings import data_path as DATA_PATH
from library.utilities.utilities_process import get_image_size



def transform_create_alignment(points, transform):
    a = np.hstack((points, np.ones((points.shape[0], 1))))
    b = transform.T[:, 0:2]
    c = np.matmul(a, b)
    return c


def create_clean_transform(animal):
    sqlController = SqlController(animal)
    fileLocationManager = FileLocationManager(animal)
    aligned_shape = np.array((sqlController.scan_run.width, 
                              sqlController.scan_run.height))
    #downsampled_aligned_shape = np.round(aligned_shape / DOWNSAMPLE_FACTOR).astype(int)
    downsampled_aligned_shape = aligned_shape / DOWNSAMPLE_FACTOR
    print(f'downsampled shape {downsampled_aligned_shape}')
    INPUT = os.path.join(fileLocationManager.prep, 'C1', 'thumbnail')
    files = sorted(os.listdir(INPUT))
    section_offsets = {}
    for file in tqdm(files):
        filepath = os.path.join(INPUT, file)
        width, height = get_image_size(filepath)
        #width = int(width)
        #height = int(height
        downsampled_shape = np.array((width, height))
        section = int(file.split('.')[0])
        section_offsets[section] = (downsampled_aligned_shape - downsampled_shape) / 2
    return section_offsets



def interpolate(points, new_len):
    points = np.array(points)
    pu = points.astype(int)
    indexes = np.unique(pu, axis=0, return_index=True)[1]
    points = np.array([points[index] for index in sorted(indexes)])
    addme = points[0].reshape(1, 2)
    points = np.concatenate((points, addme), axis=0)

    tck, u = splprep(points.T, u=None, s=3, per=1)
    u_new = np.linspace(u.min(), u.max(), new_len)
    x_array, y_array = splev(u_new, tck, der=0)
    arr_2d = np.concatenate([x_array[:, None], y_array[:, None]], axis=1)
    return list(map(tuple, arr_2d))


def create_brain_json(animal, debug):

    section_offsets = create_clean_transform(animal)
    transforms = parse_elastix(animal)
    warp_transforms = create_downsampled_transforms(animal, transforms, downsample=True)
    ordered_downsampled_transforms = sorted(warp_transforms.items())
    section_structure_vertices = defaultdict(dict)
    csvfile = os.path.join(DATA_PATH, 'atlas_data/foundation_brain_annotations',\
        f'{animal}_annotation.csv')
    hand_annotations = pd.read_csv(csvfile)
    hand_annotations['vertices'] = hand_annotations['vertices'] \
        .apply(lambda x: x.replace(' ', ',')) \
        .apply(lambda x: x.replace('\n', ',')) \
        .apply(lambda x: x.replace(',]', ']')) \
        .apply(lambda x: x.replace(',,', ',')) \
        .apply(lambda x: x.replace(',,', ',')) \
        .apply(lambda x: x.replace(',,', ',')).apply(lambda x: x.replace(',,', ','))

    hand_annotations['vertices'] = hand_annotations['vertices'].apply(lambda x: ast.literal_eval(x))
    print(hand_annotations.head())
    controller = StructureCOMController(animal)
    structures = controller.get_structures()
    for structure in structures:
        abbreviation = structure.abbreviation
        contour_annotations, first_sec, last_sec = get_contours_from_annotations(animal, abbreviation, hand_annotations, densify=0)
        for section in contour_annotations:
            section_structure_vertices[section][abbreviation] = contour_annotations[section]

    section_transform = {}
    for section, transform in ordered_downsampled_transforms:
        section_num = int(section.split('.')[0])
        transform = np.linalg.inv(transform)
        section_transform[section_num] = transform

    md585_fixes = {161: 100, 182: 60, 223: 60, 231: 80, 253: 60}
    original_structures = defaultdict(dict)
    unaligned_padded_structures = defaultdict(dict)
    aligned_padded_structures = defaultdict(dict)
    for section in section_structure_vertices:
        section = int(section)
        for structure in section_structure_vertices[section]:

            points = np.array(section_structure_vertices[section][structure]) / DOWNSAMPLE_FACTOR
            points = interpolate(points, max(1500, len(points)))
            original_structures[structure][section] = points
            offset = section_offsets[section]
            if animal == 'MD585' and section in md585_fixes.keys():
                offset = offset - np.array([0, md585_fixes[section]])
            if animal == 'MD589' and section == 297:
                offset = offset + np.array([0, 35])

            points = np.array(points) +  offset
            unaligned_padded_structures[structure][section] = points.tolist()

            points = transform_create_alignment(points, section_transform[section])  # create_alignment transform
            aligned_padded_structures[structure][section] = points.tolist()

    OUTPUT_DIR = os.path.join(DATA_PATH, 'atlas_data', animal)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'Saving data to {OUTPUT_DIR}')
    
    jsonpath1 = os.path.join(OUTPUT_DIR,  'original_structures.json')
    with open(jsonpath1, 'w') as f:
        json.dump(original_structures, f, sort_keys=True)
        
    jsonpath2 = os.path.join(OUTPUT_DIR,  'unaligned_padded_structures.json')
    with open(jsonpath2, 'w') as f:
        json.dump(unaligned_padded_structures, f, sort_keys=True)
        
    jsonpath3 = os.path.join(OUTPUT_DIR,  'aligned_padded_structures.json')
    with open(jsonpath3, 'w') as f:
        json.dump(aligned_padded_structures, f, sort_keys=True)


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
        create_brain_json(animal, debug)
