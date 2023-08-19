"""
This gets the CSV data of the 3 foundation brains' annotations.
These annotations were done by Lauren, Beth, Yuncong and Harvey
(i'm not positive about this)
The annotations are full scale vertices.
MD585 section 161,182,223,231,253 annotations are too far south
MD585 off by 100,60,60,80,60
MD589 section 297 too far north
MD594 all good

The contours were done on unaligned sections
this code does the following:
1. gets the transformation needed to align the contours
2. applies them
3. save the result
"""

from collections import defaultdict
import json
import os
import sys
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
from scipy.interpolate import splprep, splev
from skimage import io
from pathlib import Path
PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.utilities.utilities_process import get_image_size
from library.utilities.utilities_contour import get_contours_from_annotations
from library.controller.sql_controller import SqlController
from library.controller.structure_com_controller import StructureCOMController
from library.image_manipulation.filelocation_manager import data_path as DATA_PATH
from library.image_manipulation.filelocation_manager import FileLocationManager
from library.registration.brain_structure_manager import BrainStructureManager

DOWNSAMPLE_FACTOR = 32


def create_downsampled_transforms(animal, transforms, downsample):
    """
    Changes the dictionary of transforms to the correct resolution
    :param animal: prep_id of animal we are working on
    :param transforms: dictionary of filename:array of transforms
    :param transforms_resol:
    :param downsample; either true for thumbnails, false for full resolution images
    :return: corrected dictionary of filename: array  of transforms
    """

    if downsample:
        transforms_scale_factor = 1
    else:
        transforms_scale_factor = 32

    tf_mat_mult_factor = np.array([[1, 1, transforms_scale_factor], [1, 1, transforms_scale_factor]])

    transforms_to_anchor = {}
    for img_name, tf in transforms.items():
        transforms_to_anchor[img_name] = \
            convert_2d_transform_forms(np.reshape(tf, (3, 3))[:2] * tf_mat_mult_factor) 
    return transforms_to_anchor

def convert_2d_transform_forms(arr):
    return np.vstack([arr, [0, 0, 1]])


def transform_points(points, transform):
    a = np.hstack((points, np.ones((points.shape[0], 1))))
    b = transform.T[:, 0:2]
    c = np.matmul(a, b)
    return c


def create_elastix_transformation(rotation, xshift, yshift, center):
    R = np.array([[np.cos(rotation), -np.sin(rotation)],
                    [np.sin(rotation), np.cos(rotation)]])
    shift = center + (xshift, yshift) - np.dot(R, center)
    T = np.vstack([np.column_stack([R, shift]), [0, 0, 1]])
    return T


def load_elastix_transformation(animal, moving_index):
    controller = SqlController(animal)

    elastixTransformation = controller.get_elastix_row(animal, moving_index)
    #elastixTransformation = session.query(ElastixTransformation).filter(ElastixTransformation.prep_id == animal)\
    #    .filter(ElastixTransformation.section == moving_index).one()

    R = elastixTransformation.rotation
    xshift = elastixTransformation.xshift
    yshift = elastixTransformation.yshift
    return R, xshift, yshift

def parse_elastix(animal):
    """
    After the elastix job is done, this goes into each subdirectory and parses the Transformation.0.txt file
    Args:
        animal: the animal
    Returns: a dictionary of key=filename, value = coordinates
    """
    fileLocationManager = FileLocationManager(animal)
    DIR = fileLocationManager.prep
    INPUT = os.path.join(DIR, 'CH1', 'thumbnail_cleaned')

    files = sorted(os.listdir(INPUT))
    midpoint = len(files) // 2
    transformation_to_previous_sec = {}
    midfilepath = os.path.join(INPUT, files[midpoint])
    midfile = io.imread(midfilepath, img_num=0)
    height = midfile.shape[0]
    width = midfile.shape[1]
    center = np.array([width, height]) / 2
    
    for i in range(1, len(files)):
        moving_index = os.path.splitext(files[i])[0]
        rotation, xshift, yshift = load_elastix_transformation(animal, moving_index)
        T = create_elastix_transformation(rotation, xshift, yshift, center)
        transformation_to_previous_sec[i] = T
    
    
    transformations = {}
    # Converts every transformation
    for moving_index in range(len(files)):
        if moving_index == midpoint:
            transformations[files[moving_index]] = np.eye(3)
        elif moving_index < midpoint:
            T_composed = np.eye(3)
            for i in range(midpoint, moving_index, -1):
                T_composed = np.dot(np.linalg.inv(transformation_to_previous_sec[i]), T_composed)
            transformations[files[moving_index]] = T_composed
        else:
            T_composed = np.eye(3)
            for i in range(midpoint + 1, moving_index + 1):
                T_composed = np.dot(transformation_to_previous_sec[i], T_composed)
            transformations[files[moving_index]] = T_composed

    return transformations



class FoundationContourAligner(BrainStructureManager):
    def __init__(self, animal, *args, **kwrds):
        super().__init__(animal, *args, **kwrds)
        self.data_path = DATA_PATH
        self.file_location_manager = FileLocationManager(animal)
        self.thumbnail_path = self.file_location_manager.get_thumbnail()
        self.contour_path = os.path.join(
            self.data_path, 'atlas_data', 'foundation_brain_annotations', f'{self.animal}_annotation.csv')

    def create_clean_transform(self):
        """creates clean transforms"""

        section_size = np.array((self.sqlController.scan_run.width, self.sqlController.scan_run.height))
        downsampled_section_size = np.round(section_size / DOWNSAMPLE_FACTOR).astype(int)
        INPUT = self.thumbnail_path
        files = sorted(os.listdir(INPUT))
        section_offsets = {}
        for file in tqdm(files):
            filepath = os.path.join(INPUT, file)
            width, height = get_image_size(filepath)
            width,height = int(width),int(height)
            downsampled_shape = np.array((width, height))
            section = int(file.split('.')[0])
            section_offsets[section] = (downsampled_section_size - downsampled_shape) / 2
        self.section_offsets = section_offsets

    def interpolate(self,points, new_len):
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

    def load_transformation_to_align_contours(self):
        """load_transformation_to_align_contours
        """

        transforms = parse_elastix(self.animal)
        downsampled_transforms = create_downsampled_transforms(self.animal, transforms, downsample=True)
        downsampled_transforms = sorted(downsampled_transforms.items())
        self.transform_per_section = {}
        for section, transform in downsampled_transforms:
            section_num = int(section.split('.')[0])
            transform = np.linalg.inv(transform)
            self.transform_per_section[section_num] = transform

    def load_csv_for_foundation_brains(self):
        """load contours for foundation brains
        """
        
        print(f'Loading CSV data from {self.contour_path}')
        hand_annotations = pd.read_csv(self.contour_path)
        hand_annotations['vertices'] = hand_annotations['vertices'] \
            .apply(lambda x: x.replace(' ', ',')) \
            .apply(lambda x: x.replace('\n', ',')) \
            .apply(lambda x: x.replace(',]', ']')) \
            .apply(lambda x: x.replace(',,', ',')) \
            .apply(lambda x: x.replace(',,', ',')) \
            .apply(lambda x: x.replace(',,', ',')).apply(lambda x: x.replace(',,', ','))
        hand_annotations['vertices'] = hand_annotations['vertices'].apply(lambda x: ast.literal_eval(x))
        self.contour_per_structure_per_section = defaultdict(dict)
        controller = StructureCOMController(self.animal)
        structures = controller.get_structures()
        for structure in structures:
            abbreviation = structure.abbreviation
            contour_for_structurei, _, _ = get_contours_from_annotations(self.animal, abbreviation, hand_annotations, densify=0)#7-MAR-2022 MOD (PREV densify=4)
            for section in contour_for_structurei:
                self.contour_per_structure_per_section[abbreviation][section] = contour_for_structurei[section]

    def align_contours(self):
        """align contours
        TODO check section 253, it had two values, 8 and 60
        """
        
        md585_fixes = {161: 100, 182: 60, 223: 60,
                       231: 80, 229: 76, 253: 60}
        self.original_structures = defaultdict(dict)
        self.centered_structures = defaultdict(dict)
        self.aligned_structures = defaultdict(dict)
        for structure in self.contour_per_structure_per_section:
            for section in self.contour_per_structure_per_section[structure]:
                section_str = str(section)
                points = np.array(self.contour_per_structure_per_section[structure][section]) / 32
                points = self.interpolate(points, max(600, len(points)))
                self.original_structures[structure][section_str] = points
                try:
                    offset = self.section_offsets[section]
                except KeyError:
                    offset = 0
                if self.animal == 'MD585' and section in md585_fixes.keys():
                    offset = offset - np.array([0, md585_fixes[section]])
                if self.animal == 'MD589' and section == 297:
                    offset = offset + np.array([0, 35])
                if self.animal == 'MD589' and section == 295:
                    offset = offset + np.array([7, 11])
                points = np.array(points) +  offset
                self.centered_structures[structure][section_str] = points.tolist()
                points = transform_points(points, self.transform_per_section[section]) 
                self.aligned_structures[structure][section_str] = points.tolist()

    def save_json_contours(self):
        original_contour_path = os.path.join(self.animal_directory, 'original_structures.json')
        padded_contour_path = os.path.join(self.animal_directory, 'padded_structures.json')

        assert(hasattr(self, 'original_structures'))
        assert(hasattr(self, 'centered_structures'))
        assert(hasattr(self, 'aligned_structures'))
        with open(original_contour_path, 'w') as f:
            json.dump(self.original_structures, f, sort_keys=True)
        with open(padded_contour_path, 'w') as f:
            json.dump(self.centered_structures, f, sort_keys=True)
        with open(self.aligned_and_padded_contour_path, 'w') as f:
            json.dump(self.aligned_structures, f, sort_keys=True)

    def show_steps(self,structurei='10N_L'):
        self.plot_contours(self.original_structures[structurei])
        self.plot_contours(self.aligned_structures[structurei])

if __name__ == '__main__':
    animals = ['MD585', 'MD589', 'MD594']
    for animal in animals:
        aligner = FoundationContourAligner(animal)
        aligner.load_csv_for_foundation_brains()
        aligner.align_contours()
        aligner.save_json_contours()