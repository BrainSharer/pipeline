import sys
import os
import json
import argparse

import numpy as np
import cv2
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), '../'))
from utilities.SqlController import SqlController
from .utils import get_structure_number, get_segment_properties, NumpyToNeuroglancer


def read_from_cvat(cvat_data_fp):
    '''
    Read labels and vertices of each polygon in every section from cvat structure data.
    :param cvat_data_fp: file path to cvat exported data, a json file
    :return: a dictionary containing vertex coordinates organized by section numbers and structure names
    '''
    cvat_json = json.load(open(cvat_data_fp,'r'))

    # Read annotation contours coordinates
    category_dict = {category['id']: category['name'] for category in cvat_json['categories']}
    contours = {}
    for annotation in cvat_json['annotations']:
        section = annotation['image_id']
        if not section in contours.keys():
            contours[section] = {}

        landmarks = contours[section]
        structure = category_dict[annotation['category_id']]
        if not structure in landmarks.keys():
            landmarks[structure] = []

        polygon = np.array(annotation['segmentation'])
        polygon = np.c_[polygon[0,::2], polygon[0,1::2]]
        landmarks[structure].append(polygon)

    return contours

def draw_numpy(section_structure_polygons, size, section_start, section_end, draw_lines=True):
    volume = np.zeros((size[1], size[0], section_end - section_start), dtype=np.uint8)
    for section in tqdm(range(section_start, section_end)):
        if section in section_structure_polygons:
            template = np.zeros((size[1], size[0]), dtype=np.uint8)
            for structure in section_structure_polygons[section]:
                polygons = section_structure_polygons[section][structure]
                for polygon in polygons:
                    color = get_structure_number(structure)

                    if draw_lines:
                        cv2.polylines(template, [polygon.astype(np.int32)], True, color, 1)
                    else:
                        for point in polygon:
                            cv2.circle(template, tuple(point.astype(np.int32)), 0, color, -1)

            volume[:, :, section - section_start - 1] = template

    volume = np.swapaxes(volume, 0, 1)
    return volume

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Convert CVAT coco annotation file to Neuroglancer precomputed format')
    parser.add_argument("animal", type=str)
    parser.add_argument("downsample_factor", type=int, help="The downsampled factor of the brain images")
    parser.add_argument("cvat_file", type=str, help="Path to cvat exported file")
    parser.add_argument("precomputed_path", type=str, help="Path to Neuroglancer Precomputed dir")
    parser.add_argument("draw_lines", type=bool, help="Whether to draw lines or dots for Neuroglancer", default=True)

    args = parser.parse_args()

    sqlController = SqlController(args.animal)
    resolution = sqlController.scan_run.resolution
    aligned_shape = np.array((sqlController.scan_run.width, sqlController.scan_run.height))
#     num_section = len(os.listdir(IMAGE_DIR_PATH))
    num_section = 452
    downsampled_aligned_shape = np.round(aligned_shape / args.downsample_factor).astype(int)
    scales = np.array([resolution * args.downsample_factor, resolution * args.downsample_factor, 20]) * 1000

    contours = read_from_cvat(args.cvat_file)
    volume = draw_numpy(contours, downsampled_aligned_shape, 0, num_section, draw_lines=args.draw_lines)

    ng = NumpyToNeuroglancer(volume, scales)
    ng.init_precomputed(args.precompute_path)
    ng.add_segment_properties(get_segment_properties())
    ng.add_downsampled_volumes()
    ng.add_segmentation_mesh()
