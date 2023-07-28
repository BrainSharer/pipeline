"""
This script will take a source brain (where the data comes from) and an image brain 
(the brain whose images you want to display unstriped) and align the data from the point brain
to the image brain. It first aligns the point brain data to the atlas, then that data
to the image brain. It prints out the data by default and also will insert
into the database if given a layer name.
"""

import argparse
import numpy as np
import sys
from pathlib import Path
from registration.algorithm import brain_to_atlas_transform, atlas_to_brain_transform, umeyama

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())


MANUAL = 1
CORRECTED = 2
DETECTED = 3

"""
    The provided r, t is the affine transformation from atlas.pipeline.lib.Brain to atlas such that:
        t_phys = atlas_scale @ t
        atlas_coord_phys = r @ brain_coord_phys + t_phys

    The corresponding reverse transformation is:
        brain_coord_phys = r_inv @ atlas_coord_phys - r_inv @ t_phys
"""


def transform_and_add_dict(animal, person_id, row_dict, r=None, t=None):

    sqlController = SqlController(animal)
    for abbrev, v in row_dict.items():
        x = v[0]
        y = v[1]
        z = v[2]
        if r is not None:
            scan_run = sqlController.scan_run
            brain_coords = np.asarray([x, y, z])
            brain_scale = [scan_run.resolution, scan_run.resolution, scan_run.zresolution]
            transformed = brain_to_atlas_transform(brain_coords, r, t, brain_scale=brain_scale)
            x = transformed[0]
            y = transformed[1]
            z = transformed[2]

        print(animal, abbrev, x, y, z)
        # add_layer(animal, structure, x, y, z, person_id)


def get_common_structure(brains):
    sqlController = SqlController('MD594') # just to declare var
    common_structures = set()
    for brain in brains:
        common_structures = common_structures | set(sqlController.get_annotation_points_entry(brain).keys())
    common_structures = list(sorted(common_structures))
    return common_structures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--pointbrain', help='Enter point animal', required=True)
    parser.add_argument('--imagebrain', help='Enter image animal', required=True)
    parser.add_argument('--inputlayer', help='Enter input layer name', required=False, default='COM')
    parser.add_argument('--FK_input_id', help='Enter input type id', required=False, default=1)
    parser.add_argument('--outputlayer', help='Enter output layer name', required=False)
    args = parser.parse_args()
    pointbrain = args.pointbrain
    imagebrain = args.imagebrain
    inputlayer = args.inputlayer
    outputlayer = args.outputlayer
    FK_input_id = int(args.FK_input_id)
    sqlController = SqlController('MD594') # just to declare var
    pointdata = sqlController.get_annotation_points_entry(pointbrain, FK_input_id, label=inputlayer)
    atlas_centers = sqlController.get_annotation_points_entry('atlas', FK_input_id=1, person_id=16)
    common_structures = get_common_structure(['atlas', pointbrain, imagebrain])
    point_structures = sorted(pointdata.keys())
    dst_point_set = np.array([atlas_centers[s] for s in point_structures if s in common_structures and s in atlas_centers]).T
    point_set = np.array([pointdata[s] for s in point_structures if s in common_structures and s in atlas_centers]).T
    if len(common_structures) < 3 or dst_point_set.size == 0 or point_set.size == 0:
        print(f'len common structures {len(common_structures)}')
        print(f'Size of dst_point_set {dst_point_set.size} point_set {point_set.size}')
        print('No point data to work with.')
        sys.exit()
    r0, t0 = umeyama(point_set, dst_point_set)
    imagedata = sqlController.get_annotation_points_entry(imagebrain)
    image_structures = sorted(imagedata.keys())
    image_set = np.array([imagedata[s] for s in image_structures if s in common_structures and s in atlas_centers]).T
    dst_point_set = np.array([atlas_centers[s] for s in image_structures if s in common_structures and s in atlas_centers]).T
    if dst_point_set.size == 0 or image_set.size == 0:
        print(f'No image data to work with. size of dst,image {dst_point_set.size} {image_set.size}')
        sys.exit()
    r1, t1 = umeyama(image_set, dst_point_set)
    for abbrev, coord in pointdata.items():
        x0, y0 , z0 = coord
        x1, y1, z1 = brain_to_atlas_transform(coord, r0, t0)
        x2, y2, z2 = atlas_to_brain_transform((x1, y1, z1), r1, t1)
        structure = sqlController.get_structure(abbrev)
        print(structure.abbreviation, end="\t")
        print(round(x0), round(y0), round(z0), end="\t\t")
        print(round(x1), round(y1), round(z1), end="\t\t")
        print(round(x2), round(y2), round(z2))
