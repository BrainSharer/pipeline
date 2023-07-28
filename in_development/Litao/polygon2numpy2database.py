'''
Part one of the two step program to create Neuroglancer data from
hand drawn annotations.
This program will take the polygons (x,y,z) data in the DB, and creates
numpy 3D volume masks. All data in the numpy array is binary: 0,1. 
After creating the numpy volume, it pickles it and puts in the brain_shape
table. It also puts in the numpy shape (dimensions) and x,y,z offsets
in micrometers. There is also an option to create the atlas aligned version.
This calls the umeyama method which creates the rigid transformation of
the data. Each x,y,z coordinate is transformed by the 'brain_to_atlas_transform'
method. Minimums of each of the x,y,z coodinates are stored as the offsets.
Run program from root dir of the project:
    python src/atlas/polygon2numpy2database.py --animal MD594 --transform false
'''
import argparse
import os, sys
import numpy as np
from collections import defaultdict
import cv2
from tqdm import tqdm
from pathlib import Path
PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from pipeline.lib.sqlcontroller import SqlController
from pipeline.lib.file_location import FileLocationManager
from  pipeline.utilities.utilities_process import SCALING_FACTOR
from  pipeline.utilities.utilities_atlas import get_transformation
from registration.algorithm import brain_to_atlas_transform
from model.brain_shape import BrainShape
POLYGON_ID = 54


def create_segmentation(animal, transform=False):
    '''
    This method will fetch all the polygon data from the database
    and creates the numpy arrays
    :param animal: string of the animal name
    :param transform: boolean on whether to create a aligned set of structures
    '''
    fileLocationManager = FileLocationManager(animal)
    # this is the path used to create the snapshot of the midsection image
    # this will be displayed on the admin page in the portal
    OUTPATH = os.path.join(fileLocationManager.thumbnail_web, 'structures')
    os.makedirs(OUTPATH, exist_ok=True)
    sqlController = SqlController(animal)
    # vars
    sections = sqlController.get_sections(animal, 1)
    if len(sections) < 10:
        sections = os.listdir( os.path.join(fileLocationManager.prep, 'CH1/thumbnail_aligned'))
        
    num_sections = len(sections)
    if num_sections < 10:
        print('no sections')
        sys.exit()
    
    # get data from the DB
    width = sqlController.scan_run.width
    height = sqlController.scan_run.height
    scale_xy = sqlController.scan_run.resolution
    z_scale = sqlController.scan_run.zresolution

    width = int(round(width * SCALING_FACTOR))
    height = int(round(height * SCALING_FACTOR))
    
    # get all distinct structures in DB
    abbreviations = sqlController.get_distinct_labels(animal)
    # We loop through every structure from the CSV data in the DB for a brain
    # If we want to transform the data to atlas coordinates, use transform=True
    if transform:
        atlas_centers = sqlController.get_annotation_points_entry('Atlas', FK_input_id=1, person_id=16)
        R, t = get_transformation(animal)
        ## Yoav: Where is the transformation defined by (R,t) used ?
        abbreviations = [a for a in abbreviations if a in atlas_centers.keys()]
        
    print(f'Working with {len(abbreviations)} structures')
    for abbreviation in tqdm(abbreviations):
        rows = sqlController.get_annotations_by_structure(animal, 1, abbreviation, POLYGON_ID)
        polygons = defaultdict(list)
        # xyz in the DB is in micrometers
        for row in rows:
            x, y, z = row.x, row.y, row.z
            if transform:
                x, y, z = brain_to_atlas_transform((x, y, z), R, t)

            xy = (x/scale_xy, y/scale_xy)
            z = int(np.round(z/z_scale))
            polygons[z].append(xy)
        # xyz is now in full resolution Neuroglancer coordinates
        
        # Get the min and max for x,y, and z for each structure
        minx = min([ min([points[0] for points in polygon])  for polygon in polygons.values()])
        miny = min([ min([points[1] for points in polygon])  for polygon in polygons.values()])
        minz = min(polygons.keys())
        
        maxx = max([ max([points[0] for points in polygon])  for polygon in polygons.values()])
        maxy = max([ max([points[1] for points in polygon])  for polygon in polygons.values()])
        maxz = max(polygons.keys())
        
        # put xyz in downsampled Neuroglancer coordinates
        minx = int(round(minx*SCALING_FACTOR))
        miny = int(round(miny*SCALING_FACTOR))
        minz = int(round(minz))
        
        maxx = int(round(maxx*SCALING_FACTOR))
        maxy = int(round(maxy*SCALING_FACTOR))
        maxz = int(round(maxz))
        
        midz = maxz - (maxz - minz)//2

        # create an empty array in which we will stuff the mask with opencv polyfill        
        structure_volume = np.zeros((width, height, num_sections), dtype=np.uint8)
        
        #### loop through all the sections and write to the structure_volume arr
        for section, points in polygons.items():
            template = np.zeros((width, height), dtype=np.uint8)
            points = np.array(points)
            points = np.round(points*SCALING_FACTOR)
            points = points.astype(np.int32)
            cv2.fillPoly(template, pts = [points], color = 1)
            if section == midz:
                img = np.zeros((width, height), dtype=np.uint8)
                cv2.fillPoly(img, pts = [points], color = 255)
                outfile = os.path.join(OUTPATH, f'{abbreviation}.png')
                cv2.imwrite(outfile, img)
        
            structure_volume[:, :, section] += template
            
        
        # xyz offsets are in downsampled Neuroglancer coords
        arr = structure_volume[miny:maxy, minx:maxx, minz:maxz]
        saveme = np.ndarray.dumps(arr) # store as pickle
        brain_region = sqlController.get_structure(abbreviation)
        # we want the xyz coords back in micrometers
        brain_shape = BrainShape(prep_id = animal, FK_structure_id = brain_region.id, 
                                 dimensions=str(arr.shape), 
                                 xoffset = (minx*scale_xy)/SCALING_FACTOR,
                                 yoffset = (miny*scale_xy)/SCALING_FACTOR,
                                 zoffset = minz*z_scale,
                                 transformed=transform,
                                 numpy_data = saveme)
        sqlController.add_row(brain_shape)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=True)
    parser.add_argument('--transform', help='Enter true or false', required=False, default='false')
    

    args = parser.parse_args()
    animal = args.animal
    transform = bool({'true': True, 'false': False}[str(args.transform).lower()])
    create_segmentation(animal, transform)
