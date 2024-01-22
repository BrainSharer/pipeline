import argparse
import os, sys
import shutil
import numpy as np
from collections import defaultdict
import cv2
from tqdm import tqdm
from pathlib import Path
PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())


from library.image_manipulation.filelocation_manager import FileLocationManager
from library.controller.sql_controller import SqlController
from library.controller.polygon_sequence_controller import PolygonSequenceController
from library.utilities.utilities_registration import SCALING_FACTOR
from library.utilities.utilities_mask import merge_mask


def create_segmentation(animal, annotator_id, structure_id, debug=False):
    DATA_PATH = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks'
    fileLocationManager = FileLocationManager(animal,)
    sqlController = SqlController(animal)
    # vars
    INPUT = os.path.join(fileLocationManager.prep, 'C1', 'thumbnail_aligned')
    MASK_OUTPUT = os.path.join(DATA_PATH, 'tg', 'thumbnail_masked')
    os.makedirs(MASK_OUTPUT, exist_ok=True)
    NORM_OUTPUT = os.path.join(DATA_PATH, 'tg', 'thumbnail_aligned')
    os.makedirs(NORM_OUTPUT, exist_ok=True)
    MERG_OUTPUT = os.path.join(DATA_PATH, 'tg', 'thumbnail_merged')
    os.makedirs(MERG_OUTPUT, exist_ok=True)
    polygon = PolygonSequenceController(animal=animal)
    
    df = polygon.get_volume(animal, annotator_id, structure_id)
    if debug:
        print(df.head())
        print(df.tail())
    
    scale_xy = sqlController.scan_run.resolution
    z_scale = sqlController.scan_run.zresolution
    scales = np.array([scale_xy, scale_xy, z_scale])
    if debug:
        print('scales', scales, SCALING_FACTOR)
    
    polygons = defaultdict(list)

    
    for _, row in df.iterrows():
        x = row['coordinate'][0]
        y = row['coordinate'][1]
        z = row['coordinate'][2]
        xy = (x/scale_xy/SCALING_FACTOR, y/scale_xy/SCALING_FACTOR)
        section = int(np.round(z/z_scale))
        polygons[section].append(xy)
        
    color = 255 # set it below the threshold set in mask class
    
    if debug:
        for section, points in polygons.items():
            file = str(section).zfill(3) + ".tif"
            inpath = os.path.join(INPUT, file)
            if not os.path.exists(inpath):
                print(f'{inpath} does not exist')
                continue
            img = cv2.imread(inpath, cv2.IMREAD_GRAYSCALE)
            mask = np.zeros((img.shape), dtype=np.uint8)
            points = np.array(points)
            points = points.astype(np.int32)
            cv2.fillPoly(mask, pts = [points], color = color)
            filename = f"{animal}.{annotator_id}.{structure_id}.{file}"
            mask_outpath = os.path.join(MASK_OUTPUT, filename)
            norm_outpath = os.path.join(NORM_OUTPUT, filename)
            merg_outpath = os.path.join(MERG_OUTPUT, filename)
            print(f'section={section} point means={np.mean(points, axis=0)}')
    else:
        for section, points in tqdm(polygons.items()):
            file = str(section).zfill(3) + ".tif"
            inpath = os.path.join(INPUT, file)
            if not os.path.exists(inpath):
                print(f'{inpath} does not exist')
                continue
            img = cv2.imread(inpath, cv2.IMREAD_GRAYSCALE)
            mask = np.zeros((img.shape), dtype=np.uint8)
            points = np.array(points)
            points = points.astype(np.int32)
            cv2.fillPoly(mask, pts = [points], color = color)
            filename = f"{animal}.{annotator_id}.{structure_id}.{file}"
            mask_outpath = os.path.join(MASK_OUTPUT, filename)
            norm_outpath = os.path.join(NORM_OUTPUT, filename)
            merg_outpath = os.path.join(MERG_OUTPUT, filename)
            cv2.imwrite(mask_outpath, mask)
            shutil.copyfile(inpath, norm_outpath)
            merged_img = merge_mask(img, mask)
            cv2.imwrite(merg_outpath, merged_img)

            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=True)
    parser.add_argument('--annotator_id', help='Enter the annotator_id', required=True)
    parser.add_argument('--structure_id', help='Enter the structure_id', required=True)
    parser.add_argument('--debug', help='Enter true or false', required=False, default='false')
    

    args = parser.parse_args()
    animal = args.animal
    annotator_id = int(args.annotator_id)
    structure_id = int(args.structure_id)
    debug = bool({'true': True, 'false': False}[str(args.debug).lower()])
    create_segmentation(animal, annotator_id, structure_id, debug)
