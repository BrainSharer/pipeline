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


def create_masks(animal, annotator_id, debug=False):
    """First get the existing polygons from well labeled brains.
    Then we create the masks for each section. Each mask will be
    for one brain and will contain many structures. There will also 
    be a corresponding aligned image with the same name as the mask created above.
    """

    TRAINING_PATH = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/structures'
    fileLocationManager = FileLocationManager(animal,)
    sqlController = SqlController(animal)
    # vars
    IMG_INPUT = os.path.join(fileLocationManager.prep, 'C1', 'thumbnail_aligned')
    if not os.path.exists(IMG_INPUT):
        print(f'{IMG_INPUT} does not exist')
        sys.exit()
    MASK_OUTPUT = os.path.join(TRAINING_PATH, 'thumbnail_masked')
    IMG_OUTPUT = os.path.join(TRAINING_PATH, 'thumbnail_aligned')
    os.makedirs(MASK_OUTPUT, exist_ok=True)
    os.makedirs(IMG_OUTPUT, exist_ok=True)

    polygon = PolygonSequenceController(animal=animal)

    structure_ids = [21]

    for structure_id in structure_ids:
    
        df = polygon.get_volume(animal, annotator_id, structure_id)
        scale_xy = sqlController.scan_run.resolution
        z_scale = sqlController.scan_run.zresolution
        polygons = defaultdict(list)
        
        for _, row in df.iterrows():
            x = row['coordinate'][0]
            y = row['coordinate'][1]
            z = row['coordinate'][2]
            xy = (x/scale_xy/SCALING_FACTOR, y/scale_xy/SCALING_FACTOR)
            section = int(np.round(z/z_scale))
            polygons[section].append(xy)
            
        color = 200 + structure_id # set it below the threshold set in mask class
        
        for section, points in tqdm(polygons.items()):
            file = str(section).zfill(3) + ".tif"
            inpath = os.path.join(IMG_INPUT, file)
            filename = f"{animal}.{file}"
            img_outpath = os.path.join(IMG_OUTPUT, filename)
            mask_outpath = os.path.join(MASK_OUTPUT, filename)
            img = cv2.imread(inpath, cv2.IMREAD_GRAYSCALE)
            if os.path.exists(mask_outpath):
                maskfile = cv2.imread(mask_outpath, cv2.IMREAD_GRAYSCALE)
            else:
                maskfile = np.zeros((img.shape), dtype=np.uint8)
            points = np.array(points).astype(np.int32)
            cv2.fillPoly(maskfile, pts = [points], color = color)

            if not os.path.exists(img_outpath):
                shutil.copyfile(inpath, img_outpath) # only needs to be done once
            cv2.imwrite(mask_outpath, maskfile)
 
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=True)
    parser.add_argument('--annotator_id', help='Enter the annotator_id', required=True)
    parser.add_argument('--debug', help='Enter true or false', required=False, default='false')
    

    args = parser.parse_args()
    animal = args.animal
    annotator_id = int(args.annotator_id)
    debug = bool({'true': True, 'false': False}[str(args.debug).lower()])
    create_masks(animal, annotator_id, debug)
