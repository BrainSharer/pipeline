import argparse
from pathlib import Path
import os
import sys
import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.image_manipulation.filelocation_manager import FileLocationManager
from library.utilities.utilities_process import read_image, write_image
from library.utilities.utilities_mask import merge_mask

def create_mask(image):
    lower = [10, 15, 0]
    upper = [255, 100, 10]

    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    mask = (binary_fill_holes(mask)).astype(np.uint8)
    output = cv2.bitwise_and(image, image, mask=mask)
    ret,thresh = cv2.threshold(mask, 0, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areaArray = []
    for contour in contours:
        area = cv2.contourArea(contour)
        areaArray.append(area)
    # first sort the array by area
    if len(contours) > 0:
        sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
        contour1 = sorteddata[0][1]
        contour2 = sorteddata[1][1]
        contour3 = sorteddata[2][1]
        output = cv2.fillPoly(mask, pts =[contour1, contour2, contour3], color=255)
    return output



def redo_masks(animal):

    fileLocationManager = FileLocationManager(animal)
    input_path = fileLocationManager.get_thumbnail_colored(channel=1)
    output_path = os.path.join(fileLocationManager.masks, 'C1', 'thumbnail_masks_redone')
    os.makedirs(output_path, exist_ok=True)
    files = sorted(os.listdir(input_path))
    for file in tqdm(files):
        inpath = os.path.join(input_path, file)
        outpath = os.path.join(output_path, file)
        if not os.path.exists(outpath):
            img = Image.open(inpath)
            raw_img = np.array(img)

            mask = create_mask(raw_img)
            merged = merge_mask(raw_img, mask)
            cv2.imwrite(outpath, merged.astype(np.uint8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)
    args = parser.parse_args()
    animal = args.animal
    redo_masks(animal)

