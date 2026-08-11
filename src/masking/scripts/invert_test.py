import argparse
import os
import shutil
from tqdm import tqdm

from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import tifffile

import SimpleITK as sitk

from scipy import ndimage

from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    disk
)
from skimage.morphology import closing, opening
from skimage.measure import label, regionprops

@dataclass
class MaskConfig:
    clahe=True
    clahe_clip=2.0
    clahe_grid=(8,8)
    gaussian_sigma=3
    minimum_object_size=5000
    hole_size=5000
    closing_radius=7
    opening_radius=3
    invert="auto"
    fill_holes=True

class MaskGenerator:

    def __init__(self, config: MaskConfig):

        self.config = config

    def read_image(self, filename):

        img = tifffile.imread(filename)

        if img.ndim == 3:

            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if img.dtype == np.uint16:

            img = (img / 256).astype(np.uint8)

        return img

    def enhance(self, image):
        if not self.config.clahe:

            return image

        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip,
            tileGridSize=self.config.clahe_grid
        )

        return clahe.apply(image)
    
    def smooth(self, image):
        sigma = self.config.gaussian_sigma

        return cv2.GaussianBlur(image,(0,0),sigma)
    
    def threshold(self,image):
        t = threshold_otsu(image)

        mask = image > t

        return mask
    
    def auto_invert(self, mask):

        if self.config.invert is False:

            return mask

        if self.config.invert is True:

            return np.logical_not(mask)

        #
        # automatic
        #

        if np.mean(mask) > 0.5:

            return np.logical_not(mask)

        return mask
    
    def fill(self,mask):

        if not self.config.fill_holes:

            return mask

        return ndimage.binary_fill_holes(mask)
    
    def clean(self,mask):

        mask = remove_small_objects(
            mask,
            self.config.minimum_object_size
        )

        mask = remove_small_holes(
            mask,
            self.config.hole_size
        )

        return mask    

    def morphology(self,mask):

        mask = closing(mask, disk(self.config.closing_radius))

        mask = opening(mask, disk(self.config.opening_radius)
        )

        return mask
    
    def largest_component(self,mask):

        lbl = label(mask)

        props = regionprops(lbl)

        if len(props)==0:

            return np.zeros_like(mask)

        largest = max(props,key=lambda x:x.area)

        return lbl==largest.label
    
    def centroid(self,mask):

        lbl = label(mask)

        props = regionprops(lbl)

        if len(props)==0:

            return None

        return props[0].centroid

    def bounding_box(self,mask):

        lbl = label(mask)

        props = regionprops(lbl)

        if len(props)==0:

            return None

        return props[0].bbox
    
    def sitk_mask(self,mask):

        img = sitk.GetImageFromArray(mask.astype(np.uint8))

        return img
    
    def generate(self, filename):
        img = self.read_image(filename)
        img = self.enhance(img)
        img = self.smooth(img)
        mask = self.threshold(img)
        mask = self.auto_invert(mask)
        mask = self.fill(mask)
        mask = self.clean(mask)
        mask = self.largest_component(mask)
        mask = self.morphology(mask)

        return mask.astype(np.uint8)


    def save(self,mask,filename):

        tifffile.imwrite(
            filename,
            mask.astype(np.uint8)*255
        )

    def generate_directory(self,input_dir,output_dir):

        input_dir = Path(input_dir)

        output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        for file in sorted(input_dir.glob("*.tif")):

            mask = self.generate(file)

            outfile = output_dir / file.name

            self.save(mask,outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on masking")
    parser.add_argument('--animal', help='animal', required=True, type=str)
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)

    args = parser.parse_args()
    debug = bool({'true': True, 'false': False}[args.debug.lower()])
    animal = args.animal
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])

    base_path = f"/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps/C1"
    inpath = os.path.join(base_path, "thumbnail")
    if not os.path.exists(inpath):
        print(f'Input does not exist: {inpath}')
        exit(0)
    outpath = os.path.join(base_path, 'inverted')
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
        print(f'Removing {outpath}')
    os.makedirs(outpath, exist_ok=True)
    files = sorted(os.listdir(inpath))
    if len(files) == 0:
        print(f'No files in: {inpath}')
        exit(0)

    mask_config = MaskConfig()
    mask_generator = MaskGenerator(mask_config)
    mask_generator.generate_directory(input_dir=inpath, output_dir=outpath)



    print("Finished")