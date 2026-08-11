import argparse
import os
import random
import shutil
import tempfile

import ants
import cv2
import numpy as np
import tifffile
from tqdm import tqdm
from pathlib import Path
import SimpleITK as sitk

class SerialANTsRegistration:

    def __init__(
            self,
            input_dir,
            output_dir,
            transform_dir, mask_dir):

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform_dir = transform_dir
        self.mask_dir = mask_dir

        self.files = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith((".tif", ".tiff"))
        ])

    ######################################################################

    def load_image(self, filename):
        filepath = os.path.join(self.input_dir, filename)
        img = ants.image_read(filepath)
        return img

        img = tifffile.imread(os.path.join(self.input_dir, filename))

        if img.ndim == 3:
            img = img.mean(axis=2)

        img = img.astype(np.float32)

        img -= img.min()

        if img.max() > 0:
            img /= img.max()

        return img

    ######################################################################

    def create_mask(self, image):
        img = (image.numpy() * 255).astype(np.uint8)

        _, mask = cv2.threshold(
            img,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        if n > 1:

            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

            mask = (labels == largest).astype(np.uint8)

        mask[mask > 0] = 255
        #ids, counts = np.unique(mask, return_counts=True)
        #print('ids', ids)
        #print('counts', counts)
        #mask_name = str(random.randint(1000, 9999)) + ".tif"
        #outfile = os.path.join(self.mask_dir, mask_name)
        #tifffile.imwrite(outfile, mask)

        return mask

    ######################################################################

    def centroid(self, mask):

        y, x = np.nonzero(mask)

        if len(x) == 0:
            return np.array(mask.shape[::-1]) / 2

        return np.array([x.mean(), y.mean()])

    ######################################################################

    def largest_section(self):

        areas = []

        for f in tqdm(self.files, desc="Finding largest section"):

            img = self.load_image(f)
            mask = self.create_mask(img)
            areas.append(mask.sum())
        return np.argmax(areas)

    ######################################################################

    def ants_image(self, image):

        return ants.from_numpy(image)

    ######################################################################

    def initial_transform(self, fixed_mask, moving_mask):

        c_fixed = self.centroid(fixed_mask)

        c_moving = self.centroid(moving_mask)

        translation = c_fixed - c_moving

        fd, filename = tempfile.mkstemp(suffix=".mat")
        os.close(fd)

        tx = ants.create_ants_transform(
            transform_type="Euler2DTransform")

        tx.set_parameters((0.0,
                           float(translation[0]),
                           float(translation[1])))

        ants.write_transform(tx, filename)

        return filename

    ######################################################################

    def register_pair(
            self,
            fixed_file,
            moving_file,
            output_prefix):

        fixed = self.load_image(fixed_file)
        moving = self.load_image(moving_file)

        fixed_mask = self.create_mask(fixed)
        moving_mask = self.create_mask(moving)

        #fixed = self.ants_image(fixed)
        #moving = self.ants_image(moving)

        initial = self.initial_transform(
            fixed_mask,
            moving_mask)

        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Rigid",
            initial_transform=initial,
            aff_metric="mattes",
            aff_sampling=64,
            reg_iterations=(1000, 500, 250, 100),
            verbose=True
        )

        forward = reg["fwdtransforms"]

        transform_name = os.path.join(
            self.transform_dir,
            output_prefix + ".mat")

        shutil.copy(forward[0], transform_name)

        warped = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=forward)

        outfile = os.path.join(
            self.output_dir,
            output_prefix + ".tif")

        aligned = warped.numpy().astype(np.uint16)

        tifffile.imwrite(outfile, aligned)

        #os.remove(initial)

        return transform_name

    ######################################################################

    def run(self):

        ref = self.largest_section()

        print("Reference =", self.files[ref])

        for i in tqdm(range(ref - 1, -1, -1), desc="Registering backwards"):
            #print(self.files[i + 1], "<-", self.files[i])

            self.register_pair(
                self.files[i + 1],
                self.files[i],
                os.path.splitext(self.files[i])[0]
            )

        for i in tqdm(range(ref + 1, len(self.files)), desc="Registering forwards"):
            #print(self.files[i - 1], "<-", self.files[i])

            self.register_pair(
                self.files[i - 1],
                self.files[i],
                os.path.splitext(self.files[i])[0]
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Annotation with ID")
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)

    args = parser.parse_args()
    animal = args.animal
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    base_dir = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps'
    input_dir = os.path.join(base_dir, 'C1', 'thumbnail_cleaned')
    mask_dir = os.path.join(base_dir, 'C1', 'masks')
    if not os.path.exists(input_dir):
        raise ValueError(f"Lowres directory does not exist: {input_dir}")
    transform_dir = os.path.join(base_dir, 'transforms')
    output_dir = os.path.join(base_dir, 'C1', 'thumbnail_aligned')
    if os.path.exists(output_dir):
        print(f"Aligned directory {output_dir} already exists, removing.")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(mask_dir):
        print(f"Mask directory {output_dir} already exists, removing.")
        shutil.rmtree(mask_dir)
    os.makedirs(mask_dir, exist_ok=True)

    pipeline = SerialANTsRegistration(input_dir=input_dir, output_dir=output_dir, transform_dir=transform_dir, mask_dir=mask_dir)
    pipeline.run()