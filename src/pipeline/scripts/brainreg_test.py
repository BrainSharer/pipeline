import argparse
import os
import SimpleITK as sitk
from pytz import reference
from tqdm import tqdm
import shutil
import numpy as np
from pathlib import Path
from math import pi

DOWNSAMPLE=32
PIXEL_SIZE_UM=0.325



################################################################################
# Utility
################################################################################

def command_iteration(method):
    """ Callback invoked when the optimization has an iteration """
    if method.GetOptimizerIteration() == 0:
        print("Scales: ", method.GetOptimizerScales())
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():7.5f} "
        + f": {method.GetOptimizerPosition()}"
    )


def read_images(directory):

    files = sorted(Path(directory).glob("*.tif"))

    images = [sitk.ReadImage(str(f), sitk.sitkFloat32)for f in files]

    return files, images

import cv2
import numpy as np


def create_mask(image, threshold=10):
    mask = sitk.BinaryThreshold(
        image,
        lowerThreshold=threshold,
        upperThreshold=1e9,
        insideValue=255,
        outsideValue=0
    )
    eroder = sitk.GrayscaleErodeImageFilter()
    eroder.SetKernelType(sitk.sitkBall)
    eroder.SetKernelRadius(10)
    eroded_img = eroder.Execute(mask)

    return sitk.Cast(eroded_img, sitk.sitkUInt8)

def largest_contour(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    contour = max(contours, key=cv2.contourArea)
    return contour

def contour_bbox(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, w, h

def crop(image, bbox):

    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

def contour_centroid(contour):
    m = cv2.moments(contour)
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]

    return np.array([cx, cy])

def centroid_translation(fixed_contour, moving_contour):
    c_fixed = contour_centroid(fixed_contour)
    c_moving = contour_centroid(moving_contour)
    return c_fixed - c_moving

def read_image(fname):
    img = sitk.ReadImage(str(fname), sitk.sitkFloat32)
    return img


def binary_mask(image):

    otsu = sitk.OtsuThreshold(image, 0, 1)

    cc = sitk.ConnectedComponent(otsu)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    largest = max(stats.GetLabels(),
                  key=lambda x: stats.GetPhysicalSize(x))

    mask = sitk.BinaryThreshold(cc,
                                lowerThreshold=largest,
                                upperThreshold=largest,
                                insideValue=1,
                                outsideValue=0)

    return mask


def largest_area(image):

    mask = binary_mask(image)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)

    return stats.GetPhysicalSize(1)

def rigid_registration(fixed,
                       moving,
                       initial_offset):

    transform = sitk.Euler2DTransform()
    transform.SetTranslation(tuple(initial_offset))
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(50)
    registration.SetMetricSamplingPercentage(0.20)
    registration.SetMetricSamplingStrategy(
        registration.RANDOM
    )
    registration.SetInterpolator(
        sitk.sitkLinear
    )
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2,
        minStep=0.001,
        numberOfIterations=200,
    )
    registration.SetInitialTransform(
        transform,
        False,
    )
    final = registration.Execute(
        fixed,
        moving,
    )
    return final

def resample(moving,
             fixed,
             transform):

    return sitk.Resample(
        moving,
        fixed,
        transform,
        sitk.sitkLinear,
        0,
        moving.GetPixelID(),
    )

def rigid_register(fixed, moving):

    registration = sitk.ImageRegistrationMethod()
    initial_transform = sitk.Euler2DTransform()
    # METHOD can be GEOMETRY (aligns based on physical centers) or MOMENTS (aligns based on center of mass/intensity)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        initial_transform,
        sitk.CenteredTransformInitializerFilter.MOMENTS
    )

    registration.SetInitialTransform(initial_transform)

    registration.SetMetricAsMattesMutualInformation(50)
    registration.SetMetricSamplingPercentage(1.0)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,minStep=1e-4,numberOfIterations=250,gradientMagnitudeTolerance=1e-8)
    #registration.SetOptimizerAsGradientDescent(learningRate=0.2, numberOfIterations=250, convergenceWindowSize=100)

    """ MASKS
    fixed_mask = create_mask(fixed)
    moving_mask = create_mask(moving)
    fixed_contour = largest_contour(sitk.GetArrayFromImage(fixed_mask))
    moving_contour = largest_contour(sitk.GetArrayFromImage(moving_mask))
    registration.SetMetricFixedMask(fixed_mask)
    registration.SetMetricMovingMask(moving_mask)
    """

    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([6,4,2,1])
    registration.SetSmoothingSigmasPerLevel([4,2,1,0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    #registration.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration))
    try:
        transform = registration.Execute(fixed, moving)
        print(f"Optimizer stop condition: {registration.GetOptimizerStopConditionDescription()}")
        print(f" Iteration: {registration.GetOptimizerIteration()}")
        print(f" Metric value: {registration.GetMetricValue()}")    
    except Exception as e:
        print("Error during registration:", e)
        transform = sitk.Euler2DTransform()

    return transform




class RigidSectionAligner:

    def __init__(self,animal, debug=False):

        base_dir = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps'
        self.lowres_dir = os.path.join(base_dir, 'C1', 'thumbnail_cleaned')
        if not os.path.exists(self.lowres_dir):
            raise ValueError(f"Lowres directory does not exist: {self.lowres_dir}")
        self.transform_dir = os.path.join(base_dir, 'transforms')
        self.output_dir = os.path.join(base_dir, 'C1', 'thumbnail_aligned')

        self.downsample = DOWNSAMPLE
        self.debug = debug

        self.full_spacing = (
            PIXEL_SIZE_UM,
            PIXEL_SIZE_UM,
        )

        self.low_spacing = (PIXEL_SIZE_UM * DOWNSAMPLE,PIXEL_SIZE_UM * DOWNSAMPLE)
        self.low_files = sorted(os.listdir(self.lowres_dir))
        self.transforms = {}

        self.N = len(self.low_files)

        self.center = self.N // 2

    def create_save_transforms(self, images, low_files, transform_dir):
        print("Creating transforms for", len(images), "images, reference index:", self.reference_index)
        self.transforms = [None]*len(images)
        identity = sitk.Euler2DTransform()
        self.transforms[self.reference_index] = identity

        ################################################################################
        # Register forward
        ################################################################################

        for i in tqdm(range(self.reference_index+1, len(images)), desc="Registering forward", disable=self.debug):
            t = rigid_register(images[i-1], images[i])
            composite = sitk.CompositeTransform(2)
            composite.AddTransform(self.transforms[i-1])
            composite.AddTransform(t)
            self.transforms[i] = composite

        ################################################################################
        # Register backward
        ################################################################################

        for i in tqdm(range(self.reference_index-1, -1, -1), desc="Registering backward", disable=self.debug):
            t = rigid_register(images[i+1], images[i])
            composite = sitk.CompositeTransform(2)
            composite.AddTransform(self.transforms[i+1])
            composite.AddTransform(t)
            self.transforms[i] = composite

        ################################################################################
        # Save transforms
        ################################################################################

        for f, t in tqdm(zip(low_files, self.transforms), desc="Saving transforms", disable=self.debug):
            f = f.replace("tif", "tfm")
            transform_file = os.path.join(transform_dir, f)

            try:
                sitk.WriteTransform(t, transform_file)
            except Exception as e:
                t.FlattenTransform()
                try:
                    sitk.WriteTransform(t, transform_file)
                except Exception as e:
                    print(f"Error saving flattened transform {transform_file}")


    ###############################################################

    def read_low(self, filename):

        file_path = os.path.join(self.lowres_dir, filename)
        img = sitk.ReadImage(file_path, sitk.sitkFloat32)
        img.SetSpacing(self.low_spacing)

        return img


    ###############################################################

    def apply_downsampled_transforms(self):
        reference_file = self.low_files[self.reference_index]
        reference = self.read_low(reference_file)
        for file in tqdm(self.low_files, desc="Applying transforms to downsampled images", disable=self.debug):

            moving = self.read_low(file)

            transform_file = os.path.join(self.transform_dir, str(file).replace(".tif", ".tfm"))
            transform = sitk.ReadTransform(transform_file)


            if isinstance(transform, sitk.CompositeTransform):

                scaled = sitk.CompositeTransform(2)

                for n in range(transform.GetNumberOfTransforms()):
                    tr = transform.GetNthTransform(n)
                    scaled.AddTransform(tr)
            else:
                print(f"Transform for {file=} is not a CompositeTransform")
                scaled = transform
                #scaled = scale_transform(transform, DOWNSAMPLE)


            aligned = sitk.Resample(
                moving,
                reference,
                scaled,
                sitk.sitkLinear,
                0,
                moving.GetPixelID(),
            )
            aligned_16 = sitk.Cast(aligned, sitk.sitkUInt16)
            normalized = sitk.RescaleIntensity(aligned_16, 0, 45000)
            sitk.WriteImage(normalized, os.path.join(self.output_dir, file))

    ###############################################################
    def runXXX(self):
        print(f"Running RigidSectionAligner for {self.N} sections, center={self.center}")
        if os.path.exists(self.output_dir):
            print(f"Aligned directory {self.output_dir} already exists, removing.")
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        areas = []
        images = read_images(self.lowres_dir)[1]

        for image in images:

            mask = create_mask(image)

            areas.append(mask.sum())

        self.reference_index = np.argmax(areas)
        order = []

        for i in range(self.reference_index-1, -1, -1):
            order.append((i+1, i))

        for i in range(self.reference_index+1, len(images)):
            order.append((i-1, i))

        registered = list(images)

        for fixed_index, moving_index in order:

            fixed = registered[fixed_index]
            moving = registered[moving_index]

            fixed_mask = create_mask(fixed)
            moving_mask = create_mask(moving)


            fixed_contour = largest_contour(fixed_mask)
            moving_contour = largest_contour(moving_mask)
            print(f"Fixed contour: {fixed_contour.shape}, moving contour: {moving_contour.shape}")

            translation = centroid_translation(
                fixed_contour,
                moving_contour,
            )
            print(f"Registering {self.low_files[moving_index]} to {self.low_files[fixed_index]}, translation: {translation}")
            transform = rigid_registration(
                sitk.GetImageFromArray(fixed_mask.astype(np.float32)),
                sitk.GetImageFromArray(moving_mask.astype(np.float32)),
                translation,
            )

            registered[moving_index] = resample(
                moving,
                fixed,
                transform,
            )
            aligned = registered[moving_index]

            aligned_16 = sitk.Cast(aligned, sitk.sitkUInt16)
            normalized = sitk.RescaleIntensity(aligned_16, 0, 45000)
            outpath = os.path.join(self.output_dir, self.low_files[moving_index])
            sitk.WriteImage(normalized, outpath)



    ###############################################################
    def run(self):
        print(f"Running RigidSectionAligner for {self.N} sections, center={self.center}")
        if os.path.exists(self.output_dir):
            print(f"Aligned directory {self.output_dir} already exists, removing.")
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)


        low_files = sorted(os.listdir(self.lowres_dir))
        images = [read_image(os.path.join(self.lowres_dir, f)) for f in low_files]
        areas = [largest_area(img) for img in images]
        self.reference_index = np.argmax(areas)
        print("Reference image:", self.reference_index)

        self.transforms = [None]*len(images)
        # test if the transforms already exist
        if os.path.exists(self.transform_dir) and len(os.listdir(self.transform_dir)) == len(images):
            for i in range(len(images)):
                transform_file = os.path.join(self.transform_dir, str(i).zfill(3) + ".tfm")
                if os.path.exists(transform_file):
                    self.transforms[i] = sitk.ReadTransform(transform_file)
        else:
            self.create_save_transforms(images, low_files, self.transform_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Annotation with ID")
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)

    args = parser.parse_args()
    animal = args.animal
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])

    aligner = RigidSectionAligner(animal=animal, debug=debug)
    aligner.run()
    aligner.apply_downsampled_transforms()
    #aligner.apply_full_resolution()
    