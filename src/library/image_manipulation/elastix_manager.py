"""This module takes care of the section to section alignment. It imports
libraries that contain the code from the elastix command line tools:
https://elastix.lumc.nl/
The libraries are contained within the SimpleITK-SimpleElastix library
"""

import math
import os
import numpy as np
from collections import OrderedDict
from PIL import Image
from timeit import default_timer as timer
Image.MAX_IMAGE_PIXELS = None
import SimpleITK as sitk
from scipy.ndimage import affine_transform
from tqdm import tqdm
#GPU alt TESTING

#import cupy as cp

from library.image_manipulation.filelocation_manager import FileLocationManager
from library.image_manipulation.file_logger import FileLogger
from library.utilities.utilities_process import SCALING_FACTOR, read_image, test_dir, write_image
from library.utilities.utilities_mask import equalized, normalize_image
from library.utilities.utilities_registration import (
    align_image_to_affine,
    create_downsampled_transforms,
    create_rigid_parameters,
    create_scaled_transform,
    parameters_to_rigid_transform,
    tif_to_png,
)
from library.image_manipulation.image_manager import ImageManager


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
# from torch.optim.optimizer import Optimizer, required


class ElastixManager():
    """Class for generating, storing and applying transformations within 
    stack alignment [with the Elastix package]
    All methods relate to aligning images in stack
    """

    def __init__(self):
        self.pixelType = sitk.sitkFloat32
        self.registration_output = os.path.join(self.fileLocationManager.prep, 'registration')
        self.input, self.output = self.fileLocationManager.get_alignment_directories(channel=self.channel, resolution='thumbnail')
        self.maskpath = self.fileLocationManager.get_thumbnail_masked(channel=1) # usually channel=1, except for step 6
        

    def create_within_stack_transformations(self):
        """Calculate and store the rigid transformation using elastix.  
        The transformations are calculated from the next image to the previous
        This is done in a simple loop with no workers. Usually takes
        up to an hour to run for a stack. It only needs to be run once for
        each brain. 
        """
        if self.debug:
            print("DEBUG: START ElastixManager::create_within_stack_transformations")

        files, nfiles = test_dir(self.animal, self.input, self.section_count, True, same_size=True)
        
        self.fileLogger.logevent(f"Input FOLDER (COUNT): {self.input} ({nfiles=})")
        
        for i in range(1, nfiles):
            fixed_index = os.path.splitext(files[i - 1])[0]
            moving_index = os.path.splitext(files[i])[0]
            if not self.sqlController.check_elastix_row(self.animal, moving_index):
                #rotation, xshift, yshift, metric = self.align_elastix(fixed_index, moving_index, use_points=False) #DEPRECATED IN SEP-2024; REMOVE IN 2025 IF NO LONGER NEEDED
                rotation, xshift, yshift, metric = self.align_images_elastix_GPU(fixed_index, moving_index) #pyCUDA elastix GPU apt (WORKING + FASTER AS OF 20-SEP-2024)
                self.sqlController.add_elastix_row(self.animal, moving_index, rotation, xshift, yshift, metric)

    def update_within_stack_transformations(self):
        """Takes the existing transformations and aligned images and improves the transformations
        Elastix needs the points to be in files so we need to write the data from the DB to the filesystem.
        Each section will have it's own point file. We usually create 3 points per section, so there
        will be a point file containing 3 ponts for each section.
        """
        os.makedirs(self.registration_output, exist_ok=True)
        fiducials = self.sqlController.get_fiducials(self.animal)
        nchanges = len(fiducials)
        if nchanges == 0:
            print('No fiducial points were found and so no changes have been made.')
            return nchanges

        for section, points in fiducials.items():
            section = str(int(section)).zfill(3)
            point_file = os.path.join(self.registration_output, f'{section}_points.txt')
            with open(point_file, 'w') as f:
                f.write(f'{len(points)}\n')
                for point in points:
                    x = point[0]
                    y = point[1]
                    f.write(f'{x} {y}')
                    f.write('\n')

        files = sorted(os.listdir(self.input))
        nfiles = len(files)
        print(f'Making {nchanges} changes from {nfiles} images from {os.path.basename(os.path.normpath(self.input))}')
        aligned_sum = 0
        realigned_sum = 0
        for i in range(1, nfiles):
            fixed_index = os.path.splitext(files[i - 1])[0]
            moving_index = os.path.splitext(files[i])[0]
            rotation, xshift, yshift, metric = self.align_elastix(fixed_index, moving_index, use_points=True)
            self.sqlController.check_elastix_row(self.animal, moving_index)
            transformation = self.sqlController.get_elastix_row(self.animal, moving_index)

            if rotation != 0 and xshift != 0 and yshift != 0:
                aligned_sum += abs(transformation.rotation) + abs(transformation.xshift) + abs(transformation.yshift)
                realigned_sum += abs(rotation) + abs(xshift) + abs(yshift)
                print(f'\tUpdating {moving_index} with rotation={rotation}, xshift={xshift}, yshift={yshift}, metric={metric}')
                updates = dict(rotation=rotation, xshift=xshift, yshift=yshift, metric=metric)
                self.sqlController.update_elastix_row(self.animal, moving_index, updates)

        sum_changes = abs(aligned_sum - realigned_sum)
        if math.isclose(sum_changes, 0, abs_tol=0.01):
            print('Changes have already been made to the alignment, so there is no need to rerurn the alignment and neuroglancer tasks.')
            nchanges = 0
        return nchanges

    def align_elastix(self, fixed_index, moving_index, use_points=False):
        """
        DEPRECATED IN SEP-2024; REMOVE IN 2025 IF NO LONGER USED
        Aligns two images using the Elastix registration algorithm.

        Args:
            fixed_index (int): The index of the fixed image.
            moving_index (int): The index of the moving image.
            use_points (bool, optional): Whether to use corresponding points for registration. Defaults to False.

        Returns:
            tuple: A tuple containing the rotation angle (R), translation in the x-axis (x), translation in the y-axis (y),
            and the registration metric.

        Raises:
            AssertionError: If the number of fixed points does not match the number of moving points.

        """
        if use_points:
            fixed_point_file = os.path.join(self.registration_output, f'{fixed_index}_points.txt')
            moving_point_file = os.path.join(self.registration_output, f'{moving_index}_points.txt')
            if os.path.exists(fixed_point_file) and os.path.exists(moving_point_file):
                print(f'Found fixed point file: {os.path.basename(os.path.normpath(fixed_point_file))}', end=" ")
                print(f'and moving point file: {os.path.basename(os.path.normpath(moving_point_file))}')
                with open(fixed_point_file, 'r') as fp:
                    fixed_count = len(fp.readlines())
                with open(moving_point_file, 'r') as fp:
                    moving_count = len(fp.readlines())
                assert fixed_count == moving_count, \
                        f'Error, the number of fixed points in {fixed_point_file} do not match {moving_point_file}'
            else:
                return 0, 0, 0, 0

        elastixImageFilter = sitk.ElastixImageFilter()
        fixed_file = os.path.join(self.input, f"{fixed_index}.tif")
        fixed = sitk.ReadImage(fixed_file, self.pixelType)

        moving_file = os.path.join(self.input, f"{moving_index}.tif")
        moving = sitk.ReadImage(moving_file, self.pixelType)
        elastixImageFilter.SetFixedImage(fixed)
        elastixImageFilter.SetMovingImage(moving)

        rigid_params = create_rigid_parameters(elastixImageFilter, debug=self.debug)
        elastixImageFilter.SetParameterMap(rigid_params)

        if use_points:
            elastixImageFilter.SetParameter("Registration", ["MultiMetricMultiResolutionRegistration"])
            elastixImageFilter.SetParameter("Metric",  ["AdvancedMattesMutualInformation", "CorrespondingPointsEuclideanDistanceMetric"])
            elastixImageFilter.SetParameter("Metric0Weight", ["0.25"]) # the weight of 1st metric for each resolution
            elastixImageFilter.SetParameter("Metric1Weight",  ["0.75"]) # the weight of 2nd metric
            elastixImageFilter.SetFixedPointSetFileName(fixed_point_file)
            elastixImageFilter.SetMovingPointSetFileName(moving_point_file)

        elastixImageFilter.SetLogToFile(True)
        logpath =  os.path.join(self.registration_output, 'iteration_logs')
        os.makedirs(logpath, exist_ok=True)
        elastixImageFilter.SetOutputDirectory(logpath)        

        elastixImageFilter.LogToConsoleOff()
        if self.debug:
            elastixImageFilter.PrintParameterMap()
        elastixImageFilter.Execute()

        R, x, y = elastixImageFilter.GetTransformParameterMap()[0]["TransformParameters"]
        metric = self.get_metric(logpath)

        return float(R), float(x), float(y), float(metric)

    def align_images_elastix_GPU(self, fixed_index: str, moving_index: str, use_points: bool = False) -> tuple[float, float, float, float]:
        """
        Aligns two images using the Elastix registration algorithm with GPU acceleration.
        expected to replace 'align_elastix' (TESTING)

        Args:
            fixed_index (str due to filename extraction): The index of the fixed image.
            moving_index (str due to filename extraction): The index of the moving image.
            use_points (bool, optional): Whether to use corresponding points for registration. Defaults to False.

        Returns:
            tuple: A tuple containing the rotation angle (R), translation in the x-axis (x), translation in the y-axis (y),
            and the registration metric.

        Raises:
            AssertionError: If the number of fixed points does not match the number of moving points.
        """
        # Transfer images to GPU memory
        def to_gpu(image):
            array = sitk.GetArrayFromImage(image)
            gpu_array = cp.asarray(array)
            return sitk.GetImageFromArray(cp.asnumpy(gpu_array))

        
        if use_points:
            fixed_point_file = os.path.join(self.registration_output, f'{fixed_index}_points.txt')
            moving_point_file = os.path.join(self.registration_output, f'{moving_index}_points.txt')
            if os.path.exists(fixed_point_file) and os.path.exists(moving_point_file):
                print(f'Found fixed point file: {os.path.basename(os.path.normpath(fixed_point_file))}', end=" ")
                print(f'and moving point file: {os.path.basename(os.path.normpath(moving_point_file))}')
                with open(fixed_point_file, 'r') as fp:
                    fixed_count = len(fp.readlines())
                with open(moving_point_file, 'r') as fp:
                    moving_count = len(fp.readlines())
                assert fixed_count == moving_count, \
                        f'Error, the number of fixed points in {fixed_point_file} do not match {moving_point_file}'
            else:
                return 0, 0, 0, 0

        # Use CUDA-enabled SimpleITK
        #sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(16) #maybe host-dependent?

        if self.debug:
            sitk.ProcessObject.SetGlobalDefaultDebug(True)

        elastixImageFilter = sitk.ElastixImageFilter()
        fixed_file = os.path.join(self.input, f"{fixed_index}.tif")
        fixed = sitk.ReadImage(fixed_file, self.pixelType)

        moving_file = os.path.join(self.input, f"{moving_index}.tif")
        moving = sitk.ReadImage(moving_file, self.pixelType)

        fixed_gpu = to_gpu(fixed)
        moving_gpu = to_gpu(moving)

        # Set the images in the filter
        elastixImageFilter.SetFixedImage(fixed_gpu)
        elastixImageFilter.SetMovingImage(moving_gpu)

        rigid_params = create_rigid_parameters(elastixImageFilter, debug=self.debug)
        elastixImageFilter.SetParameterMap(rigid_params)

        if use_points:
            elastixImageFilter.SetParameter("Registration", ["MultiMetricMultiResolutionRegistration"])
            elastixImageFilter.SetParameter("Metric",  ["AdvancedMattesMutualInformation", "CorrespondingPointsEuclideanDistanceMetric"])
            elastixImageFilter.SetParameter("Metric0Weight", ["0.25"]) # the weight of 1st metric for each resolution
            elastixImageFilter.SetParameter("Metric1Weight",  ["0.75"]) # the weight of 2nd metric
            elastixImageFilter.SetFixedPointSetFileName(fixed_point_file)
            elastixImageFilter.SetMovingPointSetFileName(moving_point_file)

        elastixImageFilter.SetLogToFile(True)
        logpath = os.path.join(self.registration_output, 'iteration_logs')
        os.makedirs(logpath, exist_ok=True)
        elastixImageFilter.SetOutputDirectory(logpath)        

        elastixImageFilter.LogToConsoleOff()
        if self.debug:
            elastixImageFilter.PrintParameterMap()
        
        # Execute the registration on GPU
        elastixImageFilter.Execute()

        R, x, y = elastixImageFilter.GetTransformParameterMap()[0]["TransformParameters"]
        metric = self.get_metric(logpath)

        # Transfer results back to CPU if needed
        R = cp.asnumpy(R)
        x = cp.asnumpy(x)
        y = cp.asnumpy(y)
        metric = cp.asnumpy(metric)

        return float(R), float(x), float(y), float(metric)

    def align_elastix_with_affine(self, fixed_index, moving_index):
        elastixImageFilter = sitk.ElastixImageFilter()
        fixed_file = os.path.join(self.input, f"{fixed_index}.tif")
        fixed = sitk.ReadImage(fixed_file, self.pixelType)

        moving_file = os.path.join(self.input, f"{moving_index}.tif")
        moving = sitk.ReadImage(moving_file, self.pixelType)
        elastixImageFilter.SetFixedImage(fixed)
        elastixImageFilter.SetMovingImage(moving)

        defaultMap = elastixImageFilter.GetDefaultParameterMap("affine")
        elastixImageFilter.SetParameterMap(defaultMap)
        elastixImageFilter.LogToConsoleOff()
        elastixImageFilter.PrintParameterMap()
        elastixImageFilter.Execute()

        results = elastixImageFilter.GetTransformParameterMap()[0]["TransformParameters"]
        # rigid = ('0.00157607', '-0.370347', '-3.21588')
        # affine = ('1.01987', '-0.00160559', '-0.000230038', '1.02641', '-1.98157', '-1.88057')
        print(results)


    def create_affine_transformations(self):
        image_manager = ImageManager(self.input)
        # files = sorted(os.listdir(self.input))
        # nfiles = len(files)
        # midpoint = nfiles // 2
        transformation_to_previous_sec = {}
        # center = image_manager.center

        for i in tqdm(range(1, image_manager.len_files)):
            fixed_index = os.path.splitext(image_manager.files[i - 1])[0]
            moving_index = os.path.splitext(image_manager.files[i])[0]
            elastixImageFilter = sitk.ElastixImageFilter()
            fixed_file = os.path.join(self.input, f"{fixed_index}.tif")
            fixed = sitk.ReadImage(fixed_file, self.pixelType)            
            moving_file = os.path.join(self.input, f"{moving_index}.tif")
            moving = sitk.ReadImage(moving_file, self.pixelType)
            elastixImageFilter.SetFixedImage(fixed)
            elastixImageFilter.SetMovingImage(moving)

            affineParameterMap = elastixImageFilter.GetDefaultParameterMap("affine")
            affineParameterMap["UseDirectionCosines"] = ["true"]
            affineParameterMap["MaximumNumberOfIterations"] = ["250"] # 250 works ok
            affineParameterMap["MaximumNumberOfSamplingAttempts"] = ["10"]
            affineParameterMap["NumberOfResolutions"]= ["4"] # Takes lots of RAM
            affineParameterMap["WriteResultImage"] = ["false"]
            elastixImageFilter.SetParameterMap(affineParameterMap)
            elastixImageFilter.LogToConsoleOff()
            # elastixImageFilter.PrintParameterMap()
            elastixImageFilter.Execute()

            a11 , a12 , a21 , a22 , tx , ty = elastixImageFilter.GetTransformParameterMap()[0]["TransformParameters"]
            R = np.array([[a11, a12], [a21, a22]], dtype=np.float64)
            shift = image_manager.center + (float(tx), float(ty)) - np.dot(R, image_manager.center)
            A = np.vstack([np.column_stack([R, shift]), [0, 0, 1]]).astype(np.float64)
            transformation_to_previous_sec[i] = A

        for moving_index in tqdm(range(image_manager.len_files)):
            filename = str(moving_index).zfill(3) + ".tif"
            if moving_index == image_manager.midpoint:
                transformation = np.eye(3)
            elif moving_index < image_manager.midpoint:
                T_composed = np.eye(3)
                for i in range(image_manager.midpoint, moving_index, -1):
                    T_composed = np.dot(
                        np.linalg.inv(transformation_to_previous_sec[i]), T_composed
                    )
                transformation = T_composed
            else:
                T_composed = np.eye(3)
                for i in range(image_manager.midpoint + 1, moving_index + 1):
                    T_composed = np.dot(transformation_to_previous_sec[i], T_composed)
                transformation = T_composed

            # print(filename, transformation)
            infile = os.path.join(self.input, filename)
            outfile = os.path.join(self.output, filename)
            file_key = [infile, outfile, transformation]
            align_image_to_affine(file_key)
            #####self.transform_save_image(infile, outfile, transformation)

    def create_dir2dir_transformations(self):
        """Calculate and store the rigid transformation using elastix.  
        Align CH3 from CH1
        TODO this needs to be fixed to account for the channel variable name
        """
        MOVING_DIR = os.path.join(self.fileLocationManager.prep, 'CH3', 'thumbnail_cropped')
        FIXED_DIR = self.fileLocationManager.get_thumbnail_aligned(channel=2)
        image_manager = ImageManager(MOVING_DIR)
        self.output = self.fileLocationManager.get_thumbnail_aligned(channel=3)
        os.makedirs(self.output, exist_ok=True)
        moving_files = image_manager.files

        file_keys = []

        for file in moving_files:
            moving_index = str(file).replace(".tif","")
            moving_file = os.path.join(MOVING_DIR, file)
            fixed_file = os.path.join(FIXED_DIR, file)

            if self.sqlController.check_elastix_row(self.animal, moving_index):
                rotation, xshift, yshift = self.load_elastix_transformation(self.animal, moving_index)
            else:
                fixed_arr = read_image(fixed_file)
                fixed_arr = normalize_image(fixed_arr)
                fixed_arr = equalized(fixed_arr)
                fixed = sitk.GetImageFromArray(fixed_arr)

                moving_arr = read_image(moving_file)
                moving_arr = normalize_image(moving_arr)
                moving_arr = equalized(moving_arr)
                moving = sitk.GetImageFromArray(moving_arr)
                start_time = timer()
                rotation, xshift, yshift, metric = self.align_elastix(fixed, moving, use_points=False)
                end_time = timer()
                total_elapsed_time = round((end_time - start_time),2)
                print(f"Moving index={moving_index} took {total_elapsed_time} seconds")

                print(f" took {total_elapsed_time} seconds")
                self.sqlController.add_elastix_row(self.animal, moving_index, rotation, xshift, yshift)

            T = parameters_to_rigid_transform(rotation, xshift, yshift, image_manager.center)

            infile = moving_file
            outfile = os.path.join(self.output, file)
            if os.path.exists(outfile):
                continue
            file_keys.append([infile, outfile, T])

        workers = self.get_nworkers()
        self.run_commands_concurrently(align_image_to_affine, file_keys, workers)

    def apply_full_transformations(self, channel=1):
        """Calculate and store the rigid transformation using elastix.  
        Align CH3 from CH1
        """
        self.input = os.path.join(self.fileLocationManager.prep, 'CH3', 'full_cropped')
        self.output = self.fileLocationManager.get_full_aligned(channel=channel)
        os.makedirs(self.output, exist_ok=True)
        files = sorted(os.listdir(self.input))
        center = self.get_rotation_center(channel=channel)
        file_keys = []

        for file in files:
            moving_index = str(file).replace(".tif","")
            rotation, xshift, yshift = self.load_elastix_transformation(self.animal, moving_index)

            T = parameters_to_rigid_transform(rotation, xshift, yshift, center)
            Ts = create_scaled_transform(T)
            infile = os.path.join(self.input, file)
            outfile = os.path.join(self.output, file)
            if os.path.exists(outfile):
                continue

            file_key = [infile, outfile, Ts]
            file_keys.append(file_key)

        workers = self.get_nworkers()
        self.run_commands_concurrently(align_image_to_affine, file_keys, workers)

    def load_elastix_transformation(self, animal, moving_index):
        """loading the elastix transformation from the database

        :param animal: (str) Animal ID
        :param moving_index: (int) index of moving section

        :return array: 2*2 roatation matrix, float: x translation, float: y translation
        """
        elastixTransformation = self.sqlController.get_elastix_row(animal, moving_index)
        if elastixTransformation is None:
            print(f'No value for {animal} at moving index={moving_index}')
            return 0, 0, 0

        R = elastixTransformation.rotation
        xshift = elastixTransformation.xshift
        yshift = elastixTransformation.yshift
        return R, xshift, yshift

    def get_rotation_center(self):
        """return a rotation center for finding the parameters of a transformation from the transformation matrix
        use channel 1 thumbnail cropped images to find the center

        :return list: list of x and y for rotation center that set as the midpoint of the section that is in the middle of the stack
        """

        self.input = self.fileLocationManager.get_thumbnail_cropped(channel=1)
        image_manager = ImageManager(self.input)
        return image_manager.center

    def transform_image(self, img, T):
        matrix = T[:2,:2]
        offset = T[:2,2]
        offset = np.flip(offset)
        img = affine_transform(img, matrix.T, offset)
        return img

    def transform_save_image(self, infile, outfile, T):
        matrix = T[:2,:2]
        offset = T[:2,2]
        offset = np.flip(offset)
        img = read_image(infile)
        img = affine_transform(img, matrix.T, offset)
        write_image(outfile, img)

    def get_transformations(self):
        """After the elastix job is done, this fetches the rotation, xshift and yshift from the DB
        
        :param animal: the animal
        :return: a dictionary of key=filename, value = coordinates
        """

        self.input = self.fileLocationManager.get_thumbnail_cropped(channel=1) # there is no need to get cropped images from somewhere else
        try:
            files = os.listdir(self.input)
        except OSError:
            print(f"Error: Could not find the input directory: {self.input}")
            return

        transformation_to_previous_sec = {}
        image_manager = ImageManager(self.input)
        center = image_manager.center
        midpoint = image_manager.midpoint 

        for i in range(1, len(files)):
            rotation, xshift, yshift = self.load_elastix_transformation(self.animal, i)
            T = parameters_to_rigid_transform(rotation, xshift, yshift, center)
            transformation_to_previous_sec[i] = T

        transformations = {}

        if self.debug:
            print(f'elastix_manager::get_transformations #files={len(files)} in {self.input}')
            print(f'#transformation_to_previous_sec={len(transformation_to_previous_sec)}')

        for moving_index in range(len(files)):
            filename = str(moving_index).zfill(3) + ".tif"
            if moving_index == midpoint:
                transformations[filename] = np.eye(3)
            elif moving_index < midpoint:
                T_composed = np.eye(3)
                for i in range(midpoint, moving_index, -1):
                    T_composed = np.dot(
                        np.linalg.inv(transformation_to_previous_sec[i]), T_composed
                    )
                transformations[filename] = T_composed
            else:
                T_composed = np.eye(3)
                for i in range(midpoint + 1, moving_index + 1):
                    T_composed = np.dot(transformation_to_previous_sec[i], T_composed)
                transformations[filename] = T_composed
        return transformations

    def start_image_alignment(self):
        """align the full resolution tif images with the transformations provided.
           All the sections are aligned to the middle sections, the transformation
           of a given section to the middle section is the composite of the transformation
           from the given section through all the intermediate sections to the middle sections.

        :param transforms: (dict): dictionary of transformations that are index by the id of moving sections
        """
        transformations = self.get_transformations()

        if self.downsample:
            self.input, self.output = (self.fileLocationManager.get_alignment_directories(channel=self.channel, resolution="thumbnail"))
        else:
            transformations = create_downsampled_transforms(transformations, downsample=False, scaling_factor=SCALING_FACTOR)
            self.input, self.output = self.fileLocationManager.get_alignment_directories(channel=self.channel, resolution='full')

        try:
            starting_files = os.listdir(self.input)
        except OSError:
            print(f"Error: Could not find the input directory: {self.input}")
            return
        
        print(f"Aligning images from {os.path.basename(self.input)} to {os.path.basename(self.output)}")

        if len(starting_files) != len(transformations):
            print("Error: The number of files in the input directory does not match the number of transformations")
            print(f"Alignment file count: {len(starting_files)} with {len(transformations)} transforms")
            print(f"Alignment input folder: {self.input}")
            print(f"Alignment output output: {self.output}")
            return
        
        self.align_images(transformations)


    def align_section_masks(self, animal, transforms):
        """function that can be used to align the masks used for cleaning the image.  
        This not run as part of the pipeline, but is used to create the 3d shell 
        around a certain brain

        :param animal: (str) Animal ID
        :param transforms: (array): 3*3 transformation array
        """
        fileLocationManager = FileLocationManager(animal)
        self.input = fileLocationManager.rotated_and_padded_thumbnail_mask
        self.output = fileLocationManager.rotated_and_padded_and_aligned_thumbnail_mask
        self.align_images(transforms)

    def align_images(self, transforms):
        """function to align a set of images with a with the transformations between them given
        Note: image alignment is memory intensive (but all images are same size)
        6 factor of est. RAM per image for clean/transform needs firmed up but safe

        :param transforms (dict): dictionary of transformations indexed by id of moving sections
        """
        image_manager = ImageManager(self.input)        
        self.bgcolor = image_manager.get_bgcolor(self.maskpath)

        os.makedirs(self.output, exist_ok=True)
        transforms = OrderedDict(sorted(transforms.items()))
        first_file_name = list(transforms.keys())[0]
        infile = os.path.join(self.input, first_file_name)
        file_keys = []
        for file, T in transforms.items():
            infile = os.path.join(self.input, file)
            outfile = os.path.join(self.output, file)
            if os.path.exists(outfile):
                continue
            file_keys.append([infile, outfile, T, self.bgcolor])

        workers = self.get_nworkers() // 2
        start_time = timer()
        if self.debug:
            print(f'def align_images has {len(file_keys)} file keys')
        self.run_commands_concurrently(align_image_to_affine, file_keys, workers)
        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        print(f'took {total_elapsed_time} seconds.')

    def create_web_friendly_sections(self):
        """A function to create section PNG files for the database portal.
        """

        fileLocationManager = FileLocationManager(self.animal)
        self.input = fileLocationManager.get_thumbnail_aligned(channel=self.channel)
        self.output = fileLocationManager.section_web

        os.makedirs(self.output, exist_ok=True)
        files = sorted(os.listdir(self.input))
        file_keys = []
        for file in files:
            base, _ = os.path.splitext(file)
            png_file = base + ".png"
            outfile = os.path.join(self.output, png_file)
            if not os.path.exists(outfile):
                infile = os.path.join(self.input, file)    
                file_keys.append((infile, outfile))
        workers = self.get_nworkers()
        self.run_commands_concurrently(tif_to_png, file_keys, workers)

    @staticmethod
    def get_metric(logpath):
        metric_value = None
        filepath = os.path.join(logpath, 'IterationInfo.0.R4.txt')
        if os.path.exists(filepath):
            with open(filepath) as infile:
                last_line = infile.readlines()[-1]
                metric_value = last_line.split('\t')[1]
        if metric_value is None:
            metric_value = 0
        return metric_value

    @staticmethod
    def parameter_elastix_parameter_file_to_dict(filename):
        d = {}
        with open(filename, 'r') as f:
            for line in f.readlines():
                if line.startswith('('):
                    tokens = line[1:-2].split(' ')
                    key = tokens[0]
                    if len(tokens) > 2:
                        value = []
                        for v in tokens[1:]:
                            try:
                                value.append(float(v))
                            except ValueError:
                                value.append(v)
                    else:
                        v = tokens[1]
                        try:
                            value = (float(v))
                        except ValueError:
                            value = v
                    d[key] = value

            return d
