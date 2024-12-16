"""This module takes care of the section to section alignment. It imports
libraries that contain the code from the elastix command line tools:
https://elastix.lumc.nl/
The libraries are contained within the SimpleITK-SimpleElastix library
"""

import glob
import os
import shutil
import sys
import numpy as np
from collections import OrderedDict
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
import SimpleITK as sitk
from scipy.ndimage import affine_transform
from tqdm import tqdm
from pathlib import Path

from library.image_manipulation.filelocation_manager import ALIGNED, CROPPED_DIR, REALIGNED, FileLocationManager
from library.utilities.utilities_process import read_image, test_dir, use_scratch_dir, write_image
from library.utilities.utilities_registration import (
    align_image_to_affine,
    create_rigid_parameters,
    parameters_to_rigid_transform,
    rescale_transformations,
    tif_to_png,
)
from library.image_manipulation.image_manager import ImageManager


class ElastixManager():
    """Class for generating, storing and applying transformations within 
    stack alignment [with the Elastix package]
    All methods relate to aligning images in stack
    """


    def create_within_stack_transformations(self):
        """Calculate and store the rigid transformation using elastix.  
        The transformations are calculated from the next image to the previous
        This is done in a simple loop with no workers. Usually takes
        up to an hour to run for a stack. It only needs to be run once for
        each brain. 
        If cuda and GPU is available, we will use it, otherwise don't. 
        Home computers may not have a GPU
        """
        if self.debug:
            print(f"DEBUG: START ElastixManager::create_within_stack_transformations with iteration={self.iteration}")

        files, nfiles, *_ = test_dir(self.animal, self.input, self.section_count, True, same_size=True)
        
        self.fileLogger.logevent(f"Input FOLDER (COUNT): {self.input} ({nfiles=})")
        
        for i in range(1, nfiles):
            fixed_index = os.path.splitext(files[i - 1])[0]
            moving_index = os.path.splitext(files[i])[0]
            if not self.sqlController.check_elastix_row(self.animal, moving_index, self.iteration):
                rotation, xshift, yshift, metric = self.align_images_elastix(fixed_index, moving_index)
                self.sqlController.add_elastix_row(self.animal, moving_index, rotation, xshift, yshift, metric, self.iteration)

    def cleanup_fiducials(self):
        self.registration_output = os.path.join(self.fileLocationManager.prep, 'registration')
        for f in Path(self.registration_output).glob('*_points.txt'):
            try:
                f.unlink()
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))
        self.sqlController.delete_elastix_iteration(self.animal, iteration=REALIGNED)
        row_count = self.sqlController.get_elastix_count(self.animal, iteration=REALIGNED)
        if row_count != 0:
            print(f'Error: {row_count} rows still exist in the database for {self.animal} after cleanup')
            sys.exit()
        if os.path.exists(self.output):
            print(f'Removing {self.output}')
            shutil.rmtree(self.output)

        use_scratch = use_scratch_dir(self.output)
        rechunkme_path = self.fileLocationManager.get_neuroglancer_rechunkme(
            self.downsample, self.channel, iteration=REALIGNED, use_scratch_dir=use_scratch)
        output = self.fileLocationManager.get_neuroglancer(self.downsample, self.channel, iteration=REALIGNED)
        progress_dir = self.fileLocationManager.get_neuroglancer_progress(self.downsample, self.channel, iteration=REALIGNED)

        if os.path.exists(rechunkme_path):
            print(f'Removing {rechunkme_path}')
            shutil.rmtree(rechunkme_path)

        if os.path.exists(output):
            print(f'Removing {output}')
            shutil.rmtree(output)

        if os.path.exists(progress_dir):
            print(f'Removing {progress_dir}')
            shutil.rmtree(progress_dir)




    def create_fiducial_points(self):
        """ Yanks the fiducial points from the database and writes them to a file
        """


        fiducials = self.sqlController.get_fiducials(self.animal, self.debug)
        nchanges = len(fiducials)
        if nchanges == 0:
            print('No fiducial points were found. Performing an extra alignment with no fiducials.')
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


    def align_images_elastix(self, fixed_index: str, moving_index: str) -> tuple[float, float, float, float]:
        """
        Aligns two images using the Elastix registration algorithm with GPU acceleration.
        expected to replace 'align_elastix' (TESTING)
        with gpu realign took 171.49 seconds.
        without gpu realign took 1 hour(s) and 46 minute(s).
        with gpu realign took 1 hour(s) and 46 minute(s).

        Args:
            fixed_index (str due to filename extraction): The index of the fixed image.
            moving_index (str due to filename extraction): The index of the moving image.
            iteration (int): if > 0, we are re-aligning

        Returns:
            tuple: A tuple containing the rotation angle (R), translation in the x-axis (x), translation in the y-axis (y),
            and the registration metric.

        Raises:
            AssertionError: If the number of fixed points does not match the number of moving points.
        """
        
        elastixImageFilter = sitk.ElastixImageFilter()
        fixed_file = os.path.join(self.input, f"{fixed_index}.tif")
        fixed = sitk.ReadImage(fixed_file, self.pixelType)

        moving_file = os.path.join(self.input, f"{moving_index}.tif")
        moving = sitk.ReadImage(moving_file, self.pixelType)

        # Set the images in the filter
        elastixImageFilter.SetFixedImage(fixed)
        elastixImageFilter.SetMovingImage(moving)

        rigid_params = create_rigid_parameters(elastixImageFilter, debug=self.debug, iteration=self.iteration)
        elastixImageFilter.SetParameterMap(rigid_params)

        if self.iteration == REALIGNED:
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
                
                elastixImageFilter.SetParameter("Registration", ["MultiMetricMultiResolutionRegistration"])
                elastixImageFilter.SetParameter("Metric",  ["AdvancedMattesMutualInformation", "CorrespondingPointsEuclideanDistanceMetric"])
                elastixImageFilter.SetParameter("Metric0Weight", ["0.25"]) # the weight of 1st metric for each resolution
                elastixImageFilter.SetParameter("Metric1Weight",  ["0.75"]) # the weight of 2nd metric
                elastixImageFilter.SetFixedPointSetFileName(fixed_point_file)
                elastixImageFilter.SetMovingPointSetFileName(moving_point_file)
            else:
                return 0.0, 0.0, 0.0, 0.0

        elastixImageFilter.SetLogToFile(True)
        elastixImageFilter.LogToConsoleOff()
        
        elastixImageFilter.SetOutputDirectory(self.logpath)        

        if self.debug and moving_index == '001':
            elastixImageFilter.PrintParameterMap()
        
        # Execute the registration on GPU
        elastixImageFilter.Execute()

        R, x, y = elastixImageFilter.GetTransformParameterMap()[0]["TransformParameters"]
        metric = self.get_metric(self.logpath)

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

        for i in tqdm(range(1, image_manager.len_files), desc="Creating rigid transformations"):
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

        for moving_index in tqdm(range(image_manager.len_files), desc="Applying rigid transformations"):
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

    def load_elastix_transformation(self, animal, moving_index, iteration):
        """loading the elastix transformation from the database

        :param animal: (str) Animal ID
        :param moving_index: (int) index of moving section

        :return array: 2*2 roatation matrix, float: x translation, float: y translation
        """
        elastixTransformation = self.sqlController.get_elastix_row(animal, moving_index, iteration)
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

        self.input = self.fileLocationManager.get_directory(channel=1, downsample=True, inpath=CROPPED_DIR)
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
        If it is full resolution, it will fetch both iterations and combine them.
        :param animal: the animal
        :return: a dictionary of key=filename, value = coordinates
        """
        if self.debug:
            print("DEBUG: START ElastixManager::get_transformations")

        transformation_to_previous_sec = {}
        image_manager = ImageManager(self.fileLocationManager.get_directory(channel=1, downsample=True, inpath=CROPPED_DIR))
        center = image_manager.center
        midpoint = image_manager.midpoint 
        print(f'Using get_transformations iteration={self.iteration} {self.input}')
        len_files = len(image_manager.files)
        for i in range(1, len_files):                
            rotation, xshift, yshift = self.load_elastix_transformation(self.animal, i, self.iteration)
            T = parameters_to_rigid_transform(rotation, xshift, yshift, center)
            transformation_to_previous_sec[i] = T

        transformations = {}

        for moving_index in range(len_files):
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

        :param transforms: (dict): dictionary of transformations that are index by the id of moving sections #THIS ARGUMENT NOT ACTUALLY PASSED TO METHOD
        """
        if self.debug:
            print("DEBUG: START ElastixManager::start_image_alignment")

        transformations = self.get_transformations()

        if not self.downsample:
            transformations = rescale_transformations(transformations)

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


    def get_alignment_status(self):
        """
        Determines the alignment status of image files for a given channel and downsample level.

        The method checks the existence of directories and counts the number of files in the 
        'aligned' and 'realigned' directories. It compares these counts with the expected counts 
        retrieved from the SQL controller to determine which input directory to use for neuroglancer.

        Returns:
            str: The alignment status, either 'ALIGNED' or 'REALIGNED'. If neither condition is met, 
             returns None.
        """
        
        iteration = None
        aligned_directory = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath='aligned')
        realigned_directory = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath='realigned')
        if not os.path.exists(realigned_directory):
            return ALIGNED
        aligned_files = os.listdir(aligned_directory)
        realigned_files = os.listdir(realigned_directory)
        aligned_count = self.sqlController.get_elastix_count(self.animal, iteration=ALIGNED) + 1
        realigned_count = self.sqlController.get_elastix_count(self.animal, iteration=REALIGNED) + 1
        if len(aligned_files) == aligned_count and len(aligned_files) > 0:
            iteration = ALIGNED
        if len(realigned_files) == realigned_count and len(realigned_files) > 0:
            iteration = REALIGNED
        return iteration 

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
        self.bgcolor = image_manager.get_bgcolor()
        print(f'align_images Using bgcolor={self.bgcolor}')
        if self.downsample and os.path.exists(self.output):
            shutil.rmtree(self.output)

        os.makedirs(self.output, exist_ok=True)
        transforms = OrderedDict(sorted(transforms.items()))
        file_keys = []
        for file, T in transforms.items():
            infile = os.path.join(self.input, file)
            outfile = os.path.join(self.output, file)
            if os.path.exists(outfile):
                continue
            file_keys.append([infile, outfile, T, self.bgcolor])

        workers = self.get_nworkers() // 2
        if self.debug:
            print(f'def align_images has {len(file_keys)} file keys')
        self.run_commands_concurrently(align_image_to_affine, file_keys, workers)

    def create_web_friendly_sections(self):
        """A function to create section PNG files for the database portal.
        """

        fileLocationManager = FileLocationManager(self.animal)
        self.input, _ = fileLocationManager.get_alignment_directories(channel=self.channel, downsample=self.downsample, iteration=self.iteration)
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
