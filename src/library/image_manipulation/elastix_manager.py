"""This module takes care of the section to section alignment. It imports
libraries that contain the code from the elastix command line tools:
https://elastix.lumc.nl/
The libraries are contained within the SimpleITK-SimpleElastix library
"""

import os
import numpy as np
from collections import OrderedDict
from sqlalchemy.orm.exc import NoResultFound
from PIL import Image
from timeit import default_timer as timer

Image.MAX_IMAGE_PIXELS = None
from timeit import default_timer as timer
from subprocess import Popen, PIPE
from pathlib import Path
import SimpleITK as sitk
from scipy.ndimage import affine_transform


from library.database_model.elastix_transformation import ElastixTransformation
from library.image_manipulation.filelocation_manager import FileLocationManager
from library.image_manipulation.file_logger import FileLogger
from library.utilities.utilities_process import get_image_size, read_image
from library.utilities.utilities_mask import equalized, normalize_image
from library.utilities.utilities_registration import (
    align_elastix,
    align_image_to_affine,
    create_downsampled_transforms,
    create_scaled_transform,
    parameters_to_rigid_transform,
    tif_to_png,
)


class ElastixManager(FileLogger):
    """Class for generating, storing and applying transformations within 
    stack alignment [with the Elastix package]
    All methods relate to aligning images in stack
    """

    def __init__(self, iteration=0):
        self.iteration = iteration
        LOGFILE_PATH = self.fileLocationManager.stack
        self.pixelType = sitk.sitkFloat32
        super().__init__(LOGFILE_PATH)




    def create_within_stack_transformations(self):
        """Calculate and store the rigid transformation using elastix.  
        The transformations are calculated from the next image to the previous
        This is done in a simple loop with no workers. Usually takes
        up to an hour to run for a stack. It only needs to be run once for
        each brain. We are now using multiple iterations to get better alignment.
        The 2nd pass uses the results of the 1st pass to align
        """
        if self.channel == 1 and self.downsample:
            INPUT, _ = self.fileLocationManager.get_alignment_directories(iteration=self.iteration,
                                                                          iterations=self.iterations,
                                                                          channel=1, resolution='thumbnail')
            files = sorted(os.listdir(INPUT))
            nfiles = len(files)
            self.logevent(f"INPUT FOLDER: {INPUT}")
            self.logevent(f"FILE COUNT: {nfiles}")
            for i in range(1, nfiles):
                fixed_index = os.path.splitext(files[i - 1])[0]
                moving_index = os.path.splitext(files[i])[0]
                if not self.sqlController.check_elastix_row(self.animal, moving_index, self.iteration):
                    self.calculate_elastix_transformation(INPUT, fixed_index, moving_index)


    def create_dir2dir_transformations(self):
        """Calculate and store the rigid transformation using elastix.  
        Align CH3 from CH1
        """
        MOVING_DIR = os.path.join(self.fileLocationManager.prep, 'CH3', 'thumbnail_cleaned')
        FIXED_DIR = self.fileLocationManager.get_thumbnail_aligned(channel=2)
        OUTPUT = self.fileLocationManager.get_thumbnail_aligned(channel=3)
        os.makedirs(OUTPUT, exist_ok=True)
        moving_files = sorted(os.listdir(MOVING_DIR))
        files = sorted(os.listdir(MOVING_DIR))
        midpoint = len(files) // 2
        midfilepath = os.path.join(MOVING_DIR, files[midpoint])
        width, height = get_image_size(midfilepath)
        center = np.array([width, height]) / 2

        file_keys = []

        for file in moving_files:
            moving_index = str(file).replace(".tif","")
            moving_file = os.path.join(MOVING_DIR, file)
            fixed_file = os.path.join(FIXED_DIR, file)

            if self.sqlController.check_elastix_row(self.animal, moving_index, self.iteration):
                rotation, xshift, yshift = self.load_elastix_transformation(self.animal, moving_index, self.iteration)
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
                rotation, xshift, yshift = align_elastix(fixed, moving)
                end_time = timer()
                total_elapsed_time = round((end_time - start_time),2)
                print(f"Moving index={moving_index} took {total_elapsed_time} seconds")

                print(f" took {total_elapsed_time} seconds")
                self.sqlController.add_elastix_row(self.animal, moving_index, rotation, xshift, yshift, self.iteration)

            T = parameters_to_rigid_transform(rotation, xshift, yshift, center)

            infile = moving_file
            outfile = os.path.join(OUTPUT, file)
            if os.path.exists(outfile):
                continue
            file_keys.append([infile, outfile, T])

        workers = self.get_nworkers()
        self.run_commands_concurrently(align_image_to_affine, file_keys, workers)
            

    def apply_full_transformations(self, channel=1):
        """Calculate and store the rigid transformation using elastix.  
        Align CH3 from CH1
        """
        INPUT = os.path.join(self.fileLocationManager.prep, 'CH3', 'full_cleaned')
        OUTPUT = self.fileLocationManager.get_full_aligned(channel=channel)
        os.makedirs(OUTPUT, exist_ok=True)
        files = sorted(os.listdir(INPUT))
        center = self.get_rotation_center(channel=channel)
        file_keys = []

        for file in files:
            moving_index = str(file).replace(".tif","")
            rotation, xshift, yshift = self.load_elastix_transformation(self.animal, moving_index, self.iteration)

            T = parameters_to_rigid_transform(rotation, xshift, yshift, center)
            Ts = create_scaled_transform(T)
            infile = os.path.join(INPUT, file)
            outfile = os.path.join(OUTPUT, file)
            if os.path.exists(outfile):
                continue

            file_key = [infile, outfile, Ts]
            file_keys.append(file_key)
            align_image_to_affine(file_key)

        workers = self.get_nworkers()
        self.run_commands_concurrently(align_image_to_affine, file_keys, workers)


    def call_alignment_metrics(self):
        """For the metrics value that shows how well the alignment went, 
        elastix needs to be called as a separate program.
        """

        if self.channel == 1 and self.downsample:

            INPUT, _ = self.fileLocationManager.get_alignment_directories(iteration=self.iteration,
                                                                               iterations=self.iterations,
                                                                               channel=1, resolution='thumbnail')

            ELASTIX_OUTPUT = self.fileLocationManager.elastix
            os.makedirs(ELASTIX_OUTPUT, exist_ok=True)

            files = sorted(os.listdir(INPUT))
            nfiles = len(files)

            PIPELINE_ROOT = Path('./src/pipeline/scripts').absolute().as_posix()
            program = os.path.join(PIPELINE_ROOT, 'create_alignment_metrics.py')

            for i in range(1, nfiles):
                fixed_index = os.path.splitext(files[i - 1])[0]
                moving_index = os.path.splitext(files[i])[0]
                fixed_file = os.path.join(INPUT, f"{fixed_index}.tif")
                moving_file = os.path.join(INPUT, f"{moving_index}.tif")

                if not self.sqlController.check_elastix_metric_row(self.animal, moving_index, self.iteration): 
                    p = Popen(['python', program, self.animal, fixed_file, moving_file], stdin=PIPE, stdout=PIPE, stderr=PIPE)
                    output, error = p.communicate(b"input data that is passed to subprocess' stdin")

                    if len(output) > 0:
                        metric =  float(''.join(c for c in str(output) if (c.isdigit() or c =='.' or c == '-')))
                        updates = {'metric':metric}
                        self.sqlController.update_elastix_row(self.animal, moving_index, self.iteration, updates)

    def calculate_elastix_transformation(self, INPUT, fixed_index, moving_index):
        """Calculates the rigid transformation from the Elastix output
        and adds it to the database.

        :param INPUT: path of the files
        :param fixed_index: index of fixed image
        :param moving_index: index of moving image
        """
        # start register simple
        
        fixed_file = os.path.join(INPUT, f"{fixed_index}.tif")
        fixed = sitk.ReadImage(fixed_file, self.pixelType)
        moving_file = os.path.join(INPUT, f"{moving_index}.tif")
        moving = sitk.ReadImage(moving_file, self.pixelType)

        """
        initial_transform = sitk.CenteredTransformInitializer(self.midfixed, moving, 
            sitk.Euler2DTransform(), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY)

        moving = sitk.Resample(moving, self.midfixed, initial_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
        """

        rotation, xshift, yshift = align_elastix(fixed, moving)
        self.sqlController.add_elastix_row(self.animal, moving_index, rotation, xshift, yshift, self.iteration)


    def calculate_elastix_channels(self, INPUT, fixed_index, moving_index):
        """Calculates the rigid transformation from the Elastix output
        and adds it to the database.

        :param INPUT: path of the files
        :param fixed_index: index of fixed image
        :param moving_index: index of moving image
        """

        # start register simple
        pixelType = sitk.sitkFloat32
        fixed_file = os.path.join(INPUT, f"{fixed_index}.tif")
        moving_file = os.path.join(INPUT, f"{moving_index}.tif")
        fixed = sitk.ReadImage(fixed_file, pixelType)
        moving = sitk.ReadImage(moving_file, pixelType)
        rotation, xshift, yshift = align_elastix(fixed, moving)
        self.sqlController.add_elastix_row(self.animal, moving_index, rotation, xshift, yshift, self.iteration)



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

    def load_elastix_transformation(self, animal, moving_index, iteration):
        """loading the elastix transformation from the database

        :param animal: (str) Animal ID
        :param moving_index: (int) index of moving section

        :return array: 2*2 roatation matrix, float: x translation, float: y translation
        """

        try:
            elastixTransformation = (
                self.sqlController.session.query(ElastixTransformation)
                .filter(ElastixTransformation.FK_prep_id == animal)
                .filter(ElastixTransformation.iteration == iteration)
                .filter(ElastixTransformation.section == moving_index)
                .one()
            )
        except NoResultFound as nrf:
            print("No value for {} {} error: {}".format(animal, moving_index, nrf))
            return 0, 0, 0

        R = elastixTransformation.rotation
        xshift = elastixTransformation.xshift
        yshift = elastixTransformation.yshift
        return R, xshift, yshift


    def get_rotation_center(self, channel=1):
        """return a rotation center for finding the parameters of a transformation from the transformation matrix

        :return list: list of x and y for rotation center that set as the midpoint of the section that is in the middle of the stack
        """

        INPUT = self.fileLocationManager.get_thumbnail_cleaned(channel)
        files = sorted(os.listdir(INPUT))
        midpoint = len(files) // 2
        midfilepath = os.path.join(INPUT, files[midpoint])
        width, height = get_image_size(midfilepath)
        center = np.array([width, height]) / 2
        return center
    
    def transform_image(self, img, T):
        matrix = T[:2,:2]
        offset = T[:2,2]
        offset = np.flip(offset)
        img = affine_transform(img, matrix.T, offset)
        return img



    def get_transformations(self):
        """After the elastix job is done, this goes into each subdirectory and parses the Transformation.0.txt file
        
        :param animal: the animal
        :return: a dictionary of key=filename, value = coordinates
        """

        INPUT = self.fileLocationManager.get_thumbnail_cleaned(1)
        files = sorted(os.listdir(INPUT))
        midpoint = len(files) // 2
        transformation_to_previous_sec = {}
        center = self.get_rotation_center()

        for i in range(1, len(files)):
            rotation, xshift, yshift = self.load_elastix_transformation(self.animal, i, self.iteration)
            T = parameters_to_rigid_transform(rotation, xshift, yshift, center)
            transformation_to_previous_sec[i] = T

        transformations = {}

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


    def align_full_size_image(self, transforms):
        """align the full resolution tif images with the transformations provided.
           All the sections are aligned to the middle sections, the transformation
           of a given section to the middle section is the composite of the transformation
           from the given section through all the intermediate sections to the middle sections.

        :param transforms: (dict): dictionary of transformations that are index by the id of moving sections
        """
        if not self.downsample:
            transforms = create_downsampled_transforms(transforms, downsample=False)
            INPUT, OUTPUT = self.fileLocationManager.get_alignment_directories(iteration=self.iteration,
                                                                               iterations=self.iterations,
                                                                               channel=1, resolution='full')

            self.logevent(f"INPUT FOLDER: {INPUT}")
            starting_files = os.listdir(INPUT)
            self.logevent(f"FILE COUNT: {len(starting_files)}")
            self.logevent(f"OUTPUT FOLDER: {OUTPUT}")
            self.align_images(INPUT, OUTPUT, transforms)


    def align_downsampled_images(self, transforms):
        """align the downsample tiff images

        :param transforms: (dict) dictionary of transformations indexed by id of moving sections
        """

        if self.downsample:
            INPUT, OUTPUT = self.fileLocationManager.get_alignment_directories(iteration=self.iteration,
                                                                               iterations=self.iterations,
                                                                               channel=1, resolution='thumbnail')

            print(f'Aligning {len(os.listdir(INPUT))} images from {os.path.basename(os.path.normpath(INPUT))} to {os.path.basename(os.path.normpath(OUTPUT))}', end=" ")
            self.align_images(INPUT, OUTPUT, transforms)


    def align_section_masks(self, animal, transforms):
        """function that can be used to align the masks used for cleaning the image.  
        This not run as part of the pipeline, but is used to create the 3d shell 
        around a certain brain

        :param animal: (str) Animal ID
        :param transforms: (array): 3*3 transformation array
        """
        fileLocationManager = FileLocationManager(animal)
        INPUT = fileLocationManager.rotated_and_padded_thumbnail_mask
        OUTPUT = fileLocationManager.rotated_and_padded_and_aligned_thumbnail_mask
        self.align_images(INPUT, OUTPUT, transforms)


    def align_images(self, INPUT, OUTPUT, transforms):
        """function to align a set of images with a with the transformations between them given
        Note: image alignment is memory intensive (but all images are same size)
        6 factor of est. RAM per image for clean/transform needs firmed up but safe

        :param INPUT: (str) directory of images to be aligned
        :param OUTPUT (str): directory output the aligned images
        :param transforms (dict): dictionary of transformations indexed by id of moving sections
        """

        os.makedirs(OUTPUT, exist_ok=True)
        transforms = OrderedDict(sorted(transforms.items()))
        first_file_name = list(transforms.keys())[0]
        infile = os.path.join(INPUT, first_file_name)
        file_keys = []
        for file, T in transforms.items():
            infile = os.path.join(INPUT, file)
            outfile = os.path.join(OUTPUT, file)
            if os.path.exists(outfile):
                continue
            file_keys.append([infile, outfile, T])

        workers = self.get_nworkers() // 2
        start_time = timer()
        self.run_commands_concurrently(align_image_to_affine, file_keys, workers)
        end_time = timer()
        total_elapsed_time = round((end_time - start_time),2)
        print(f'took {total_elapsed_time} seconds.')


    def create_web_friendly_sections(self):
        """A function to create section PNG files for the database portal.
        """

        fileLocationManager = FileLocationManager(self.animal)
        INPUT = fileLocationManager.get_thumbnail_aligned(channel=1)
        OUTPUT = fileLocationManager.section_web

        os.makedirs(OUTPUT, exist_ok=True)
        files = sorted(os.listdir(INPUT))
        file_keys = []
        for file in files:
            png = str(file).replace(".tif", ".png")
            infile = os.path.join(INPUT, file)
            outfile = os.path.join(OUTPUT, png)
            if os.path.exists(outfile):
                continue
            file_keys.append([infile, outfile])

        workers = self.get_nworkers()
        self.run_commands_concurrently(tif_to_png, file_keys, workers)


