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
Image.MAX_IMAGE_PIXELS = None
from timeit import default_timer as timer
from subprocess import Popen, PIPE
from pathlib import Path

from lib.FileLocationManager import FileLocationManager
from utilities.utilities_alignment import align_image_to_affine, create_downsampled_transforms
from utilities.utilities_registration import (
    register_simple,
    parameters_to_rigid_transform,
    rigid_transform_to_parmeters,
)
from database_model.elastix_transformation import ElastixTransformation
from utilities.utilities_process import get_image_size


class ElastixManager:
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
        """
        if self.channel == 1 and self.downsample:
            INPUT = os.path.join(self.fileLocationManager.prep, "CH1", "thumbnail_cleaned")
            files = sorted(os.listdir(INPUT))
            nfiles = len(files)
            self.fileLogger.logevent(f"INPUT FOLDER: {INPUT}")
            self.fileLogger.logevent(f"FILE COUNT: {nfiles}")

            for i in range(1, nfiles):
                fixed_index = os.path.splitext(files[i - 1])[0]
                moving_index = os.path.splitext(files[i])[0]
                if not self.sqlController.check_elastix_row(self.animal, moving_index):
                    self.calculate_elastix_transformation(INPUT, fixed_index, moving_index)
                    

    def call_alignment_metrics(self):
        if self.channel == 1 and self.downsample:
            INPUT = os.path.join(self.fileLocationManager.prep, "CH1", "thumbnail_cleaned")
            files = sorted(os.listdir(INPUT))
            nfiles = len(files)
            self.fileLogger.logevent(f"INPUT FOLDER: {INPUT}")
            self.fileLogger.logevent(f"FILE COUNT: {nfiles}")

            PIPELINE_ROOT = Path('./pipeline').absolute().as_posix()
            program = os.path.join(PIPELINE_ROOT, 'create_alignment_metrics.py')

            for i in range(1, nfiles):
                fixed_index = os.path.splitext(files[i - 1])[0]
                moving_index = os.path.splitext(files[i])[0]
                fixed_file = os.path.join(INPUT, f"{fixed_index}.tif")
                moving_file = os.path.join(INPUT, f"{moving_index}.tif")
                if not self.sqlController.check_elastix_metric_row(self.animal, moving_index): 
                    p = Popen(['python', program, fixed_file, moving_file], stdin=PIPE, stdout=PIPE, stderr=PIPE)
                    output, _ = p.communicate(b"input data that is passed to subprocess' stdin")
                    if len(output) > 0:
                        metric =  float(''.join(c for c in str(output) if (c.isdigit() or c =='.' or c == '-')))
                        updates = {'metric':metric}
                        self.sqlController.update_elastix_row(self.animal, moving_index, updates)

    def calculate_elastix_transformation(self, INPUT, fixed_index, moving_index):
        center = self.get_rotation_center()
        second_transform_parameters, initial_transform_parameters = register_simple(
            INPUT, fixed_index, moving_index)
        T1 = parameters_to_rigid_transform(*initial_transform_parameters)
        T2 = parameters_to_rigid_transform(*second_transform_parameters, center)
        T = T1 @ T2
        xshift, yshift, rotation, center = rigid_transform_to_parmeters(T, center)
        self.sqlController.add_elastix_row(
            self.animal, moving_index, rotation, xshift, yshift
        )


    def rigid_transform_to_parmeters(self, transform):
        """convert a 2d transformation matrix (3*3) to the rotation angles, rotation center and translation

        Args:
            transform (array like): 3*3 array that stores the 2*2 transformation matrix and the 1*2 translation vector for a
            2D image.  the third row of the array is a place holder of values [0,0,1].

        Returns:
            float: x translation
            float: y translation
            float: rotation angle in arc
            list:  lisf of x and y for rotation center
        """
        return rigid_transform_to_parmeters(transform, self.get_rotation_center())


    def parameters_to_rigid_transform(self, rotation: float, xshift: float, yshift: float, center):
        """convert a set of rotation parameters to the transformation matrix

        Args:
            rotation (float): rotation angle in arc
            xshift (float): translation in x
            yshift (float): translation in y
            center (list): list of x and y for the rotation center

        Returns:
            array: 3*3 transformation matrix for 2D image, contain the 2*2 array and 1*2 translation vector
        """
        return parameters_to_rigid_transform(rotation, xshift, yshift, center)


    def load_elastix_transformation(self, animal, moving_index):
        """loading the elastix transformation from the database

        Args:
            animal (str): Animal ID
            moving_index (int): index of moving section

        Returns:
            array: 2*2 roatation matrix
            float: x translation
            float: y translation
        """
        try:
            elastixTransformation = (
                self.sqlController.session.query(ElastixTransformation)
                .filter(ElastixTransformation.prep_id == animal)
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


    def get_rotation_center(self):
        """return a rotation center for finding the parameters of a transformation from the transformation matrix

        Returns:
            list: list of x and y for rotation center that set as the midpoint of the section that is in the middle of the stack
        """
        INPUT = self.fileLocationManager.get_thumbnail_cleaned(1)
        files = sorted(os.listdir(INPUT))
        midpoint = len(files) // 2
        midfilepath = os.path.join(INPUT, files[midpoint])
        width, height = get_image_size(midfilepath)
        center = np.array([width, height]) / 2
        return center


    def get_transformations(self):
        """After the elastix job is done, this goes into each subdirectory and parses the Transformation.0.txt file
        
        Args:
        :animal: the animal
        Returns: a dictionary of key=filename, value = coordinates
        """

        sections = self.sqlController.get_sections(self.animal, self.channel)

        midpoint = len(sections) // 2

        transformation_to_previous_sec = {}
        center = self.get_rotation_center()

        for i in range(1, len(sections)):
            rotation, xshift, yshift = self.load_elastix_transformation(self.animal, i)
            T = self.parameters_to_rigid_transform(rotation, xshift, yshift, center)
            transformation_to_previous_sec[i] = T

        transformations = {}

        for moving_index in range(len(sections)):
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

        Args:
            transforms (dict): dictionary of transformations that are index by the id of moving sections
        """
        if not self.downsample:
            transforms = create_downsampled_transforms(transforms, downsample=False)
            INPUT = self.fileLocationManager.get_full_cleaned(self.channel)
            OUTPUT = self.fileLocationManager.get_full_aligned(self.channel)
            self.fileLogger.logevent(f"INPUT FOLDER: {INPUT}")
            starting_files = os.listdir(INPUT)
            self.fileLogger.logevent(f"FILE COUNT: {len(starting_files)}")
            self.fileLogger.logevent(f"OUTPUT FOLDER: {OUTPUT}")
            self.align_images(INPUT, OUTPUT, transforms)
            progress_id = self.sqlController.get_progress_id(
                downsample=False, channel=self.channel, action="ALIGN"
            )
            self.sqlController.set_task(self.animal, progress_id)


    def align_downsampled_images(self, transforms):
        """align the downsample tiff images

        Args:
            transforms (dict): dictionary of transformations indexed by id of moving sections
        """
        if self.downsample:
            INPUT = self.fileLocationManager.get_thumbnail_cleaned(self.channel)
            OUTPUT = self.fileLocationManager.get_thumbnail_aligned(self.channel)
            self.align_images(INPUT, OUTPUT, transforms)
            progress_id = self.sqlController.get_progress_id(
                downsample=True, channel=self.channel, action="ALIGN"
            )
            self.sqlController.set_task(self.animal, progress_id)


    def align_section_masks(self, animal, transforms):
        """function that can be used to align the masks used for cleaning the image.  This not run as part of
        the pipeline, but is used to create the 3d shell around a certain brain

        Args:
            animal (str): Animal ID
            transforms (array): 3*3 transformation array
        """
        fileLocationManager = FileLocationManager(animal)
        INPUT = fileLocationManager.rotated_and_padded_thumbnail_mask
        OUTPUT = fileLocationManager.aligned_rotated_and_padded_thumbnail_mask
        self.align_images(INPUT, OUTPUT, transforms)


    def align_images(self, INPUT, OUTPUT, transforms):
        """function to align a set of images with a with the transformations between them given

        Args:
            INPUT (str): directory of images to be aligned
            OUTPUT (str): directory output the aligned images
            transforms (dict): dictionary of transformations indexed by id of moving sections

        Note: image alignment is memory intensive (but all images are same size)
        6 factor of est. RAM per image for clean/transform needs firmed up but safe

        """
        os.makedirs(OUTPUT, exist_ok=True)
        transforms = OrderedDict(sorted(transforms.items()))
        first_file_name = list(transforms.keys())[0]
        infile = os.path.join(INPUT, first_file_name)
        file_keys = []
        for i, (file, T) in enumerate(transforms.items()):
            infile = os.path.join(INPUT, file)
            outfile = os.path.join(OUTPUT, file)
            if os.path.exists(outfile):
                continue
            file_keys.append([i, infile, outfile, T])

        workers = self.get_nworkers() // 2
        start_time = timer()
        self.run_commands_concurrently(align_image_to_affine, file_keys, workers)
        end_time = timer()
        total_elapsed_time = round((end_time - start_time),2)
        print(f'Aligning images took {total_elapsed_time} seconds.')

