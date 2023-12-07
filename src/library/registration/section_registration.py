""" Get better alignment for sections that don't align well
"""

import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import affine_transform
import tifffile as tiff
import copy

from library.image_manipulation.filelocation_manager import FileLocationManager
from library.image_manipulation.pipeline_process import Pipeline
from library.image_manipulation.elastix_manager import create_downsampled_transforms
from library.utilities.utilities_registration import (
    create_rigid_parameters,
    parameters_to_rigid_transform,
    rigid_transform_to_parmeters,
)

class SectionRegistration(Pipeline):
    """This class takes a fixed and moving files and aligns them
    """

    def __init__(self, animal, section, debug=False):
        #from settings import host, password, user, schema
        channel = "1"
        downsample = True
        self.debug = debug
        rescan_number = 0
        task = 'align'        
        super().__init__(animal, rescan_number, channel, downsample, 
                         task, self.debug)
        self.sections = self.sqlController.get_sections(self.animal, self.channel, rescan_number)
        self.midpoint = len(self.sections) // 2
        self.moving_index = section
        self.fixed_index = section - 1
        INPUT = os.path.join(self.fileLocationManager.prep, 'C1', 'thumbnail_cleaned')
        self.moving_file = os.path.join(INPUT, f'{str(self.moving_index).zfill(3)}.tif')
        self.fixed_file = os.path.join(INPUT, f'{str(self.fixed_index).zfill(3)}.tif')
        self.moving = tiff.imread(self.moving_file)
        self.fixed = tiff.imread(self.fixed_file)
        self.transformations = self.get_transformations()
        self.center = self.get_rotation_center()

        
        self.registration_output = os.path.join(self.fileLocationManager.prep, 'registration')
        
        self.fixed_point_file = os.path.join(self.registration_output, f'{self.fixed_index}_points.txt')
        self.moving_point_file = os.path.join(self.registration_output, f'{self.moving_index}_points.txt')
        if self.debug:
            self.iterations = "100"
        else:
            self.iterations = "2500"
 
    def setup_registration(self):
        os.makedirs(self.registration_output, exist_ok=True)
        
        fixedImage = sitk.ReadImage(self.fixed_file, sitk.sitkFloat32)
        movingImage = sitk.ReadImage(self.moving_file, sitk.sitkFloat32)
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixedImage)
        elastixImageFilter.SetMovingImage(movingImage)

        translationMap = elastixImageFilter.GetDefaultParameterMap("translation")
        elastixImageFilter.SetParameterMap(translationMap)
        rigid_params = create_rigid_parameters(elastixImageFilter)
        elastixImageFilter.AddParameterMap(rigid_params)


        if os.path.exists(self.fixed_point_file) and os.path.exists(self.moving_point_file):
            with open(self.fixed_point_file, 'r') as fp:
                fixed_count = len(fp.readlines())
            with open(self.moving_point_file, 'r') as fp:
                moving_count = len(fp.readlines())
            assert fixed_count == moving_count, \
                f'Error, the number of fixed points in {self.fixed_point_file} \
                do not match {self.moving_point_file}'

            elastixImageFilter.SetParameter("Registration", ["MultiMetricMultiResolutionRegistration"])
            elastixImageFilter.SetParameter("Metric",  ["AdvancedMattesMutualInformation", "CorrespondingPointsEuclideanDistanceMetric"])
            elastixImageFilter.SetParameter("Metric0Weight", ["0.5"]) # the weight of 1st metric for each resolution
            elastixImageFilter.SetParameter("Metric1Weight",  ["0.5"]) # the weight of 2nd metric

            elastixImageFilter.SetFixedPointSetFileName(self.fixed_point_file)
            elastixImageFilter.SetMovingPointSetFileName(self.moving_point_file)
        else:
            print('No point files')


        elastixImageFilter.SetLogToFile(True)
        elastixImageFilter.LogToConsoleOff()

        elastixImageFilter.SetParameter("WriteIterationInfo",["false"])
        elastixImageFilter.SetLogFileName('elastix.log')

        return elastixImageFilter
    
    def getMetricValue(self):
        logpath = os.path.join(self.registration_output, 'elastix.log')
        with open(logpath, "r") as fp:
            for line in self.lines_that_start_with("Final", fp):
                return line

    
    def get_transformations(self):
        transformations = {}
        center = self.get_rotation_center()
        for i in range(1, len(self.sections)):
            rotation, xshift, yshift = self.load_elastix_transformation(self.animal, i)
            T = parameters_to_rigid_transform(rotation, xshift, yshift, center)
            transformations[i] = T
        return transformations

    def get_transformation_of_section_to_midpoint(self, section, transformations):
        if section == self.midpoint:
            transformation = np.eye(3)
        elif section < self.midpoint:
            T_composed = np.eye(3)
            for i in range(self.midpoint, section, -1):
                T_composed = np.dot(np.linalg.inv(transformations[i]), T_composed)
                # print(f'midpoint={self.midpoint}, i={i}, section={section}')
            transformation = T_composed
        else:
            # original
            T_composed = np.eye(3)
            for i in range(self.midpoint + 1, section + 1):
                #print(f'midpoint={self.midpoint}, i={i}, section={section}')
                T_composed = np.dot(transformations[i], T_composed)
            transformation = T_composed
            
        
        return transformation
    
    def get_transformation_of_section(self, section):
        transformation = self.get_transformation_of_section_to_midpoint(
            section, self.transformations)
        return transformation

    def get_modified_transformation_of_section(self, fixed_index, rotation, xshift, yshift):
        T = parameters_to_rigid_transform(rotation, xshift, yshift, self.center)
        transformations = copy.copy(self.transformations)
        transformations[fixed_index] = T
        transformation = self.get_transformation_of_section_to_midpoint(
            fixed_index - 1, transformations)
        return transformation

    @staticmethod
    def lines_that_start_with(string, fp):
        return [line for line in fp if line.startswith(string)]
