"""This module takes care of the section to section alignment. It imports
libraries that contain the code from the elastix command line tools:
https://elastix.lumc.nl/
The libraries are contained within the SimpleITK-SimpleElastix library
"""

import os
import numpy as np
from collections import OrderedDict
from PIL import Image
from timeit import default_timer as timer

Image.MAX_IMAGE_PIXELS = None
from timeit import default_timer as timer
import SimpleITK as sitk
from scipy.ndimage import affine_transform


from library.image_manipulation.filelocation_manager import FileLocationManager
from library.image_manipulation.file_logger import FileLogger
from library.utilities.utilities_process import get_image_size, read_image
from library.utilities.utilities_mask import equalized, normalize_image
from library.utilities.utilities_registration import (
    align_image_to_affine,
    create_downsampled_transforms,
    create_rigid_parameters,
    create_scaled_transform,
    parameters_to_rigid_transform,
    tif_to_png,
)


class ElastixManager(FileLogger):
    """Class for generating, storing and applying transformations within 
    stack alignment [with the Elastix package]
    All methods relate to aligning images in stack
    """

    def __init__(self, LOGFILE_PATH):
        self.pixelType = sitk.sitkFloat32
        super().__init__(LOGFILE_PATH)
        self.registration_output = os.path.join(self.fileLocationManager.prep, 'registration')
        self.input, self.output = self.fileLocationManager.get_alignment_directories(channel=self.channel, resolution='thumbnail')

    def create_within_stack_transformations(self):
        """Calculate and store the rigid transformation using elastix.  
        The transformations are calculated from the next image to the previous
        This is done in a simple loop with no workers. Usually takes
        up to an hour to run for a stack. It only needs to be run once for
        each brain. 
        TODO this needs to be modified to account for the channel variable name
        """
        if self.channel == 1 and self.downsample:
            files = sorted(os.listdir(self.input))
            nfiles = len(files)
            self.logevent(f"INPUT FOLDER: {self.input}")
            self.logevent(f"FILE COUNT: {nfiles}")
            for i in range(1, nfiles):
                fixed_index = os.path.splitext(files[i - 1])[0]
                moving_index = os.path.splitext(files[i])[0]
                if not self.sqlController.check_elastix_row(self.animal, moving_index):
                    rotation, xshift, yshift = self.align_elastix_with_points(fixed_index, moving_index)
                    self.sqlController.add_elastix_row(self.animal, moving_index, rotation, xshift, yshift)

    def create_affine_transformations(self):
        files = sorted(os.listdir(self.input))
        nfiles = len(files)
        midpoint = nfiles // 2
        transformation_to_previous_sec = {}
        center = self.get_rotation_center()
              
        
        for i in range(1, nfiles):
            fixed_index = os.path.splitext(files[i - 1])[0]
            moving_index = os.path.splitext(files[i])[0]
            elastixImageFilter = sitk.ElastixImageFilter()
            fixed_file = os.path.join(self.input, f"{fixed_index}.tif")
            fixed = sitk.ReadImage(fixed_file, self.pixelType)            
            moving_file = os.path.join(self.input, f"{moving_index}.tif")
            moving = sitk.ReadImage(moving_file, self.pixelType)
            elastixImageFilter.SetFixedImage(fixed)
            elastixImageFilter.SetMovingImage(moving)

            affineParameterMap = elastixImageFilter.GetDefaultParameterMap("affine")
            affineParameterMap["UseDirectionCosines"] = ["true"]
            affineParameterMap["MaximumNumberOfIterations"] = ["500"] # 250 works ok
            affineParameterMap["MaximumNumberOfSamplingAttempts"] = ["10"]
            affineParameterMap["NumberOfResolutions"]= ["6"] # Takes lots of RAM
            affineParameterMap["WriteResultImage"] = ["false"]
            elastixImageFilter.SetParameterMap(affineParameterMap)
            elastixImageFilter.LogToConsoleOff()
            # elastixImageFilter.PrintParameterMap()
            elastixImageFilter.Execute()

            a11 , a12 , a21 , a22 , tx , ty = elastixImageFilter.GetTransformParameterMap()[0]["TransformParameters"]
            R = np.array([[a11, a12], [a21, a22]], dtype=np.float64)
            shift = center + (float(tx), float(ty)) - np.dot(R, center)
            A = np.vstack([np.column_stack([R, shift]), [0, 0, 1]]).astype(np.float64)
            transformation_to_previous_sec[i] = A

        for moving_index in range(nfiles):
            filename = str(moving_index).zfill(3) + ".tif"
            if moving_index == midpoint:
                transformation = np.eye(3)
            elif moving_index < midpoint:
                T_composed = np.eye(3)
                for i in range(midpoint, moving_index, -1):
                    T_composed = np.dot(
                        np.linalg.inv(transformation_to_previous_sec[i]), T_composed
                    )
                transformation = T_composed
            else:
                T_composed = np.eye(3)
                for i in range(midpoint + 1, moving_index + 1):
                    T_composed = np.dot(transformation_to_previous_sec[i], T_composed)
                transformation = T_composed
            
            #print(filename, transformation)
            infile = os.path.join(self.input, filename)
            outfile = os.path.join(self.output, filename)
            file_key = [infile, outfile, transformation]
            align_image_to_affine(file_key)



    def align_elastix_with_points(self, fixed_index, moving_index):
        """This takes the moving and fixed images runs Elastix on them. Note
            the huge list of parameters Elastix uses here.

            :param fixed: sitk float array for the fixed image (the image behind the moving).
            :param moving: sitk float array for the moving image.
            :return: the Elastix transformation results that get parsed into the rigid transformation
            """
        bgcolor = str(self.sqlController.scan_run.bgcolor) # elastix likes strings
        elastixImageFilter = sitk.ElastixImageFilter()
        fixed_file = os.path.join(self.input, f"{fixed_index}.tif")
        fixed = sitk.ReadImage(fixed_file, self.pixelType)

        moving_file = os.path.join(self.input, f"{moving_index}.tif")
        moving = sitk.ReadImage(moving_file, self.pixelType)
        elastixImageFilter.SetFixedImage(fixed)
        elastixImageFilter.SetMovingImage(moving)

        translationMap = elastixImageFilter.GetDefaultParameterMap("translation")
        rigid_params = create_rigid_parameters(elastixImageFilter, defaultPixelValue=bgcolor)
        elastixImageFilter.SetParameterMap(translationMap)
        elastixImageFilter.AddParameterMap(rigid_params)
        fixed_point_file = os.path.join(self.registration_output, f'{fixed_index}_points.txt')
        moving_point_file = os.path.join(self.registration_output, f'{moving_index}_points.txt')

        if os.path.exists(fixed_point_file) and os.path.exists(moving_point_file):
            if self.debug:
                print(f'Found point files for {fixed_point_file} and {moving_point_file}')
            with open(fixed_point_file, 'r') as fp:
                fixed_count = len(fp.readlines())
            with open(moving_point_file, 'r') as fp:
                moving_count = len(fp.readlines())
            assert fixed_count == moving_count, \
                    f'Error, the number of fixed points in {fixed_point_file} do not match {moving_point_file}'

            elastixImageFilter.SetParameter("Registration", ["MultiMetricMultiResolutionRegistration"])
            elastixImageFilter.SetParameter("Metric",  ["AdvancedNormalizedCorrelation", "CorrespondingPointsEuclideanDistanceMetric"])
            elastixImageFilter.SetParameter("Metric0Weight", ["0.25"]) # the weight of 1st metric for each resolution
            elastixImageFilter.SetParameter("Metric1Weight",  ["0.75"]) # the weight of 2nd metric
            elastixImageFilter.SetFixedPointSetFileName(fixed_point_file)
            elastixImageFilter.SetMovingPointSetFileName(moving_point_file)
        else:
            if self.debug:
                print(f'No point files for {fixed_point_file} and {moving_point_file}')

        elastixImageFilter.LogToConsoleOff()
        if self.debug:
            elastixImageFilter.PrintParameterMap()
        elastixImageFilter.Execute()

        translations = elastixImageFilter.GetTransformParameterMap()[0]["TransformParameters"]
        rigid = elastixImageFilter.GetTransformParameterMap()[1]["TransformParameters"]

        x1, y1 = translations
        R, x2, y2 = rigid
        x = float(x1) + float(x2)
        y = float(y1) + float(y2)
        return float(R), float(x), float(y)

    def align_elastix_with_no_points(self, fixed_index, moving_index):
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

    def create_dir2dir_transformations(self):
        """Calculate and store the rigid transformation using elastix.  
        Align CH3 from CH1
        TODO this needs to be fixed to account for the channel variable name
        """
        MOVING_DIR = os.path.join(self.fileLocationManager.prep, 'CH3', 'thumbnail_cropped')
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
                rotation, xshift, yshift = self.align_elastix_with_points(fixed, moving)
                end_time = timer()
                total_elapsed_time = round((end_time - start_time),2)
                print(f"Moving index={moving_index} took {total_elapsed_time} seconds")

                print(f" took {total_elapsed_time} seconds")
                self.sqlController.add_elastix_row(self.animal, moving_index, rotation, xshift, yshift)

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
        INPUT = os.path.join(self.fileLocationManager.prep, 'CH3', 'full_cropped')
        OUTPUT = self.fileLocationManager.get_full_aligned(channel=channel)
        os.makedirs(OUTPUT, exist_ok=True)
        files = sorted(os.listdir(INPUT))
        center = self.get_rotation_center(channel=channel)
        file_keys = []

        for file in files:
            moving_index = str(file).replace(".tif","")
            rotation, xshift, yshift = self.load_elastix_transformation(self.animal, moving_index)

            T = parameters_to_rigid_transform(rotation, xshift, yshift, center)
            Ts = create_scaled_transform(T)
            infile = os.path.join(INPUT, file)
            outfile = os.path.join(OUTPUT, file)
            if os.path.exists(outfile):
                continue

            file_key = [infile, outfile, Ts]
            file_keys.append(file_key)

        workers = self.get_nworkers()
        self.run_commands_concurrently(align_image_to_affine, file_keys, workers)

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

        :return list: list of x and y for rotation center that set as the midpoint of the section that is in the middle of the stack
        """

        INPUT = self.fileLocationManager.get_thumbnail_cropped(self.channel)
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
        """After the elastix job is done, this fetches the rotation, xshift and yshift from the DB
        
        :param animal: the animal
        :return: a dictionary of key=filename, value = coordinates
        """

        INPUT = self.fileLocationManager.get_thumbnail_cropped(self.channel)
        files = sorted(os.listdir(INPUT))
        midpoint = len(files) // 2
        transformation_to_previous_sec = {}
        center = self.get_rotation_center()

        for i in range(1, len(files)):
            rotation, xshift, yshift = self.load_elastix_transformation(self.animal, i)
            T = parameters_to_rigid_transform(rotation, xshift, yshift, center)
            transformation_to_previous_sec[i] = T

        transformations = {}

        if self.debug:
            print(f'elastix_manager::get_transformations #files={len(files)} in {INPUT}')
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

    def align_full_size_image(self, transforms):
        """align the full resolution tif images with the transformations provided.
           All the sections are aligned to the middle sections, the transformation
           of a given section to the middle section is the composite of the transformation
           from the given section through all the intermediate sections to the middle sections.

        :param transforms: (dict): dictionary of transformations that are index by the id of moving sections
        """
        if not self.downsample:
            transforms = create_downsampled_transforms(transforms, downsample=False, scaling_factor=self.scaling_factor)
            INPUT, OUTPUT = self.fileLocationManager.get_alignment_directories(channel=self.channel, resolution='full')
            self.logevent(f"INPUT FOLDER: {INPUT}")
            starting_files = os.listdir(INPUT)
            self.logevent(f"FILE COUNT: {len(starting_files)} with {len(transforms)} transforms")
            self.logevent(f"OUTPUT FOLDER: {OUTPUT}")
            self.align_images(INPUT, OUTPUT, transforms)

    def align_downsampled_images(self, transforms):
        """align the downsample tiff images

        :param transforms: (dict) dictionary of transformations indexed by id of moving sections
        """

        if self.downsample:
            INPUT, OUTPUT = self.fileLocationManager.get_alignment_directories(channel=self.channel, resolution='thumbnail')
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
        INPUT = fileLocationManager.get_thumbnail_aligned(channel=self.channel)
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
