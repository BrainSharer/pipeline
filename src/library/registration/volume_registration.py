"""
Important notes,
If your fixed image has a smaller field of view than your moving image, 
your moving image will be cropped. (This is what happens when the brain stem
gets cropped out. However, when the fixed image is bigger than the moving image, we get the following error:
Too many samples map outside moving image buffer. The moving image needs to be properly initialized.

In other words, only the part your moving image 
that overlap with the fixed image is included in your result image.
To warp the whole image, you can edit the size of the domain in the 
transform parameter map to match your moving image, and pass your moving image and 
the transform parameter map to sitk.Transformix().
10um allen is 1320x800
25um allen is 528x320
aligned volume @ 32 is 2047x1109 - unreg size matches allen10um
aligned volume @ 64 is 1024x555 - unreg size matches allen25um
aligned volume @ 128 is 512x278
aligned volume @50 is 1310x710
full aligned is 65500x35500
Need to scale a moving image as close as possible to the fixed image
COM info:
allen SC: (368, 62, 227)
pred  SC: 369, 64, 219
TODO, transform polygons in DB using the transformation below
"""

from collections import defaultdict
import os
import shutil
import sys
import numpy as np
from skimage import io
from scipy.ndimage import zoom
from skimage.filters import gaussian        
#from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from tqdm import tqdm
import SimpleITK as sitk
import itk

from taskqueue import LocalTaskQueue
import igneous.task_creation as tc
import pandas as pd
import cv2
import json

from library.atlas.atlas_utilities import adjust_volume, average_images
from library.controller.sql_controller import SqlController
from library.controller.annotation_session_controller import AnnotationSessionController
from library.image_manipulation.neuroglancer_manager import NumpyToNeuroglancer
from library.image_manipulation.filelocation_manager import FileLocationManager
from library.utilities.utilities_mask import normalize16
from library.utilities.utilities_process import SCALING_FACTOR, get_scratch_dir, read_image, write_image
from library.atlas.brain_structure_manager import BrainStructureManager
from library.atlas.brain_merger import BrainMerger
from library.image_manipulation.image_manager import ImageManager
from library.utilities.utilities_registration import create_affine_parameters

# constants
MOVING_CROP = 50
M_UM_SCALE = 1000000


class VolumeRegistration:
    """This class takes a downsampled image stack and registers it to the Allen volume
    Data resides in /net/birdstore/Active_Atlas_Data/data_root/brains_info/registration
    Volumes have the following naming convention:
        1. image stack = {animal}_{um}um_{orientation}.tif -> MD589_25um_sagittal.tif
        2. registered volume = {moving}_{fixed}_{um}um_{orientation}.tif -> MD585_MD589_25um_sagittal.tif
        3. point file = {animal}_{um}um_{orientation}.pts -> MD589_25um_sagittal.pts
        
    """

    def __init__(self, moving, channel=1, um=25, fixed=None, orientation='sagittal', bspline=False, debug=False):
        self.registration_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
        self.moving_path = os.path.join(self.registration_path, moving)
        os.makedirs(self.moving_path, exist_ok=True)
        self.atlas_path = '/net/birdstore/Active_Atlas_Data/data_root/atlas_data/Atlas' 
        self.allen_path = os.path.join(self.registration_path, 'Allen')
        self.tmp_dir = get_scratch_dir()
        self.moving = moving
        self.animal = moving
        self.debug = debug
        self.fixed = fixed
        self.um = um
        self.mask_color = 254
        self.channel = f'C{channel}'
        self.orientation = orientation
        self.bspline = bspline
        self.output_dir = f'{moving}_{fixed}_{um}um_{orientation}'
        self.scaling_factor = 32 # This is the downsampling factor used to create the aligned volume at 10um
        self.fileLocationManager = FileLocationManager(self.moving)
        self.sqlController = SqlController(self.animal)
        self.thumbnail_aligned = os.path.join(self.fileLocationManager.prep, self.channel, 'thumbnail_aligned')
        self.moving_volume_path = os.path.join(self.moving_path, f'{self.moving}_{um}um_{orientation}.tif' )
        self.registered_volume = os.path.join(self.moving_path, f'{self.moving}_{self.fixed}_{um}um_{orientation}.tif' )
        self.changes_path = os.path.join(self.moving_path, f'{self.moving}_{um}um_{orientation}_changes.json' )
        
        self.registration_output = os.path.join(self.moving_path, self.output_dir)
        self.elastix_output = os.path.join(self.registration_output, 'elastix_output')
        self.reverse_elastix_output = os.path.join(self.registration_output, 'reverse_elastix_output')
        
        self.registered_point_file = os.path.join(self.registration_output, 'outputpoints.txt')
        self.unregistered_point_file = os.path.join(self.moving_path, f'{self.animal}_{um}um_{orientation}_unregistered.pts')
        self.number_of_sampling_attempts = "10"
        self.number_of_resolutions = "4"
        if self.debug:
            iterations = "50"
            self.rigidIterations = iterations
            self.affineIterations = iterations
            self.bsplineIterations = iterations
        else:
            self.rigidIterations = "1000"
            self.affineIterations = "2500"
            self.bsplineIterations = "15000"


        if fixed is not None:
            self.fixed = fixed
            self.fixed_path = os.path.join(self.registration_path, fixed)
            self.fixed_volume_path = os.path.join(self.fixed_path, f'{self.fixed}_{um}um_{orientation}.tif' )
            self.neuroglancer_data_path = os.path.join(self.fileLocationManager.neuroglancer_data, f'{self.channel}_{self.fixed}_{um}um')
            os.makedirs(self.fixed_path, exist_ok=True)
        else:
            self.neuroglancer_data_path = os.path.join(self.fileLocationManager.neuroglancer_data, f'{self.channel}_{um}um')



        self.report_status()

    def report_status(self):
        print("Running volume registration with the following settings:")
        print("\tprep_id:".ljust(20), f"{self.animal}".ljust(20))
        print("\tum:".ljust(20), f"{str(self.um)}".ljust(20))
        print("\torientation:".ljust(20), f"{str(self.orientation)}".ljust(20))
        print("\tdebug:".ljust(20), f"{str(self.debug)}".ljust(20))
        print("\tresolutions:".ljust(20), f"{str(self.number_of_resolutions)}".ljust(20))
        print("\trigid iterations:".ljust(20), f"{str(self.affineIterations)}".ljust(20))
        print()



    def setup_transformix(self, outputpath):
        """Method used to transform volumes and points
        """

        number_of_transforms = 1
        
        os.makedirs(self.registration_output, exist_ok=True)

        transform_parameter0_path = os.path.join(outputpath, 'TransformParameters.0.txt')
        transform_parameter1_path = os.path.join(outputpath, 'TransformParameters.1.txt')

        if not os.path.exists(transform_parameter0_path):
            print(f'{transform_parameter0_path} does not exist, exiting.')
            sys.exit()

        if os.path.exists(transform_parameter1_path):
            print(f'{transform_parameter1_path} exists, using two transformation.')
            number_of_transforms = 2
            

        transformixImageFilter = sitk.TransformixImageFilter()
        parameterMap0 = sitk.ReadParameterFile(transform_parameter0_path)
        transformixImageFilter.SetTransformParameterMap(parameterMap0)
        if number_of_transforms == 2:
            # Read the second transform parameter map
            parameterMap1 = sitk.ReadParameterFile(transform_parameter1_path)
            transformixImageFilter.AddTransformParameterMap(parameterMap1)
        transformixImageFilter.LogToFileOn()
        transformixImageFilter.LogToConsoleOff()
        transformixImageFilter.SetOutputDirectory(self.registration_output)
        movingImage = sitk.ReadImage(self.moving_volume_path)
        transformixImageFilter.SetMovingImage(movingImage)
        return transformixImageFilter

    def transformix_volume(self):
        """Helper method when you want to rerun the same transform on another volume
        """
        
        transformixImageFilter = self.setup_transformix(self.elastix_output)
        transformixImageFilter.Execute()
        transformed = transformixImageFilter.GetResultImage()
        sitk.WriteImage(transformed, os.path.join(self.registration_output, self.registered_volume))

    def transformix_com(self):
        """Helper method when you want to rerun the transform on a set of points.
        Get the pickle file and transform it. It is in full resolution pixel size.
        The points in the pickle file need to be translated from full res pixel to
        the current resolution of the downsampled volume.
        Points are inserted in the DB in micrometers from the full resolution images

        
        The points.pts file takes THIS format:
        point
        3
        102.8 -33.4 57.0
        178.1 -10.9 14.5
        180.4 -18.1 78.9
        """
        d = pd.read_pickle(self.unregistered_pickle_file)
        point_dict = dict(sorted(d.items()))
        with open(self.unregistered_point_file, 'w') as f:
            f.write('point\n')
            f.write(f'{len(point_dict)}\n')
            for _, points in point_dict.items():
                x = points[0]/self.scaling_factor
                y = points[1]/self.scaling_factor
                z = points[2] # the z is not scaled
                #print(structure, points, x,y,z)
                f.write(f'{x} {y} {z}')
                f.write('\n')
        
        transformixImageFilter = self.setup_transformix(self.reverse_elastix_output)
        transformixImageFilter.SetFixedPointSetFileName(self.unregistered_point_file)
        transformixImageFilter.Execute()

    
    def create_unregistered_pointfile(self):
        origin_dir = os.path.join(self.atlas_path, 'origin')
        origin_files = sorted(os.listdir(origin_dir))
        pointfile = os.path.join(self.registration_path, 'Atlas_25um_sagittal_unregistered.pts')
        with open(pointfile, 'w') as f:
            f.write('point\n')
            f.write(f'{len(origin_files)}\n')
            for origin_file in origin_files:
                x,y,z = np.loadtxt(os.path.join(origin_dir, origin_file))
                f.write(f'{x} {y} {z}')
                f.write('\n')
        return origin_files

    def transformix_points(self):
        """Helper method when you want to rerun the transform on a set of points.
        Get the pickle file and transform it. It is in full resolution pixel size.
        The points in the pickle file need to be translated from full res pixel to
        the current resolution of the downsampled volume.
        Points are inserted in the DB in micrometers from the full resolution images
        The TransformParameter.1.txt file is the one that contains the transformation
        from the affine registration
        
        The points.pts file takes THIS format:
        point
        3
        102.8 -33.4 57.0
        178.1 -10.9 14.5
        180.4 -18.1 78.9
        """
        if not os.path.exists(self.unregistered_point_file):
            print(f'{self.unregistered_point_file} does not exist, exiting.')
            sys.exit()

        reverse_transformation_pfile = os.path.join(self.reverse_elastix_output, 'TransformParameters.0.txt')
        if not os.path.exists(reverse_transformation_pfile):
            print(f'{reverse_transformation_pfile} does not exist, exiting.')
            sys.exit()
        
        transformixImageFilter = self.setup_transformix(self.reverse_elastix_output)
        transformixImageFilter.SetFixedPointSetFileName(self.unregistered_point_file)
        transformixImageFilter.Execute()

    def transformix_origins(self):
        registered_origin_path = os.path.join(self.atlas_path, 'registered_origin')
        os.makedirs(registered_origin_path, exist_ok=True)
        origin_files = self.create_unregistered_pointfile()
        structures = sorted([str(origin).replace('.txt','') for origin in origin_files])
        self.transformix_points()
        point_or_index = 'OutputPoint'

        with open(self.registered_point_file, "r") as f:                
            lines=f.readlines()
            f.close()

        point_or_index = 'OutputPoint'
        for i in range(len(lines)):        
            lx=lines[i].split()[lines[i].split().index(point_or_index)+3:lines[i].split().index(point_or_index)+6] #x,y,z
            lf = [float(f) for f in lx]
            x = lf[0]
            y = lf[1]
            z = lf[2]
            structure = structures[i]
            print(i, structure,  int(x), int(y), int(z))
            origin_filepath = os.path.join(registered_origin_path, f'{structure}.txt')
            np.savetxt(origin_filepath, (x,y,z))




    def insert_points(self):
        """This method will take the pickle file of COMs and insert them.
        The COMs in the pickle files are in pixel coordinates.
        For typical COMs, the full scaled xy version gets multiplied by 0.325 then inserted
        Upon retrieval, xy gets: divided by 0.325. Here we scale by our downsampling factor when we created the volume,
        then multiple by the scan run resolution which is hard coded below.
        """

        if not os.path.exists(self.registered_point_file):
            print(f'{self.registered_point_file} does not exist, exiting.')
            sys.exit()

        com_annotator_id = 2
        structureController = self.sql
        coms = structureController.get_coms(self.moving, annotator_id=com_annotator_id)

        point_or_index = 'OutputPoint'
        source='COMPUTER'
        sessionController = AnnotationSessionController(self.moving)

        with open(self.registered_point_file, "r") as f:                
            lines=f.readlines()
            f.close()


        if len(lines) != len(coms):
            print(f'Length of {self.registered_point_file}={len(lines)} != length of coms={len(coms)}')
            return

        point_or_index = 'OutputPoint'
        for i in range(len(lines)):        
            lx=lines[i].split()[lines[i].split().index(point_or_index)+3:lines[i].split().index(point_or_index)+6] #x,y,z
            lf = [float(f) for f in lx]
            x = lf[0] * self.um
            y = lf[1] * self.um
            z = lf[2] * self.um
            com = coms[i]
            structure = com.session.brain_region.abbreviation
           
            brain_region = sessionController.get_brain_region(structure)
            if brain_region is not None:
                annotation_session = sessionController.get_annotation_session(self.moving, brain_region.id, com_annotator_id)
                entry = {'source': source, 'FK_session_id': annotation_session.id, 'x': x, 'y':y, 'z': z}
                sessionController.upsert_structure_com(entry)
            else:
                print(f'No brain region found for {structure}')

            if self.debug and brain_region is not None:
                #lf = [round(l) for l in lf]
                print(i, annotation_session.id, self.moving, brain_region.id, source, structure,  int(x/25), int(y/25), int(z/25))


    def transformix_polygons(self):
        """Marissa is ID=38, TG_L =80 and TG_R=81
        """
        def linear_stretch(old_min, old_max, x, stretch):
            new_max = old_max * stretch
            new_min = old_min
            return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

        # check for necessary files
        if not os.path.exists(self.registered_volume):
            print(f'{self.registered_volume} does not exist, exiting.')
            sys.exit()
        if not os.path.exists(self.fixed_volume_path):
            print(f'{self.fixed_volume_path} does not exist, exiting.')
            sys.exit()
        transformix_pointset_file = os.path.join(self.registration_output, "transformix_input_points.txt")
        if os.path.exists(transformix_pointset_file):
            print(f'{transformix_pointset_file} exists, removing')
            os.remove(transformix_pointset_file)
        if not os.path.exists(self.reverse_elastix_output):
            print(f'{self.reverse_elastix_output} does not exist, exiting.')
            sys.exit()
        result_path = os.path.join(self.registration_output, f'Allen_{self.um}um_annotated.tif')
        if not os.path.exists(self.changes_path):
            print(f'{self.changes_path} does not exist, exiting.')
            sys.exit()
        if not os.path.exists(self.registered_volume):
            print(f'{self.registered_volume} does not exist, exiting.')
            sys.exit()
        
        sqlController = SqlController(self.moving) 
        scale_xy = sqlController.scan_run.resolution
        z_scale = sqlController.scan_run.zresolution
        #annotation_left = sqlController.get_annotation_session(self.moving, label_id=80, annotator_id=38)
        annotation_right = sqlController.get_annotation_session(self.moving, label_id=81, annotator_id=38)
        #annotation_left = annotation_left.annotation
        annotation_right = annotation_right.annotation
        #left_childJsons = annotation_left['childJsons']
        right_childJsons = annotation_right['childJsons']
        rows_left = []
        rows_right = []
        polygons = defaultdict(list)
        #for child in left_childJsons:
        #    for i, row in enumerate(child['childJsons']):
        #        rows_left.append(row['pointA'])
        for child in right_childJsons:
            for i, row in enumerate(child['childJsons']):
                rows_right.append(row['pointA'])
        input_points = itk.PointSet[itk.F, 3].New()
        #df_L = pd.DataFrame(rows_left, columns=['x1','y1','z1'])
        df_R = pd.DataFrame(rows_right, columns=['x1','y1','z1'])
        #frames = [df_L, df_R]
        frames = [df_R]
        df = pd.concat(frames)
        #len_L = df_L.shape[0]
        len_R = df_R.shape[0]
        len_total = df.shape[0]
        #assert len_L + len_R == len_total, "Lengths of dataframes do not add up."
        
        with open(self.changes_path, 'r') as file:
            change = json.load(file)

        df['xn'] = df['x1'] * M_UM_SCALE / (scale_xy * SCALING_FACTOR)
        df['yn'] = df['y1'] * M_UM_SCALE / (scale_xy * SCALING_FACTOR)
        df['zn'] = df['z1'] * M_UM_SCALE / z_scale
        
        df['x'] = df['x1'] * change['change_x'] * M_UM_SCALE / (scale_xy * SCALING_FACTOR)
        df['y'] = df['y1'] * change['change_y'] * M_UM_SCALE / (scale_xy * SCALING_FACTOR)
        df['z2'] = df['z1'] * M_UM_SCALE / z_scale
        df['z'] = linear_stretch(df['z2'].min(), df['z2'].max(), df['z2'], change['change_z'])
        for idx, (_, row) in enumerate(df.iterrows()):
            x = row['x']
            y = row['y']
            z = int(round(row['z']))
            section = int(round(row['zn']))
            point = [x,y, z]
            input_points.GetPoints().InsertElement(idx, point)
            polygons[section].append((x,y))

        if self.debug:
            output_dir = os.path.join(self.fileLocationManager.prep, self.channel, 'thumbnail_debug')
            if os.path.exists(output_dir):
                print(f'{output_dir} exists, removing')
                shutil.rmtree(output_dir)

            os.makedirs(output_dir, exist_ok=True)
            print(df.describe())
            for section, points in tqdm(polygons.items()):
                file = str(section).zfill(3) + ".tif"
                inpath = os.path.join(self.thumbnail_aligned, file)
                if not os.path.exists(inpath):
                    print(f'{inpath} does not exist')
                    continue
                img = cv2.imread(inpath, cv2.IMREAD_GRAYSCALE)
                points = np.array(points)
                points = points.astype(np.int32)
                cv2.fillPoly(img, pts = [points], color = 255)
                outpath = os.path.join(output_dir, file)
                cv2.imwrite(outpath, img)
            del polygons
            return

        del df
        # Write points to be transformed
        with open(transformix_pointset_file, "w") as f:
            f.write("point\n")
            f.write(f"{input_points.GetNumberOfPoints()}\n")
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
            for idx in range(input_points.GetNumberOfPoints()):
                point = input_points.GetPoint(idx)
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
                
        transformixImageFilter = self.setup_transformix(self.reverse_elastix_output)
        transformixImageFilter.SetFixedPointSetFileName(transformix_pointset_file)
        transformixImageFilter.Execute()
                    
        polygons = defaultdict(list)
        with open(self.registered_point_file, "r") as f:                
            lines=f.readlines()
            f.close()

        point_or_index = 'OutputPoint'
        points = []
        for i in tqdm(range(len(lines))):        
            lx=lines[i].split()[lines[i].split().index(point_or_index)+3:lines[i].split().index(point_or_index)+6] #x,y,z
            lf = [float(f) for f in lx]
            x = lf[0]
            y = lf[1]
            z = lf[2]
            section = int(np.round(z))
            polygons[section].append((x,y))
            points.append((x,y,section))
        resultImage = io.imread(self.fixed_volume_path)
        registered_volume = io.imread(self.registered_volume)
        
        for section, points in tqdm(polygons.items()):
            if self.debug:
                for point in points:
                    x = int(point[0])
                    y = int(point[1])
                    cv2.circle(resultImage[section,:,:], (x,y), 12, 254, thickness=3)
            else:
                points = np.array(points, dtype=np.int32)
                try:
                    cv2.fillPoly(resultImage[section,:,:], pts = [points], color = self.mask_color)
                    cv2.fillPoly(registered_volume[section,:,:], pts = [points], color = self.mask_color)
                except IndexError as e:
                    print(f'Section: {section} error: {e}')

        io.imsave(result_path, resultImage)
        print(f'Saved a 3D volume {result_path} with shape={resultImage.shape} and dtype={resultImage.dtype}')

        io.imsave(self.registered_volume, registered_volume)
        print(f'Saved a 3D volume {self.registered_volume} with shape={registered_volume.shape} and dtype={registered_volume.dtype}')


    def transformix_coms(self):
        com_annotator_id = 2
        #controller = StructureCOMController(self.moving)
        controller = None
        coms = controller.get_COM(self.moving, annotator_id=com_annotator_id)

        
        transformix_pointset_file = os.path.join(self.registration_output,"transformix_input_points.txt")        
        with open(transformix_pointset_file, "w") as f:
            f.write("point\n")
            f.write(f"{len(coms)}\n")

            for idx, (structure, (x,y,z)) in enumerate(coms.items()):
                x /= self.um
                y /= self.um
                z /= self.um
                print(idx, structure, x,y,z)
                f.write(f"{x} {y} {z}\n")
                
        transformixImageFilter = self.setup_transformix(self.reverse_elastix_output)
        transformixImageFilter.SetFixedPointSetFileName(transformix_pointset_file)
        transformixImageFilter.ExecuteInverse()


    def fill_contours(self):
        sqlController = SqlController(self.moving)
        # vars
        self.input = os.path.join(self.movingLocationManager.prep, 'C1', 'thumbnail_aligned')
        self.output = os.path.join(self.movingLocationManager.prep, 'C1', 'thumbnail_merged')
        os.makedirs(self.output, exist_ok=True)
        #polygon = PolygonSequenceController(animal=self.moving)        
        polygon = None
        scale_xy = sqlController.scan_run.resolution
        z_scale = sqlController.scan_run.zresolution
        polygons = defaultdict(list)
        color = 0 # set it below the threshold set in mask class
        """
        df_L = polygon.get_volume(self.moving, 3, 12)
        df_R = polygon.get_volume(self.moving, 3, 13)
        frames = [df_L, df_R]
        df = pd.concat(frames)
        len_L = df_L.shape[0]
        len_R = df_R.shape[0]
        len_total = df.shape[0]
        assert len_L + len_R == len_total, "Lengths of dataframes do not add up."
        """
        df = polygon.get_volume(self.moving, 3, 33)

        for _, row in df.iterrows():
            x = row['coordinate'][0]
            y = row['coordinate'][1]
            z = row['coordinate'][2]
            xy = (x/scale_xy/self.scaling_factor, y/scale_xy/self.scaling_factor)
            section = int(np.round(z/z_scale))
            polygons[section].append(xy)
                    
        for section, points in tqdm(polygons.items()):
            file = str(section).zfill(3) + ".tif"
            inpath = os.path.join(self.input, file)
            if not os.path.exists(inpath):
                print(f'{inpath} does not exist')
                continue
            img = cv2.imread(inpath, cv2.IMREAD_GRAYSCALE)
            points = np.array(points)
            points = points.astype(np.int32)
            cv2.fillPoly(img, pts = [points], color = color)
            outpath = os.path.join(self.output, file)
            cv2.imwrite(outpath, img)

        files = sorted(os.listdir(self.input))
        for file in tqdm(files):
            inpath = os.path.join(self.input, file)
            img = cv2.imread(inpath, cv2.IMREAD_GRAYSCALE)
            outpath = os.path.join(self.output, file)
            if not os.path.exists(outpath):
                cv2.imwrite(outpath, img)


    def create_volume(self):
        """
        Create a 3D volume
            image stack = 10.4um x 10.4um x 20um
            atlas stack = 10um x 10um x 10um
            MD589 z is 20um * 447 sections so 894um
            Allen z @ 10um  = 1140 sections = 11400um
            Allen @50 x,y,z = 264x160x224, 50um=[13200  8000 11400]

        """
        image_manager = ImageManager(self.thumbnail_aligned)
        xy_resolution = self.sqlController.scan_run.resolution * SCALING_FACTOR /  self.um
        z_resolution = self.sqlController.scan_run.zresolution / self.um

        change_z = z_resolution
        change_y = xy_resolution
        change_x = xy_resolution
        change = (change_z, change_y, change_x) 
        changes = {'change_z': change_z, 'change_y': change_y, 'change_x': change_x}
        print(f'change_z={change_z} change_y={change_y} change_x={change_x}')

        
        if os.path.exists(self.moving_volume_path):
            print(f'{self.moving_volume_path} exists, exiting')
            return

        image_stack = np.zeros(image_manager.volume_size)
        file_list = []
        for ffile in tqdm(image_manager.files, desc='Creating volume'):
            fpath = os.path.join(self.thumbnail_aligned, ffile)
            farr = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            file_list.append(farr)

        if self.animal == 'DK73':
            for ffile in tqdm(sorted(image_manager.files, reverse=True), desc='Creating volume'):
                fpath = os.path.join(self.thumbnail_aligned, ffile)
                farr = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                file_list.append(farr)
            
        image_stack = np.stack(file_list, axis = 0)
        
        
        with open(self.changes_path, 'w') as f:
            json.dump(changes, f)            
        
        zoomed = zoom(image_stack, change)
        #zoomed = image_stack.copy()
        write_image(self.moving_volume_path, zoomed.astype(image_manager.dtype))
        print(f'Saved a 3D volume {self.moving_volume_path} with shape={image_stack.shape} and dtype={image_stack.dtype}')

    def pad_volume(self):
        pad = 500
        volume = io.imread(self.fixed_volume_path)
        dtype = volume.dtype
        print(f'volume shape={volume.shape} dtype={volume.dtype}')
        volume = np.concatenate((volume, np.zeros((volume.shape[0], pad, volume.shape[2])) ), axis=1)
        print(f'volume shape={volume.shape} dtype={volume.dtype}')
        #volume = np.concatenate((volume, np.zeros((volume.shape[0], volume.shape[1], pad)) ), axis=2)
        #print(f'volume shape={volume.shape} dtype={volume.dtype}')
        outpath = os.path.join(self.registration_path, f'{self.fixed}_{self.um}um_{self.orientation}_padded.tif')
        write_image(outpath, volume.astype(dtype))

    def create_precomputed(self):
        chunk = 64
        chunks = (chunk, chunk, chunk)
        if self.fixed is None:
            volumepath = os.path.join(self.registration_output, self.moving_volume_path)
        else:
            volumepath = self.registration_output + '.tif'
        if not os.path.exists(volumepath):
            print(f'{volumepath} does not exist, exiting.')
            sys.exit()
        else:
            print(f'Creating precomputed from {volumepath}')


        PRECOMPUTED = self.neuroglancer_data_path
        if os.path.exists(PRECOMPUTED):
            print(f'{PRECOMPUTED} exists, removing')
            shutil.rmtree(PRECOMPUTED)

        scale = self.um * 1000
        scales = (scale, scale, scale)
        os.makedirs(PRECOMPUTED, exist_ok=True)
        volume = read_image(volumepath)
        volume = np.swapaxes(volume, 0, 2)
        num_channels = 1
        volume_size = volume.shape
        
        print(f'volume shape={volume.shape} dtype={volume.dtype} creating at {PRECOMPUTED}')
        volume = normalize16(volume)

        ng = NumpyToNeuroglancer(
            self.moving,
            None,
            scales,
            "image",
            volume.dtype,
            num_channels=num_channels,
            chunk_size=chunks,
        )

        ng.init_precomputed(PRECOMPUTED, volume_size)
        ng.precomputed_vol[:, :, :] = volume
        ng.precomputed_vol.cache.flush()
        tq = LocalTaskQueue(parallel=4)
        cloudpath = f"file://{PRECOMPUTED}"
        tasks = tc.create_downsampling_tasks(cloudpath, num_mips=2)
        tq.insert(tasks)
        tq.execute()


    def register_volume(self):
        """This will perform the elastix registration of the volume to the atlas.
        It first does an affine registration, then a bspline registration.
        """
        if os.path.exists(self.elastix_output):
            print(f'Removing {self.elastix_output}')
            shutil.rmtree(self.elastix_output)

        if self.fixed_path is None or self.moving_path is None:
            print('Fixed or moving path is None, exiting')
            sys.exit()

        os.makedirs(self.elastix_output, exist_ok=True)
        fixed_path = self.fixed_path
        moving_path = self.moving_path
        fixed_basename = f'{self.fixed}_{self.um}um_{self.orientation}'
        moving_basename = f'{self.moving}_{self.um}um_{self.orientation}'
        elastixImageFilter = self.setup_registration(fixed_path, moving_path, fixed_basename, moving_basename)
        elastixImageFilter.SetOutputDirectory(self.elastix_output)
        elastixImageFilter.PrintParameterMap()
        resultImage = elastixImageFilter.Execute()
                
        resultImage = sitk.Cast(sitk.RescaleIntensity(resultImage), sitk.sitkUInt16)

        sitk.WriteImage(resultImage, self.registered_volume)
        print(f'Saved img to {self.registered_volume}')

    def reverse_register_volume(self):
        """This method also uses an affine and a bspline registration process, but it does 
        it in reverse. The fixed and moving images get switched so we can get the transformation
        for the points to get registered to the atlas. 
        """
       
        os.makedirs(self.reverse_elastix_output, exist_ok=True)
        # switch fixed and moving
        fixed_path = self.moving_path
        moving_path = self.fixed_path
        fixed_basename = f'{self.moving}_{self.um}um_{self.orientation}'
        moving_basename = f'{self.fixed}_{self.um}um_{self.orientation}'
        elastixImageFilter = self.setup_registration(fixed_path, moving_path, fixed_basename, moving_basename)
        elastixImageFilter.SetOutputDirectory(self.reverse_elastix_output)
        elastixImageFilter.Execute()
        print('Done performing inverse')

    def setup_registration(self, fixed_path, moving_path, fixed_basename, moving_basename):
        
        fixed_path = os.path.join(fixed_path, f'{fixed_basename}.tif' )
        moving_path = os.path.join(moving_path, f'{moving_basename}.tif') 
        thumbnail_path = os.path.join(self.fileLocationManager.prep, self.channel, 'thumbnail_aligned')

        if not os.path.exists(fixed_path):
            print(f'Fixed {fixed_path} does not exist')
            sys.exit()
        if not os.path.exists(moving_path):
            print(f'Moving {moving_path} does not exist')
            sys.exit()
        # set point paths
        fixed_point_path = os.path.join(self.registration_path, self.fixed, f'{fixed_basename}.pts')
        moving_point_path = os.path.join(self.registration_path, self.moving, f'{moving_basename}.pts')
        if self.debug:
            print(f'moving volume path={moving_path}')
            print(f'fixed volume path={fixed_path}')
            print(f'moving point path={moving_point_path}')
            print(f'fixed point path={fixed_point_path}')
        
        fixedImage = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
        movingImage = sitk.ReadImage(moving_path, sitk.sitkFloat32)
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixedImage)
        elastixImageFilter.SetMovingImage(movingImage)

        genericMap = create_affine_parameters(elastixImageFilter=elastixImageFilter)

        if self.bspline:
            bsplineParameterMap = sitk.GetDefaultParameterMap('bspline')
            bsplineParameterMap["MaximumNumberOfIterations"] = [self.bsplineIterations] # 250 works ok
            if not self.debug:
                bsplineParameterMap["WriteResultImage"] = ["false"]
                bsplineParameterMap["UseDirectionCosines"] = ["true"]
                bsplineParameterMap["FinalGridSpacingInVoxels"] = [f"{self.um}"]
                bsplineParameterMap["MaximumNumberOfSamplingAttempts"] = [self.number_of_sampling_attempts]
                bsplineParameterMap["NumberOfResolutions"]= ["6"]
                bsplineParameterMap["GridSpacingSchedule"] = ["6.219", "4.1", "2.8", "1.9", "1.4", "1.0"]
                bsplineParameterMap["Optimizer"] = ["StandardGradientDescent"]
                del bsplineParameterMap["FinalGridSpacingInPhysicalUnits"]

        elastixImageFilter.SetParameterMap(genericMap)
        if os.path.exists(fixed_point_path) and os.path.exists(moving_point_path):
            with open(fixed_point_path, 'r') as fp:
                fixed_count = len(fp.readlines())
            with open(moving_point_path, 'r') as fp:
                moving_count = len(fp.readlines())
            assert fixed_count == moving_count, f'Error, the number of fixed points in {fixed_point_path} do not match {moving_point_path}'

            elastixImageFilter.SetParameter("Registration", ["MultiMetricMultiResolutionRegistration"])
            elastixImageFilter.SetParameter("Metric",  ["AdvancedMattesMutualInformation", "CorrespondingPointsEuclideanDistanceMetric"])
            elastixImageFilter.SetParameter("Metric0Weight", ["0.0"]) # the weight of 1st metric
            elastixImageFilter.SetParameter("Metric1Weight",  ["1.0"]) # the weight of 2nd metric
            elastixImageFilter.SetParameter("MaximumNumberOfIterations", "250")

            elastixImageFilter.SetFixedPointSetFileName(fixed_point_path)
            elastixImageFilter.SetMovingPointSetFileName(moving_point_path)
            if self.debug:
                print(f'moving point path={moving_point_path}')
                print(f'fixed point path={fixed_point_path}')
        else:
            print(f'Fixed point path {fixed_point_path} or \nmoving point path {moving_point_path} do not exist')
            sys.exit()

        if self.bspline:
            elastixImageFilter.AddParameterMap(bsplineParameterMap)
            elastixImageFilter.SetParameter("MaximumNumberOfIterations", "2500")
        elastixImageFilter.SetParameter("ResultImageFormat", "tif")
        elastixImageFilter.SetParameter("NumberOfResolutions", "6") #### Very important, less than 6 gives lousy results.
        elastixImageFilter.SetParameter("ComputeZYX", "true")
        elastixImageFilter.SetParameter("DefaultPixelValue", "0")
        elastixImageFilter.PrintParameterMap
        elastixImageFilter.SetLogToFile(True)
        elastixImageFilter.LogToConsoleOff()

        elastixImageFilter.SetLogFileName('elastix.log')

        return elastixImageFilter


    def crop_volume(self):

        moving_volume = io.imread(self.moving_volume_path)
        moving_volume = moving_volume[:,MOVING_CROP:500, MOVING_CROP:725]
        savepath = os.path.join(self.registration_path, f'Atlas_{self.um}um_{self.orientation}.tif')
        print(f'Saving img to {savepath}')
        io.imsave(savepath, moving_volume)

    def create_brain_coms(self):
        brains = ['MD585', 'MD594', 'MD589', 'AtlasV8']
        base_com_path = '/net/birdstore/Active_Atlas_Data/data_root/atlas_data'
        
        for brain in tqdm(brains, desc='Creating brain coms'):
            brain_point_path = os.path.join(self.registration_path, brain, f'{brain}_{self.um}um_{self.orientation}.pts')
            brain_com_path = os.path.join(base_com_path, brain, 'com')
            comfiles = sorted(os.listdir(brain_com_path))
            with open(brain_point_path, 'w') as f:
                f.write('point\n')
                f.write(f'{len(comfiles)}\n')
                for comfile in comfiles:
                    compath = os.path.join(brain_com_path, comfile)
                    x,y,z = np.loadtxt(compath)
                    f.write(f'{x/self.um} {y/self.um} {z/self.um}')
                    if 'SC' in compath:
                        print(f'{brain=} {x/self.um} {y/self.um} {z/self.um}')
                    f.write('\n')

    def create_average_volume(self):
        moving_brains = ['MD585', 'MD594', 'MD589']
        fixed_brain = 'AtlasV8'

        volumes = {}
        for brain in moving_brains:
            brainpath = os.path.join(self.registration_path, brain, f'{brain}_{self.um}um_{self.orientation}.tif')
            if not os.path.exists(brainpath):
                print(f'{brainpath} does not exist, exiting.')
                continue
            brainimg = read_image(brainpath)
            volumes[brain] = sitk.GetImageFromArray(brainimg.astype(np.float32))

        fixed_path = os.path.join(self.registration_path, fixed_brain, f'{fixed_brain}_{self.um}um_{self.orientation}.tif')
        if not os.path.exists(fixed_path):
            fixed_brain = 'MD589'
            print(f'{fixed_path} does not exist, using {fixed_brain}')
            fixed_path = os.path.join(self.registration_path, fixed_brain, f'{fixed_brain}_{self.um}um_{self.orientation}.tif')
        fixed_img = read_image(fixed_path)
        volumes[fixed_brain] = sitk.GetImageFromArray(fixed_img.astype(np.float32))
        reference_image = volumes[fixed_brain]
        fixed_brain = 'AtlasV8'
        fixed_point_path = os.path.join(self.registration_path, fixed_brain, f'{fixed_brain}_{self.um}um_{self.orientation}.pts')
        fixed_brain = 'MD589'
        affineParameterMap = sitk.GetDefaultParameterMap('affine')
        bsplineParameterMap = sitk.GetDefaultParameterMap('bspline')

        bsplineParameterMap["Optimizer"] = ["StandardGradientDescent"]
        bsplineParameterMap["FinalGridSpacingInVoxels"] = [f"{self.um}"]
        bsplineParameterMap["MaximumNumberOfIterations"] = ["2500"]
        bsplineParameterMap["MaximumNumberOfSamplingAttempts"] = ["10"]
        if self.um > 20:
            bsplineParameterMap["NumberOfResolutions"]= ["6"]
            affineParameterMap["NumberOfResolutions"]= ["6"] # Takes lots of RAM
            bsplineParameterMap["GridSpacingSchedule"] = ["6.219", "4.1", "2.8", "1.9", "1.4", "1.0"]
        else:
            affineParameterMap["NumberOfResolutions"]= ["8"] # Takes lots of RAM
            bsplineParameterMap["NumberOfResolutions"]= ["8"]
            bsplineParameterMap["GridSpacingSchedule"] = ["11.066214285714288", "8.3785", "6.219", "4.1", "2.8", "1.9", "1.4", "1.0"]

        del bsplineParameterMap["FinalGridSpacingInPhysicalUnits"]
        bsplineParameterMap["MaximumNumberOfIterations"] = [self.affineIterations]
        registered_images = []
        savepath = os.path.join(self.registration_path, 'AtlasV8')
        for brain, image in volumes.items():
            if brain == fixed_brain:
                print(f'Skipping {brain} = {fixed_brain}')
                continue
            else:
                print(f'Processing {brain} to {fixed_brain}')
            elastixImageFilter = sitk.ElastixImageFilter()
            elastixImageFilter.SetFixedImage(reference_image)
            elastixImageFilter.SetMovingImage(image)

            moving_point_path = os.path.join(self.registration_path, brain, f'{brain}_{self.um}um_{self.orientation}.pts')
            if os.path.exists(fixed_point_path) and os.path.exists(moving_point_path):
                with open(fixed_point_path, 'r') as fp:
                    fixed_count = len(fp.readlines())
                with open(moving_point_path, 'r') as fp:
                    moving_count = len(fp.readlines())
                assert fixed_count == moving_count, f'Error, the number of fixed points in {fixed_point_path} do not match {moving_point_path}'
                print(f'Transforming points from {os.path.basename(fixed_point_path)} -> {os.path.basename(moving_point_path)}')
                affineParameterMap["Registration"] = ["MultiMetricMultiResolutionRegistration"]
                affineParameterMap["Metric"] =  ["AdvancedMattesMutualInformation", "CorrespondingPointsEuclideanDistanceMetric"]
                affineParameterMap["Metric0Weight"] = ["0.50"] # the weight of 1st metric
                affineParameterMap["Metric1Weight"] =  ["0.50"] # the weight of 2nd metric

                elastixImageFilter.SetFixedPointSetFileName(fixed_point_path)
                elastixImageFilter.SetMovingPointSetFileName(moving_point_path)
            else:
                print(f'No point files found for {brain}')
                sys.exit()
            
            elastixImageFilter.SetParameterMap(affineParameterMap)
            #elastixImageFilter.AddParameterMap(bsplineParameterMap)

            elastixImageFilter.SetParameter("ResultImageFormat", "tif")
            elastixImageFilter.SetParameter("ComputeZYX", "true")
            elastixImageFilter.SetParameter("DefaultPixelValue", "0")
            elastixImageFilter.SetParameter("UseDirectionCosines", "false")
            elastixImageFilter.SetParameter("WriteResultImage", "false")
            elastixImageFilter.SetParameter("FixedImageDimension", "3")
            elastixImageFilter.SetParameter("MovingImageDimension", "3")
            elastixImageFilter.SetLogToFile(True)
            elastixImageFilter.LogToConsoleOff()
            elastixImageFilter.SetLogFileName('elastix.log')
            elastixImageFilter.SetOutputDirectory(savepath)
            elastixImageFilter.PrintParameterMap()

            resultImage = elastixImageFilter.Execute()
            resultImage = sitk.Cast(sitk.RescaleIntensity(resultImage), sitk.sitkUInt8)
            registered_images.append(sitk.GetArrayFromImage(resultImage))
            del resultImage

        reference_image = sitk.Cast(sitk.RescaleIntensity(reference_image), sitk.sitkUInt8)
        #registered_images.append(sitk.GetArrayFromImage(reference_image))
        avg_array = np.mean(registered_images, axis=0)
        avg_array = gaussian(avg_array, sigma=1)
        #avg_array = average_images(registered_images, iterations="500", default_pixel_value=defaul_pixel_value)

        #avg_array = gaussian(avg_array, 1.0)

        os.makedirs(savepath, exist_ok=True)
        save_atlas_path = os.path.join(savepath, f'AtlasV8_{self.um}um_{self.orientation}.tif')
        print(f'Saving img to {save_atlas_path}')
        write_image(save_atlas_path, avg_array.astype(np.uint8))


    def group_volume(self):

        population = ['MD585', 'MD594', 'MD589']
        vectorOfImages = sitk.VectorOfImage()
        for brain in population:
            brainpath = os.path.join(self.registration_path, brain, f'{brain}_{self.um}um_{self.orientation}.tif')
            if not os.path.exists(brainpath):
                print(f'{brainpath} does not exist, exiting.')
                sys.exit()
            vectorOfImages.push_back(sitk.ReadImage(brainpath))

        image = sitk.JoinSeries(vectorOfImages)

        # Register
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(image)
        elastixImageFilter.SetMovingImage(image)
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap('groupwise'))
        resultImage = elastixImageFilter.Execute()        
        resultImage = sitk.Cast(sitk.RescaleIntensity(resultImage), sitk.sitkUInt8)

        savepath = os.path.join(self.registration_path, 'AtlasV8')
        os.makedirs(savepath, exist_ok=True)
        save_atlas_path = os.path.join(savepath, f'AtlasV8_grouped_{self.um}um_{self.orientation}.tif')
        print(f'Saving img to {save_atlas_path}')
        write_image(save_atlas_path, resultImage.astype(np.uint8))



    def volume_origin_creation(self):
        #structureController = StructureCOMController('MD589')
        #polygonController = PolygonSequenceController('MD589')
        structureController = None
        polygonController = None
        pg_sessions = polygonController.get_available_volumes_sessions()
        animal_users = {}
        animals = set()
        for session in pg_sessions:
            animals.add(session.FK_prep_id)
            animal_users[session.FK_prep_id] = session.FK_user_id

        animal_polygon_com = {}
        for animal in sorted(animals):
            com_annotator_id = structureController.get_com_annotator(FK_prep_id=animal)
            if com_annotator_id is not None:
                animal_polygon_com[animal] = (com_annotator_id, animal_users[animal])

        brainMerger = BrainMerger(self.debug)
        """
        animal_polygon_com = {}
        animal_polygon_com['MD585'] = (2,3)
        animal_polygon_com['MD589'] = (2,3)
        animal_polygon_com['MD594'] = (2,3)
        """
        for animal, (com_annotator_id, polygon_annotator_id) in animal_polygon_com.items():
            if 'test' in animal or 'Atlas' in animal:
                continue
            
            print(f'{animal} {com_annotator_id} {polygon_annotator_id}')
            
            brainManager = BrainStructureManager(animal, 'all', self.debug)
            brainManager.polygon_annotator_id = polygon_annotator_id
            brainManager.fixed_brain = BrainStructureManager('MD589', self.debug)
            brainManager.fixed_brain.com_annotator_id = 2
            brainManager.com_annotator_id = com_annotator_id
            brainManager.compute_origin_and_volume_for_brain_structures(brainManager, brainMerger, 
                                                                        polygon_annotator_id)
            if brainManager.volume is not None:
                brainManager.save_brain_origins_and_volumes_and_meshes()

        if self.debug:
            return
        
        for structure in brainMerger.volumes_to_merge:
            volumes = brainMerger.volumes_to_merge[structure]
            volume = brainMerger.merge_volumes(structure, volumes)
            brainMerger.volumes[structure]= volume

        if len(brainMerger.origins_to_merge) > 0:
            print('Finished filling up volumes and origins')
            brainMerger.save_atlas_origins_and_volumes_and_meshes()
            brainMerger.save_coms_to_db()
            #brainMerger.evaluate(region)
            brainMerger.save_brain_area_data()
            print('Finished saving data to disk and to DB.')
        else:
            print('No data to save')


    def check_status(self):
        """Starter method to check for existing directories and files
        """
        status = []
        
        if self.fixed is not None and os.path.exists(self.fixed_volume_path):
            status.append(f'\tFixed volume at {self.fixed_volume_path}')
            arr = read_image(self.fixed_volume_path)
            status.append(f'\t\tshape={arr.shape} dtype={arr.dtype}')

        if os.path.exists(self.moving_volume_path):
            status.append(f'\tMoving volume at {self.moving_volume_path}')
            arr = read_image(self.moving_volume_path)
            status.append(f'\t\tshape={arr.shape} dtype={arr.dtype}')

        result_path = os.path.join(self.registration_output, self.registered_volume)
        if os.path.exists(result_path):
            status.append(f'\tRegistered volume at {result_path}')

        reverse_transformation_pfile = os.path.join(self.reverse_elastix_output, 'TransformParameters.0.txt')
        if os.path.exists(reverse_transformation_pfile):
            status.append(f'\tTransformParameters file to register points at: {reverse_transformation_pfile}')

        if os.path.exists(self.neuroglancer_data_path):
            status.append(f'\tPrecomputed data at: {self.neuroglancer_data_path}')


        fixed_point_path = os.path.join(self.registration_path, f'{self.fixed}_{self.um}um_{self.orientation}.pts')
        moving_point_path = os.path.join(self.registration_path, f'{self.moving}_{self.um}um_{self.orientation}.pts')

        if os.path.exists(fixed_point_path):
            status.append(f'\tFixed points at: {fixed_point_path}')
        if os.path.exists(moving_point_path):
            status.append(f'\tMoving points at: {moving_point_path}')
        if os.path.exists(self.unregistered_point_file):
            status.append(f'\tUnregistered moving points at: {self.unregistered_point_file}')
        if os.path.exists(self.registered_point_file):
            status.append(f'\tRegistered moving points at: {self.registered_point_file}')




        if len(status) > 0:
            print("These are the processes that have run:")
            print("\n".join(status))
        else:
            print(f'Nothing has been run to register {self.moving} at {self.um}um.')


def sort_from_center(polygon:list) -> list:
    """Get the center of the unique points in a polygon and then use math.atan2 to get
    the angle from the x-axis to the x,y point. Use that to sort.
    This only works with convex shaped polygons.
    
    :param polygon:
    """

    coords = np.array(polygon)
    coords = np.unique(coords, axis=0)
    center = coords.mean(axis=0)
    centered = coords - center
    angles = -np.arctan2(centered[:, 1], centered[:, 0])
    sorted_coords = coords[np.argsort(angles)]
    return list(map(tuple, sorted_coords))


def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def pad_volume(volume, padto):
    re = (padto[2] - volume.shape[2]) // 1
    ce = (padto[1] - volume.shape[1]) // 1
    return np.pad(volume, [[0, 0], [0, ce], [0, re]], constant_values=(0))
