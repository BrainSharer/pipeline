from collections import defaultdict
import os
import shutil
import sys
import ants
#from anyio import value
import numpy as np
from skimage import io
import dask.array as da
from dask import delayed
from dask.diagnostics import ProgressBar
from skimage.transform import resize
from scipy.ndimage import zoom
from skimage.filters import gaussian        
#from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import sqlalchemy
from tqdm import tqdm
import SimpleITK as sitk
import itk
from taskqueue import LocalTaskQueue
import igneous.task_creation as tc
import pandas as pd
import cv2
import json
import zarr
from shapely.geometry import Point, Polygon

from library.atlas.atlas_utilities import affine_transform_point, fetch_coms, list_coms, load_transformation, register_volume, resample_image
from library.controller.sql_controller import SqlController
from library.controller.annotation_session_controller import AnnotationSessionController
from library.database_model.annotation_points import AnnotationSession
from library.image_manipulation.neuroglancer_manager import NumpyToNeuroglancer
from library.image_manipulation.filelocation_manager import FileLocationManager
from library.utilities.utilities_mask import rescaler
from library.utilities.utilities_process import SCALING_FACTOR, get_scratch_dir, random_string, read_image, write_image
from library.atlas.brain_structure_manager import BrainStructureManager
from library.atlas.brain_merger import BrainMerger
from library.image_manipulation.image_manager import ImageManager
from library.utilities.utilities_registration import create_affine_parameters

# constants
MOVING_CROP = 50
M_UM_SCALE = 1000000


class VolumeRegistration:

    def __init__(self, moving, channel=1, annotation_id=0, xy_um=16, z_um=16,  scaling_factor=SCALING_FACTOR, fixed='Allen', orientation='sagittal', bspline=False, debug=False):
        self.registration_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
        self.moving_path = os.path.join(self.registration_path, moving)
        os.makedirs(self.moving_path, exist_ok=True)
        self.atlas_path = '/net/birdstore/Active_Atlas_Data/data_root/atlas_data' 
        self.allen_path = os.path.join(self.registration_path, 'Allen')
        self.tmp_dir = get_scratch_dir()
        self.moving = moving
        self.animal = moving
        self.debug = debug
        self.fixed = fixed
        self.xy_um = xy_um
        self.z_um = z_um
        self.mask_color = 254
        self.channel = f'C{channel}'
        self.annotation_id = annotation_id
        self.orientation = orientation
        self.bspline = bspline
        self.output_dir = f'{moving}_{fixed}_{z_um}x{xy_um}x{xy_um}um_{orientation}'
        self.scaling_factor = scaling_factor # This is the downsampling factor used to create the aligned volume at 10um
        self.fileLocationManager = FileLocationManager(self.moving)
        self.sqlController = SqlController(self.animal)
        self.thumbnail_aligned = os.path.join(self.fileLocationManager.prep, self.channel, 'thumbnail_aligned')
        self.moving_volume_path = os.path.join(self.moving_path, f'{self.moving}_{z_um}x{xy_um}x{xy_um}um_{orientation}.tif' )
        self.moving_nii_path = os.path.join(self.moving_path, f'{self.moving}_{z_um}x{xy_um}x{xy_um}um_{orientation}.nii' )
        self.registered_volume = os.path.join(self.moving_path, f'{self.moving}_{self.fixed}_{z_um}x{xy_um}x{xy_um}um_{orientation}.nii' )
        self.changes_path = os.path.join(self.moving_path, f'{self.moving}_{z_um}x{xy_um}x{xy_um}um_{orientation}_changes.json' )
        
        self.registration_output = os.path.join(self.moving_path, self.output_dir)
        self.elastix_output = os.path.join(self.registration_output, 'elastix_output')
        self.reverse_elastix_output = os.path.join(self.registration_output, 'reverse_elastix_output')
        
        self.registered_point_file = os.path.join(self.registration_output, 'outputpoints.txt')
        self.unregistered_point_file = os.path.join(self.moving_path, f'{self.animal}_{z_um}x{xy_um}x{xy_um}um_{orientation}_unregistered.pts')
        self.fiducial_moving_file_path = os.path.join(self.registration_path, self.moving, f'fiducials_{z_um}x{xy_um}x{xy_um}um_{self.orientation}.pts')

        self.transformation = 'Affine'
        self.transform_filepath = os.path.join(self.moving_path, f'{self.moving}_{self.fixed}_{self.z_um}x{self.xy_um}x{self.xy_um}um.tfm')
        self.inverse_transform_filepath = os.path.join(self.moving_path, f'{self.moving}_{self.fixed}_{self.z_um}x{self.xy_um}x{self.xy_um}um_inverse.tfm')


        if self.fixed is not None:
            self.fiducial_fixed_file_path = os.path.join(self.registration_path, self.fixed, f'fiducials_{z_um}x{xy_um}x{xy_um}um_{self.orientation}.pts')

        self.number_of_sampling_attempts = "10"
        if self.debug:
            self.iterations = "100"
            self.number_of_resolutions = "4"
        else:
            self.iterations = "2500"
            self.number_of_resolutions = "8"


        if fixed is not None:
            # Fixed is Allen @10um
            self.fixed = fixed
            self.fixed_path = os.path.join(self.registration_path, fixed)
            self.fixed_volume_path = os.path.join(self.fixed_path, f'{self.fixed}_{z_um}x{xy_um}x{xy_um}um_{orientation}.tif' )
            self.fixed_nii_path = os.path.join(self.fixed_path, 'Allen_10.0x10.0x10.0um_sagittal.nii' )
            self.neuroglancer_data_path = os.path.join(self.fileLocationManager.neuroglancer_data, f'{self.channel}_{self.fixed}_{z_um}x{xy_um}x{xy_um}um')
            os.makedirs(self.fixed_path, exist_ok=True)
        else:
            self.neuroglancer_data_path = os.path.join(self.fileLocationManager.neuroglancer_data, f'{self.channel}_{z_um}x{xy_um}x{xy_um}um')

        self.report_status()

    def report_status(self):
        """
        Prints the current status of the volume registration process with its settings.

        This method outputs a formatted summary of the key parameters used in the 
        volume registration process, including the preparation ID, unit of measurement, 
        orientation, debug mode status, number of resolutions, and rigid iterations.

        Attributes:
            animal (str): The preparation ID or identifier for the subject.
            um (float): The unit of measurement used in the registration process.
            orientation (str): The orientation setting for the volume registration.
            debug (bool): Indicates whether debug mode is enabled.
            number_of_resolutions (int): The number of resolutions used in the registration.
            affineIterations (int): The number of iterations for rigid affine transformations.
        """
        print("Running volume registration with the following settings:")
        print("\tprep_id:".ljust(20), f"{self.animal}".ljust(20))
        print("\tZ um:".ljust(20), f"{str(self.z_um)}".ljust(20))
        print("\tXY um:".ljust(20), f"{str(self.xy_um)}".ljust(20))
        print("\torientation:".ljust(20), f"{str(self.orientation)}".ljust(20))
        print("\tdebug:".ljust(20), f"{str(self.debug)}".ljust(20))
        print("\tresolutions:".ljust(20), f"{str(self.number_of_resolutions)}".ljust(20))
        print("\trigid iterations:".ljust(20), f"{str(self.iterations)}".ljust(20))
        print()



    def setup_transformix(self, outputpath):
        """Method used to transform volumes and points
        """

        number_of_transforms = 1
        
        os.makedirs(self.registration_output, exist_ok=True)

        transform_parameter0_path = os.path.join(outputpath, 'TransformParameters.0.txt')

        if not os.path.exists(transform_parameter0_path):
            print(f'{transform_parameter0_path} does not exist, exiting.')
            sys.exit()


        transformixImageFilter = sitk.TransformixImageFilter()
        parameterMap0 = sitk.ReadParameterFile(transform_parameter0_path)
        transformixImageFilter.SetTransformParameterMap(parameterMap0)
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
        pointfile = os.path.join(self.registration_path, self.moving, '{self.moving}_{self.z_um}x{self.xy_um}x{self.xy_um}_sagittal_registered.pts')
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
        else:
            print(f'Transforming {self.unregistered_point_file}')

        reverse_transformation_pfile = os.path.join(self.reverse_elastix_output, 'TransformParameters.0.txt')
        if not os.path.exists(reverse_transformation_pfile):
            print(f'{reverse_transformation_pfile} does not exist, exiting.')
            sys.exit()
        else:
            print(f'Using {reverse_transformation_pfile} for reverse transformation')
        
        transformixImageFilter = self.setup_transformix(self.reverse_elastix_output)
        transformixImageFilter.SetFixedPointSetFileName(self.unregistered_point_file)
        transformixImageFilter.Execute()

    def transformix_coms(self):
        formatted_registered_pointfile = os.path.join(self.registration_path, self.moving, '{self.moving}_{self.z_um}x{self.xy_um}x{self.xy_um}_sagittal_registered.pts')
        moving_all = fetch_coms(self.moving, self.um)
        structures = sorted(moving_all.keys())

        if not os.path.exists(self.registered_point_file):
            print(f'Error, registered point file does not exist: {self.registered_point_file}' )
            return
        else:
            print(f'Formatting {self.registered_point_file}')

        if os.path.exists(formatted_registered_pointfile):
            print(f'{formatted_registered_pointfile} exists, removing')
            os.remove(formatted_registered_pointfile)


        with open(self.registered_point_file, "r") as f:                
            lines=f.readlines()
            f.close()

        assert len(lines) == len(structures), f'Length of {self.registered_point_file}={len(lines)} != length of structures={len(structures)}'
        registered_com_path = os.path.join(self.atlas_path, 'AtlasV8', 'registered_coms')
        os.makedirs(registered_com_path, exist_ok=True)
        formatted_structures = {}
        point_or_index = 'OutputPoint'
        for i in range(len(lines)):        
            lx=lines[i].split()[lines[i].split().index(point_or_index)+3:lines[i].split().index(point_or_index)+6] #x,y,z
            lf = [float(f) for f in lx]
            x = lf[0]
            y = lf[1]
            z = lf[2]
            structure = structures[i]
            #print(structure,  int(x), int(y), int(z))
            formatted_structures[structure] = (x,y,z)
            com_filepath = os.path.join(registered_com_path, f'{structure}.txt')
            np.savetxt(com_filepath, (x,y,z))
        print(formatted_structures)



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
        """id for ALLEN771602, cerebellum test is 8357
        """
        transform_file = os.path.join(self.reverse_elastix_output, 'TransformParameters.0.txt')
        
        if not os.path.exists(transform_file):
            print(f'{transform_file} does not exist, exiting.')
            sys.exit()
        else:
            print(f'Using {transform_file} for reverse transformation')
        elastix_affine_matrix = elastix_to_affine_4x4(transform_file)
        print('Affine matrix from elastix:')
        print(elastix_affine_matrix)
        #if not os.path.exists(self.inverse_transform_filepath):
        #    print(f'{self.inverse_transform_filepath} does not exist, exiting.')
        #    sys.exit()
        #else:
        #    print(f'Using {self.inverse_transform_filepath} for reverse transformation')
        #transform = sitk.ReadTransform(self.inverse_transform_filepath)
        #ants_affine_matrix = create_affine_matrix_from_mat(self.inverse_transform_filepath)
        #print('Affine matrix from ants:')
        #print(ants_affine_matrix)
        # check for necessary files
        if not os.path.exists(self.reverse_elastix_output):
            print(f'{self.reverse_elastix_output} does not exist, exiting.')
            sys.exit()
        
        sqlController = SqlController(self.moving) 
        id = 8357
        annotation_session = sqlController.get_annotation_by_id(id)
        childJsons = annotation_session.annotation['childJsons']
        rows = []
        polygons = defaultdict(list)
        input_points = itk.PointSet[itk.F, 3].New()
        transformed_polygons = defaultdict(list)
        for child in childJsons:
            for i, row in enumerate(child['childJsons']):
                x,y,z = row['pointA']
                rows.append((x,y,z))
        df = pd.DataFrame(rows, columns=['xm','ym','zm'])

        print(f'Creating polygons for {self.moving} DB full resolution: scaling factors: xy_um={self.xy_um} z_um={self.z_um} ')
        df['x'] = df['xm'] * M_UM_SCALE / self.xy_um
        df['y'] = df['ym'] * M_UM_SCALE / self.xy_um
        df['z'] = df['zm'] * M_UM_SCALE / self.z_um

        df.drop(columns=['xm', 'ym', 'zm'], inplace=True)

        for idx, (_, row) in enumerate(df.iterrows()):
            x = row['x']
            y = row['y']
            z = row['z']
            section = int(round(row['z']))
            polygons[section].append((x, y))
            #input_points.GetPoints().InsertElement(idx, (x,y,z))
            tx, ty, tz = affine_transform_point((x,y,z), elastix_affine_matrix)
            transformed_section = int(round(tz))
            transformed_polygons[transformed_section].append((tx, ty))
            print(f"({tx}, {ty}, {tz}),")
        exit(1)

        for section, points in transformed_polygons.items():
            print(f'Section {section} has {len(points)} points average points: {np.mean(points, axis=0)}')


        def write_polygons_to_files(input_dir, output_dir, polygons, desc):
            
            ##### Draw the scaled and transformed polygons on the images
            if not os.path.exists(input_dir):
                print(f'{input_dir} does not exist for drawing transformed polygons, exiting.')
                return
            else:
                print(f'Drawing polygons on images from: {input_dir}')
                print(f'Saving polygons to: {output_dir}')
            if os.path.exists(output_dir):
                print(f'Removing: {output_dir}')
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            for section, points in tqdm(polygons.items(), desc=desc):
                file = str(section).zfill(4) + ".tif"
                inpath = os.path.join(input_dir, file)
                if not os.path.exists(inpath):
                    print(f'{inpath} does not exist')
                    continue
                img = cv2.imread(inpath, cv2.IMREAD_GRAYSCALE)
                points = np.array(points)
                points = points.astype(np.int32)
                cv2.polylines(img, pts = [points], isClosed=True, color=255, thickness=3)
                outpath = os.path.join(output_dir, file)
                cv2.imwrite(outpath, img)

        ##### Draw the scaled polygons on the images
        input_dir = os.path.join(self.fileLocationManager.prep, self.channel, f'{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal')
        output_dir = os.path.join(self.fileLocationManager.prep, self.channel, f'drawn_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal')
        write_polygons_to_files(input_dir, output_dir, polygons, desc='Drawing scaled polygons')

        """
        ### Now use transformix on the points
        transformix_pointset_file = os.path.join(self.registration_output, "transformix_input_points.txt")
        if os.path.exists(transformix_pointset_file):
            print(f'Removing existing {transformix_pointset_file}')
            os.remove(transformix_pointset_file)

        with open(transformix_pointset_file, "w") as f:
            f.write("point\n")
            f.write(f"{input_points.GetNumberOfPoints()}\n")
            for idx in range(input_points.GetNumberOfPoints()):
                point = input_points.GetPoint(idx)
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
                
        transformixImageFilter = self.setup_transformix(self.reverse_elastix_output)
        transformixImageFilter.SetFixedPointSetFileName(transformix_pointset_file)
        transformixImageFilter.Execute()

        del polygons            
        transformed_polygons = defaultdict(list)
        with open(self.registered_point_file, "r") as f:                
            lines=f.readlines()
            f.close()

        print(f'Number of lines in {self.registered_point_file}: {len(lines)}')

        point_or_index = 'OutputPoint'
        points = []
        for i in tqdm(range(len(lines))):        
            lx=lines[i].split()[lines[i].split().index(point_or_index)+3:lines[i].split().index(point_or_index)+6] #x,y,z
            lf = [float(f) for f in lx]
            x = lf[0]
            y = lf[1]
            z = lf[2]
            section = int(np.round(z))
            transformed_polygons[section].append((x,y))
            #points.append((x,y,section))
        """
        ##### Draw the scaled polygons on the images
        input_dir = os.path.join(self.fileLocationManager.prep, self.channel, f'registered_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal')
        output_dir = os.path.join(self.fileLocationManager.prep, self.channel, f'registered_drawn_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal')
        write_polygons_to_files(input_dir, output_dir, transformed_polygons, desc='Drawing scaled polygons')






    def transformix_coms_db(self):
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
        e.g., to get to 10um isotropic with a 0.325x0.325x20 typical brain whose images (thumbnail_aligned)
        are downsampled by 1/32 do:
        xy change = 0.325 * 32 / 10um
        z change = 20 * 32 / 10um

        """
        image_manager = ImageManager(self.thumbnail_aligned)

        xy_resolution = self.sqlController.scan_run.resolution * self.scaling_factor
        z_resolution = self.sqlController.scan_run.zresolution
        print(f'Using images from {self.thumbnail_aligned} to create volume')
        print(f'xy_resolution={xy_resolution} z_resolution={z_resolution} scaling_factor={self.scaling_factor} scan run resolution={self.sqlController.scan_run.resolution} um={self.xy_um}')
        """

        change_z = z_resolution
        change_y = xy_resolution
        change_x = xy_resolution
        change = (change_z, change_y, change_x) 
        changes = {'change_z': change_z, 'change_y': change_y, 'change_x': change_x}
        print(f'change_z={change_z} change_y={change_y} change_x={change_x}')
        print(f'Using files from {self.thumbnail_aligned} to create volume at\n\t {self.moving_volume_path}')

        if self.debug:
            return
        with open(self.changes_path, 'w') as f:
            json.dump(changes, f)            
        
        if os.path.exists(self.moving_volume_path):
            print(f'{self.moving_volume_path} exists, exiting')
            return
        """
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
        sitk_image = sitk.GetImageFromArray(image_stack.astype(image_manager.dtype))    
        sitk_image.SetOrigin((0,0,0))
        sitk_image.SetSpacing((xy_resolution, xy_resolution, z_resolution))
        sitk_image.SetDirection((1,0,0,0,1,0,0,0,1))
        sitk.WriteImage(sitk_image, self.moving_nii_path)
            
        #zoomed = zoom(image_stack, change)
        
        #write_image(self.moving_volume_path, zoomed.astype(image_manager.dtype))
        print(f'Saved a 3D volume {self.moving_nii_path} with shape={sitk_image.GetSize()} and spacing={sitk_image.GetSpacing()}')

    def downsample_stack(self):
        image_manager = ImageManager(self.full_aligned)



    def pad_volume(self):
        pad = 500
        volume = io.imread(self.fixed_volume_path)
        dtype = volume.dtype
        print(f'volume shape={volume.shape} dtype={volume.dtype}')
        volume = np.concatenate((volume, np.zeros((volume.shape[0], pad, volume.shape[2])) ), axis=1)
        print(f'volume shape={volume.shape} dtype={volume.dtype}')
        #volume = np.concatenate((volume, np.zeros((volume.shape[0], volume.shape[1], pad)) ), axis=2)
        #print(f'volume shape={volume.shape} dtype={volume.dtype}')
        outpath = os.path.join(self.registration_path, f'{self.fixed}_{self.z_um}x{self.xy_um}x{self.xy_um}_{self.orientation}_padded.tif')
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

        scale_xy = int(self.xy_um * 1000)
        scale_z = int(self.z_um * 1000)
        scales = (scale_xy, scale_xy, scale_z)
        os.makedirs(PRECOMPUTED, exist_ok=True)
        volume = read_image(volumepath)
        volume = volume.astype(np.uint8)
        volume = np.swapaxes(volume, 0, 2)
        num_channels = 1
        volume_size = volume.shape
        
        print(f'volume shape={volume.shape} dtype={volume.dtype} creating at {PRECOMPUTED}')
        #volume = normalize16(volume)

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
        """Register volumes to Allen 10um
        """
        
        def resample_to_isotropic(img, iso=0.1):
            """Resample image to isotropic spacing."""
            spacing = [iso, iso, iso]
            original_spacing = img.GetSpacing()
            original_size = img.GetSize()

            new_size = [
                int(round(osz * ospc / iso))
                for osz, ospc in zip(original_size, original_spacing)
            ]

            print(f"Original size: {original_size}, spacing: {original_spacing}")
            print(f"New size: {new_size}, spacing: {spacing}")

            return sitk.Resample(
                img,
                new_size,
                sitk.Transform(),
                sitk.sitkLinear,
                img.GetOrigin(),
                spacing,
                img.GetDirection(),
                0.0,
                img.GetPixelID(),
            )

        # Load fixed and moving images
        fixed = sitk.ReadImage(self.fixed_nii_path, sitk.sitkFloat32)
        print(f"Read fixed image: {self.fixed_nii_path}")
        if os.path.exists(self.moving_nii_path):
            moving = sitk.ReadImage(self.moving_nii_path, sitk.sitkFloat32)
            print(f"Read moving image: {self.moving_nii_path}")
        else:
            moving_path = os.path.join(self.registration_path, self.moving, f'{self.moving}_10.4x10.4x20um_sagittal.nii')
            if not os.path.exists(moving_path):
                print(f'Input for moving does not exist {moving_path}')
                print('You need to create a standard volume from the thumbnail_aligned dir. Exiting.')
                sys.exit()
            else:
                moving = sitk.ReadImage(moving_path)
                print(f"Read moving image: {moving_path}")
                moving = resample_to_isotropic(moving, iso=10.0)
                print("Resampled images to isotropic spacing of 10.0 um")
                sitk.WriteImage(moving, self.moving_nii_path)
                print(f'Wrote resampled image to {self.moving_nii_path}')



        # ------------------------------------------------------------
        # 2. Normalize intensities (helpful for microscopy)
        # ------------------------------------------------------------
        #fixed_iso  = sitk.Normalize(fixed_iso)
        #moving_iso = sitk.Normalize(moving_iso)
        # ------------------------------------------------------------
        # 3. Initial alignment using center of mass
        # ------------------------------------------------------------
        initial_transform = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        # ------------------------------------------------------------
        # 4. Rigid+Affine registration (MI metric)
        # ------------------------------------------------------------
        registration = sitk.ImageRegistrationMethod()

        registration.SetMetricAsMattesMutualInformation(100)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(0.2)

        registration.SetInterpolator(sitk.sitkLinear)

        registration.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=500,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        registration.SetOptimizerScalesFromPhysicalShift()

        registration.SetShrinkFactorsPerLevel([8, 4, 2, 1])
        registration.SetSmoothingSigmasPerLevel([3, 2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        registration.SetInitialTransform(initial_transform, inPlace=False)

        affine_transform = registration.Execute(fixed, moving)

        print("Affine done. Final metric:", registration.GetMetricValue())
        resampled = sitk.Resample(
            moving,
            fixed,
            affine_transform,
            sitk.sitkLinear,
            0.0,
            moving.GetPixelID(),
        )
        """
        # Initial alignment of the centers of the two volumes
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, 
            moving_image, 
            sitk.AffineTransform(3), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        # Set up the registration method
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(0.2)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsGradientDescent(
            learningRate=1.0, 
            numberOfIterations=3000, 
            convergenceMinimumValue=1e-6, 
            convergenceWindowSize=10)

        R.SetOptimizerScalesFromPhysicalShift()
        R.SetShrinkFactorsPerLevel([8, 4, 2, 1])
        R.SetSmoothingSigmasPerLevel([4, 2, 1, 0])
        R.SetInitialTransform(initial_transform, inPlace=False)
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Perform registration
        affine_transform = R.Execute(fixed_image, moving_image)
        print("Final metric value: ", R.GetMetricValue())
        print("Optimizer's stopping condition: ", R.GetOptimizerStopConditionDescription())
        # Resample moving image onto fixed image grid
        resampled = sitk.Resample(moving_image, fixed_image, affine_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
        resampled = sitk.Cast(sitk.RescaleIntensity(resampled), pixel_type)
        """

        sitk.WriteImage(resampled, self.registered_volume)
        print(f"Resampled moving image written to {self.registered_volume}")

        # Save the transform
        sitk.WriteTransform(affine_transform, self.transform_filepath)
        print(f"Registration written to {self.transform_filepath}")
        return
    

    def register_volume_elastix(self):
        """
        Registers a moving volume to a fixed volume using elastix and saves the resulting registered volume.
        This method performs the following steps:

        1. Removes the existing elastix output directory if it exists.
        2. Checks if the fixed and moving volume paths are provided; exits if not.
        3. Creates the elastix output directory if it does not exist.
        4. Sets up the registration process using the provided fixed and moving volume paths.
        5. Executes the registration and processes the resulting image.
        6. Saves the registered volume to the specified output path.
        
        Attributes:
            self.elastix_output (str): Path to the directory where elastix output will be stored.
            self.fixed_path (str): Path to the fixed volume image.
            self.moving_path (str): Path to the moving volume image.
            self.fixed (str): Identifier for the fixed volume.
            self.moving (str): Identifier for the moving volume.
            self.um (int): Resolution of the images in micrometers.
            self.orientation (str): Orientation of the images.
            self.registered_volume (str): Path to save the registered volume.
        Raises:
            SystemExit: If either `self.fixed_path` or `self.moving_path` is None.
        Outputs:
            - The registered volume is saved to the path specified by `self.registered_volume`.
            - Prints status messages to indicate progress and output paths.
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
        fixed_basename = f'{self.fixed}_{self.z_um}x{self.xy_um}x{self.xy_um}um_{self.orientation}'
        moving_basename = f'{self.moving}_{self.z_um}x{self.xy_um}x{self.xy_um}um_{self.orientation}'
        #fixed_path, moving_path, moving_point_path, fixed_point_path, fixed_basename, moving_basename
        elastixImageFilter = self.setup_registration(fixed_path, moving_path, 
                                                     self.fiducial_moving_file_path, 
                                                     self.fiducial_fixed_file_path,
                                                     fixed_basename, moving_basename)
        elastixImageFilter.SetOutputDirectory(self.elastix_output)
        elastixImageFilter.PrintParameterMap()
        resultImage = elastixImageFilter.Execute()
                
        resultImage = sitk.Cast(sitk.RescaleIntensity(resultImage), sitk.sitkUInt16)

        sitk.WriteImage(resultImage, self.registered_volume)
        parameterMap = elastixImageFilter.GetTransformParameterMap()[0]
        TransformParameters = parameterMap['TransformParameters']
        center = parameterMap['CenterOfRotationPoint']
        print(f'TransformParameters: {TransformParameters}')
        print(f'Center of rotation point: {center}')
        R = [float(x) for x in TransformParameters[:9]]
        t = [float(x) for x in TransformParameters[9:]]
        c = [float(x) for x in center]

        affine_transform = sitk.AffineTransform(3)
        affine_transform.SetMatrix(R)
        affine_transform.SetTranslation(t)
        affine_transform.SetCenter(c)

        R = np.array(R).reshape((3, 3))
        t = np.array(t)
        affine = np.eye(4)
        affine[:3, :3] = R
        r_trans = (np.dot(R, c) - c - t).T * [1, 1, -1]

        affine[:3, 3] = r_trans



        sitk.WriteTransform(affine_transform, self.affine_matrix_path)
        np.save(self.affine_matrix_path.replace('.tfm', '.npy'), affine)

        print(f'Saved img to {self.registered_volume}')
        print(f'Saved affine matrix to {self.affine_matrix_path}')

    def reverse_register_volume(self):
        """This method also uses an affine and a bspline registration process, but it does 
        it in reverse. The fixed and moving images get switched so we can get the transformation
        for the points to get registered to the atlas. 
        """
       
        os.makedirs(self.reverse_elastix_output, exist_ok=True)
        # switch fixed and moving
        fixed_path = self.moving_path
        moving_path = self.fixed_path
        fixed_basename = f'{self.moving}_{self.z_um}x{self.xy_um}x{self.xy_um}um_{self.orientation}'
        moving_basename = f'{self.fixed}_{self.z_um}x{self.xy_um}x{self.xy_um}um_{self.orientation}'
        elastixImageFilter = self.setup_registration(fixed_path, moving_path, 
                                                     self.fiducial_fixed_file_path,
                                                     self.fiducial_moving_file_path, 
                                                     fixed_basename, moving_basename)

        elastixImageFilter.SetOutputDirectory(self.reverse_elastix_output)
        elastixImageFilter.Execute()
        print('Done performing inverse')

    def setup_registration(self, fixed_path, moving_path, moving_point_path, fixed_point_path, fixed_basename, moving_basename):
        
        fixed_path = os.path.join(fixed_path, f'{fixed_basename}.tif' )
        moving_path = os.path.join(moving_path, f'{moving_basename}.tif') 

        if not os.path.exists(fixed_path):
            print(f'Fixed {fixed_path} does not exist')
            sys.exit()
        if not os.path.exists(moving_path):
            print(f'Moving {moving_path} does not exist')
            sys.exit()
        # set point paths
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
            bsplineParameterMap["MaximumNumberOfIterations"] = [self.iterations] # 250 works ok
            if not self.debug:
                bsplineParameterMap["WriteResultImage"] = ["false"]
                bsplineParameterMap["UseDirectionCosines"] = ["true"]
                bsplineParameterMap["FinalGridSpacingInVoxels"] = [f"{self.xy_um}"]
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
            print(f'\nUsing fiducial points, fixed points: {fixed_count} moving points: {moving_count}\n')
            elastixImageFilter.SetParameter("Registration", ["MultiMetricMultiResolutionRegistration"])
            elastixImageFilter.SetParameter("Metric",  ["AdvancedMattesMutualInformation", "CorrespondingPointsEuclideanDistanceMetric"])
            elastixImageFilter.SetParameter("Metric0Weight", ["0.5"]) # the weight of 1st metric
            elastixImageFilter.SetParameter("Metric1Weight",  ["0.5"]) # the weight of 2nd metric

            elastixImageFilter.SetFixedPointSetFileName(fixed_point_path)
            elastixImageFilter.SetMovingPointSetFileName(moving_point_path)
        else:
            print(f'Fixed point path {fixed_point_path} or \nmoving point path {moving_point_path} do not exist')

        if self.bspline:
            elastixImageFilter.AddParameterMap(bsplineParameterMap)

        elastixImageFilter.SetParameter("MaximumNumberOfIterations", self.iterations)
        elastixImageFilter.SetParameter("ResultImageFormat", "tif")
        elastixImageFilter.SetParameter("NumberOfResolutions", self.number_of_resolutions) #### Very important, less than 6 gives lousy results.
        elastixImageFilter.SetParameter("DefaultPixelValue", "0")
        elastixImageFilter.SetParameter("ComputeZYX", "true")
        
        elastixImageFilter.PrintParameterMap
        elastixImageFilter.SetLogToFile(True)
        elastixImageFilter.LogToConsoleOff()

        elastixImageFilter.SetLogFileName('elastix.log')

        return elastixImageFilter


    def crop_volume(self):

        moving_volume = io.imread(self.moving_volume_path)
        moving_volume = moving_volume[:,MOVING_CROP:500, MOVING_CROP:725]
        savepath = os.path.join(self.registration_path, f'Atlas_{self.z_um}x{self.xy_um}x{self.xy_um}_{self.orientation}.tif')
        print(f'Saving img to {savepath}')
        io.imsave(savepath, moving_volume)

    def create_brain_coms(self):
        """
        Creates brain center of mass (COM) files for a set of brains and validates their consistency.
        This method performs the following steps:

        1. Validates that the number of COM files for each brain is consistent.
        2. Generates a `.pts` file for each brain containing the COM points.
        
        The method uses the following attributes:

        - `self.registration_path`: Path to the registration directory.
        - `self.um`: Unit scaling factor for coordinates.
        - `self.orientation`: Orientation of the brain data.
        
        The method processes a predefined list of brains and assumes the COM files are stored
        in a specific directory structure.
        Raises:
            SystemExit: If the number of COM files is inconsistent across brains.
        Outputs:
            - Prints the number of COM files for each brain.
            - Prints an error message and exits if the number of COM files is inconsistent.
            - Writes `.pts` files containing the COM points for each brain.
        
        Note:
            If a COM file contains "SC" in its path, the corresponding coordinates are printed
            to the console for debugging purposes.
        """

        brains = ['MD585', 'MD594', 'MD589', 'AtlasV8', ]
        base_com_path = '/net/birdstore/Active_Atlas_Data/data_root/atlas_data'
        number_of_coms = {}
        for brain in tqdm(brains, desc='Validating brain coms'):
            brain_point_path = os.path.join(self.registration_path, brain, f'{brain}_{self.z_um}x{self.xy_um}x{self.xy_um}_{self.orientation}.pts')
            brain_com_path = os.path.join(base_com_path, brain, 'com')
            comfiles = sorted(os.listdir(brain_com_path))
            nfiles = len(comfiles)
            number_of_coms[brain] = nfiles
            print(f'{brain} has {nfiles} coms')
        
        test_length = len(set(list(number_of_coms.values())))
        if test_length > 1:
            print(f'Error, the number of coms for each brain is not the same: {number_of_coms}')
            sys.exit()


        for brain in tqdm(brains, desc='Creating brain coms'):
            brain_point_path = os.path.join(self.registration_path, brain, f'{brain}_{self.z_um}x{self.xy_um}x{self.xy_um}_{self.orientation}.pts')
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

    def create_moving_fixed_points(self):
        if self.fixed is None or self.moving is None:
            print('Specify both moving and fixed brains, exiting')
            sys.exit()


        moving_all = list_coms(self.moving, scaling_factor=self.um)
        fixed_all = list_coms(self.fixed, scaling_factor=self.um)

        bad_keys = ('RtTg', 'AP')

        common_keys = list(moving_all.keys() & fixed_all.keys())
        good_keys = set(common_keys) - set(bad_keys)

        moving_src = {k:moving_all[k] for k in good_keys}
        fixed_src = {k:fixed_all[k] for k in good_keys}
        print(f'Found {len(good_keys)} common keys')
        moving_file = open(self.fiducial_moving_file_path, "w")
        fixed_file = open(self.fiducial_fixed_file_path, "w")
        moving_file.write('point\n')
        moving_file.write(f'{len(good_keys)}\n')
        fixed_file.write('point\n')
        fixed_file.write(f'{len(good_keys)}\n')
        
        for (mk,mv),(fk,fv) in zip(moving_src.items(), fixed_src.items()):
            if mk != fk:
                print(f'Error, moving key {mk} does not match fixed key {fk}')
                continue
            mx, my, mz = mv
            fx, fy, fz = fv
            moving_file.write(f'{mx} {my} {mz}')
            moving_file.write('\n')
            fixed_file.write(f'{fx} {fy} {fz}')
            fixed_file.write('\n')
            print(f'{fk} {fv}\t{mv}')
        moving_file.close()
        fixed_file.close()


    def create_average_volume(self):
        volumes = {}
        moving_brains = ['MD585', 'MD594', 'MD589']
        for brain in tqdm(moving_brains, 'Adding registered volume'):
            brainpath = os.path.join(self.registration_path, brain, f'{brain}_{self.z_um}x{self.xy_um}x{self.xy_um}um_{self.orientation}.tif')
            if not os.path.exists(brainpath):
                print(f'{brainpath} does not exist, exiting.')
                return
            brainimg = read_image(brainpath)
            volumes[brain] = sitk.GetImageFromArray(brainimg.astype(np.float32))

        images = [img for img in volumes.values()]
        reference_image = max(images, key=lambda img: np.prod(img.GetSize()))
        resampled_images = [resample_image(img, reference_image) for img in images]
        registered_images = [register_volume(img, reference_image, "500", "0") for img in resampled_images if img != reference_image]
        avg_array = np.mean(registered_images, axis=0)
        avg_array = gaussian(avg_array, sigma=1)
        savepath = os.path.join(self.registration_path, 'AtlasV8')
        save_atlas_path = os.path.join(savepath, f'MDXXX_{self.z_um}x{self.xy_um}x{self.xy_um}_{self.orientation}.tif')
        print(f'Saving img to {save_atlas_path}')
        write_image(save_atlas_path, avg_array.astype(np.uint8))


    def group_volume(self):

        population = ['MD585', 'MD594', 'MD589']
        vectorOfImages = sitk.VectorOfImage()
        for brain in population:
            brainpath = os.path.join(self.registration_path, brain, f'{brain}_{self.z_um}x{self.xy_um}x{self.xy_um}_{self.orientation}.tif')
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
        save_atlas_path = os.path.join(savepath, f'AtlasV8_grouped_{self.z_um}x{self.xy_um}x{self.xy_um}_{self.orientation}.tif')
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

    def register_volume_with_fiducials(self):
        """
        Registers two 3D numpy volumes using affine transformation based on fiducials.
        
        Args:
            fixed_volume_np (np.ndarray): The fixed/reference 3D volume.
            moving_volume_np (np.ndarray): The moving 3D volume.
            fixed_fiducials (np.ndarray): Nx3 array of fiducials for the fixed volume.
            moving_fiducials (np.ndarray): Nx3 array of fiducials for the moving volume.

        Returns:
            registered_moving_image (sitk.Image): The transformed moving image.
            transform (sitk.Transform): The computed affine transformation.
        """
        moving = 'MD594'
        reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
        moving_path = os.path.join(reg_path, moving, 'MD594_10um_sagittal.tif')
        fixed_path = os.path.join(reg_path, 'Allen', 'Allen_10um_sagittal.tif')
        moving_volume_np = read_image(moving_path)
        fixed_volume_np = read_image(fixed_path)

        # Convert NumPy volumes to SimpleITK images
        fixed_image = sitk.GetImageFromArray(fixed_volume_np)
        moving_image = sitk.GetImageFromArray(moving_volume_np)

        # Define fiducials for fixed and moving images
        moving_all = list_coms(self.moving, scaling_factor=self.um)
        fixed_all = list_coms(self.fixed, scaling_factor=self.um)
        bad_keys = ('RtTg', 'AP')
        common_keys = list(moving_all.keys() & fixed_all.keys())
        good_keys = set(common_keys) - set(bad_keys)
        fixed_fiducials = [fixed_all[k] for k in good_keys]
        moving_fiducials = [moving_all[k] for k in good_keys]

        # Convert fiducial points to SimpleITK point sets
        fixed_points = [tuple(p) for p in fixed_fiducials]
        moving_points = [tuple(p) for p in moving_fiducials]

        # Compute affine transform using fiducials
        transform = sitk.LandmarkBasedTransformInitializer(
            sitk.AffineTransform(3), fixed_points, moving_points
        )

        # Resample moving image
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(transform)

        registered_moving_image = resampler.Execute(moving_image)
        registered_moving_image = sitk.Cast(sitk.RescaleIntensity(registered_moving_image), sitk.sitkUInt16)
        registered_moving_image.WriteImage(self.registered_volume)
        print(f'Saved img to {self.registered_volume}')
        transform.WriteTransform(self.registered_transform_file)


    def zarr2tif(self):
        input_zarr = os.path.join(self.fileLocationManager.prep, self.channel, f'{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal.zarr')
        output_dir = os.path.join(self.fileLocationManager.prep, self.channel, f'{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal')
        if os.path.exists(output_dir):
            print(f"Output directory {output_dir} already exists. Removing.")
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.isdir(input_zarr):
            print(f"Zarr dir not found at {input_zarr}")
            exit(1)
        volume = zarr.open(input_zarr, mode='r')
        print(volume.info)
        for i in tqdm(range(int(volume.shape[0])), disable=self.debug, desc="Creating tifs from zarr"): # type: ignore
            section = volume[i, ...]
            section = np.squeeze(section)
            ids, _ = np.unique(section, return_counts=True)
            lids = len(ids)
            if lids > 75:
                section = rescaler(section)
            else:
                section = np.zeros_like(section, dtype=np.uint16)

            filepath = os.path.join(output_dir, f'{str(i).zfill(4)}.tif')
            if os.path.exists(filepath):
                continue
            write_image(filepath, section)

        input_dir = output_dir
        del output_dir
        image_manager = ImageManager(input_dir)
        
        if os.path.exists(self.moving_volume_path):
            print(f'{self.moving_volume_path} exists, exiting')
            return

        image_stack = np.zeros(image_manager.volume_size)
        file_list = []
        for ffile in tqdm(image_manager.files, desc='Creating volume'):
            fpath = os.path.join(input_dir, ffile)
            farr = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            file_list.append(farr)

            
        image_stack = np.stack(file_list, axis = 0)    

        write_image(self.moving_volume_path, image_stack.astype(image_manager.dtype))
        print(f'Saved a 3D volume {self.moving_volume_path} with shape={image_stack.shape} and dtype={image_stack.dtype}')

    def volume2tif(self):
        
        output_dir = os.path.join(self.fileLocationManager.prep, self.channel, f'{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal')
        if os.path.exists(output_dir):
            print(f"Output directory {output_dir} already exists. Removing.")
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        if self.fixed is not None:
            volume_path = self.register_volume
        else:
            volume_path = self.moving_volume_path

        if os.path.exists(volume_path):
            print(f'Loading volume from {volume_path}')
            volume = read_image(volume_path)
        else:
            print(f"{volume_path} not found, exiting.")
            return

        print(f'Loading volume with shape: {volume.shape}, dtype: {volume.dtype}')
        for i in tqdm(range(int(volume.shape[0])), disable=self.debug, desc="Creating tifs from 3D volume"): # type: ignore
            section = volume[i, ...]
            section = np.squeeze(section)
            ids, _ = np.unique(section, return_counts=True)
            lids = len(ids)
            if lids > 75:
                section = rescaler(section)
            else:
                section = np.zeros_like(section, dtype=np.uint16)

            filepath = os.path.join(output_dir, f'{str(i).zfill(4)}.tif')
            write_image(filepath, section)

        print(f'Saved a 3D stack at {output_dir} with {len(os.listdir(output_dir))} sections')


    def volume2nifti(self):
        print(f'Converting volume to nifti format')
        print(f'Moving volume path: {self.moving_volume_path}')
        if os.path.exists(self.moving_volume_path):
            img = read_image(self.moving_volume_path)
        else:
            print(f'Moving volume not found at {self.moving_volume_path}')
            return
    
        # Convert numpy array to SimpleITK image
        sitk_img = sitk.GetImageFromArray(img)
        
        # Optionally set spacing (if known, e.g., microns per voxel)
        # sitk_img.SetSpacing((0.5, 0.5, 2.0))  # example spacing (x, y, z)
        
        output_path = self.moving_volume_path.replace('.tif', '.nii')
        if os.path.exists(output_path):
            print(f"NIfTI file already exists at: {output_path}, removing.")
            os.remove(output_path)

        # Write to NIfTI format
        voxel_size = (self.z_um, self.xy_um, self.xy_um)  # z, y, x in microns
        origin = (0.0, 0.0, 0.0)
        direction = np.eye(3).flatten().tolist()
        sitk_img.SetSpacing(voxel_size)
        sitk_img.SetOrigin(origin)
        sitk_img.SetDirection(direction)
        print(f'Setting spacing to: {voxel_size}')

        sitk.WriteImage(sitk_img, output_path)
        
        print(f"✅ Saved NIfTI file to: {output_path}")        


    def tif2zarr(self):
        """
        Downsample a folder of large TIFFs by (z, y, x) scale and write to Zarr.

        Parameters:
        - input_folder: str, path to folder with TIFFs (assumed to be Z-stack)
        - output_zarr: str, path to output Zarr directory
        - scale_factors: tuple of (scale_z, scale_y, scale_x), e.g., (0.5, 0.5, 0.5)
        - chunk_size: tuple of (z, y, x) chunk size
        - multichannel: whether the TIFFs are RGB
        """
        sqlController = SqlController(self.moving) 
        scale_xy = sqlController.scan_run.resolution
        z_scale = sqlController.scan_run.zresolution


        input_dir = os.path.join(self.fileLocationManager.prep, self.channel, 'full_aligned')
        output_zarr = os.path.join(self.fileLocationManager.prep, self.channel, f'{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal.zarr')
        if not os.path.isdir(input_dir):
            print(f"Input directory not found at {input_dir}")
            exit(1)
        else:
            print(f"Input directory found at {input_dir}", end= " ")
        files = sorted(os.listdir(input_dir))
        if len(files) == 0:
            print(f"No TIFF files found in {input_dir}")
            exit(1)
        else:
            print(f"with {len(files)} TIFF files")

        scale_x = scale_xy / self.xy_um
        scale_y = scale_xy / self.xy_um
        scale_z = z_scale / self.z_um

        print(f"Downsampling by (z, y, x) scale: ({scale_z}, {scale_y}, {scale_x})")

        # Load one image to get shape
        sample_path = os.path.join(input_dir, files[0])
        sample = read_image(sample_path)
        is_rgb = sample.ndim == 3
        original_shape = sample.shape
        chunk_size = (1, 256, 256)

        def load_and_resize(path):
            filepath = os.path.join(input_dir, path)
            img = read_image(filepath)
            if is_rgb:
                new_shape = (
                    int(img.shape[0] * scale_y),
                    int(img.shape[1] * scale_x),
                    img.shape[2],
                )
            else:
                new_shape = (
                    int(img.shape[0] * scale_y),
                    int(img.shape[1] * scale_x),
                )
            return resize(img, new_shape, anti_aliasing=True, preserve_range=True).astype(img.dtype)

        # Create delayed images
        lazy_imgs = [delayed(load_and_resize)(f) for f in files]

        # Convert to dask array
        sample_down = load_and_resize(sample_path)
        dask_imgs = da.stack([da.from_delayed(im, shape=sample_down.shape, dtype=sample_down.dtype) for im in lazy_imgs])

        # Downsample Z dimension if needed
        if scale_z != 1.0:
            new_z = int(len(files) * scale_z)

            def resize_z(arr):
                return resize(arr, (new_z,) + arr.shape[1:], anti_aliasing=True, preserve_range=True).astype(arr.dtype)

            dask_imgs = da.from_array(resize_z(dask_imgs.compute()), chunks=chunk_size)

        # Save to Zarr
        print(f"Saving downsampled stack to {output_zarr}")

        if os.path.exists(output_zarr):
            print(f"Zarr directory {output_zarr} already exists. Removing.")
            shutil.rmtree(output_zarr)

        with ProgressBar():
            dask_imgs.to_zarr(output_zarr, overwrite=True)

        print(f"✅ Downsampled stack saved to {output_zarr}")

        volume = zarr.open(output_zarr, mode='r')
        print(volume.info)



    def points_within_polygons(self):
        

        transform = load_transformation(self.moving, self.z_um, self.xy_um)
        if transform is None:
            print('No transformation found, exiting')
            return
        sqlController = SqlController(self.moving)
        xy_resolution = sqlController.scan_run.resolution
        z_resolution = sqlController.scan_run.zresolution
        id = self.annotation_id
        annotator_id = 1 # Hard coded to edward
        existing_annotation_session = sqlController.get_annotation_by_id(id)
        label_objects = existing_annotation_session.labels
        labels = [label_object.label for label_object in label_objects]
        childJsons = existing_annotation_session.annotation['childJsons']
        json_entry = {}
        rows = []
        props = ["#00FF00", 1, 1, 5, 3, 1]
        parentAnnotationId = random_string()
        for child in childJsons[10:20]:
            if 'point' in child:
                xm0, ym0, zm0 = child['point'] # data is in meters
                xm0 *= M_UM_SCALE  / xy_resolution / 32
                ym0 *= M_UM_SCALE / xy_resolution / 22
                zm0 *= M_UM_SCALE / z_resolution
                if self.debug:
                    print(f"[{xm0}, {ym0}, {zm0}],")
                xt, yt, zt = transform.GetInverse().TransformPoint((xm0, ym0, zm0)) # transformed data to 10um
                xm = xt / M_UM_SCALE * self.xy_um # back to meters
                ym = yt / M_UM_SCALE * self.xy_um
                zm = zt / M_UM_SCALE * self.z_um
                rows.append({'point': [xm, ym, zm], 'type': 'point', 'parentAnnotationId': parentAnnotationId, 'props': props})
        if self.debug:
            return        
        json_entry["source"] = np.min([row["point"] for row in rows], axis=0).tolist()
        json_entry["centroid"] = np.mean([row["point"] for row in rows], axis=0).tolist()
        json_entry["childrenVisible"] = True
        json_entry["type"] = "cloud"
        description = existing_annotation_session.annotation.get("description", f"{self.moving}-Allen data")
        description = description + " Registered to Allen @ 10um"
        json_entry["description"] = description
        json_entry["props"] = props
        json_entry["childJsons"] = rows
        
        try:
            annotation_session = (
                self.sqlController.session.query(AnnotationSession)
                .filter(AnnotationSession.active == True)
                .filter(AnnotationSession.FK_user_id == annotator_id)
                .filter(AnnotationSession.FK_prep_id == self.moving)
                .filter(AnnotationSession.annotation["description"] == description)
                .one_or_none()
            )
        except Exception as e:
            print(f"Found more than one structure for {self.moving} {description}. Exiting program, please fix")
            print(e)
            exit(1)

        
        if annotation_session is None:
            print(f'Inserting {self.moving} with {description}')
            
            try:
                self.sqlController.insert_annotation_with_labels(
                    FK_user_id=annotator_id,
                    FK_prep_id=self.moving,
                    annotation=json_entry,
                    labels=labels)
            except sqlalchemy.exc.OperationalError as e:
                print(f"Operational error inserting annotation: {e}")
                self.sqlController.session.rollback()
            
        else:                
            update_dict = {'annotation': json_entry}
            print(f'Updating {self.moving} session {annotation_session.id} with {description}')
            self.sqlController.update_session(annotation_session.id, update_dict=update_dict)

        print('\nfinished processing points')


    def check_status(self):
        """Starter method to check for existing directories and files
        """
        status = []
        
        if self.fixed is not None:
            if os.path.exists(self.fixed_volume_path):
                status.append(f'\tFixed volume at {self.fixed_volume_path}')
                arr = read_image(self.fixed_volume_path)
                status.append(f'\t\tshape={arr.shape} dtype={arr.dtype}')
            else:
                status.append(f'\tFixed volume at {self.fixed_volume_path} does not exist')

        if os.path.exists(self.moving_volume_path):
            status.append(f'\tMoving volume at {self.moving_volume_path}')
            arr = read_image(self.moving_volume_path)
            status.append(f'\t\tshape={arr.shape} dtype={arr.dtype}')
        else:
            status.append(f'\tMoving volume at {self.moving_volume_path} does not exist')

        result_path = os.path.join(self.registration_output, self.registered_volume)
        if os.path.exists(result_path):
            status.append(f'\tRegistered volume at {result_path}')

        reverse_transformation_pfile = os.path.join(self.reverse_elastix_output, 'TransformParameters.0.txt')
        if os.path.exists(reverse_transformation_pfile):
            status.append(f'\tTransformParameters file to register points at: {reverse_transformation_pfile}')

        if os.path.exists(self.neuroglancer_data_path):
            status.append(f'\tPrecomputed data at: {self.neuroglancer_data_path}')


        fixed_point_path = os.path.join(self.registration_path, f'{self.fixed}_{self.z_um}x{self.xy_um}x{self.xy_um}_{self.orientation}.pts')
        moving_point_path = os.path.join(self.registration_path, f'{self.moving}_{self.z_um}x{self.xy_um}x{self.xy_um}_{self.orientation}.pts')

        if os.path.exists(fixed_point_path):
            status.append(f'\tFixed points at: {fixed_point_path}')
        if os.path.exists(moving_point_path):
            status.append(f'\tMoving points at: {moving_point_path}')
        if os.path.exists(self.unregistered_point_file):
            status.append(f'\tUnregistered moving points at: {self.unregistered_point_file}')
        if os.path.exists(self.registered_point_file):
            status.append(f'\tRegistered moving points at: {self.registered_point_file}')


        if os.path.exists(self.transform_filepath):
            status.append(f'\tRegistered transform at: {self.transform_filepath}')


        if len(status) > 0:
            print("These are the processes that have run:")
            print("\n".join(status))
        else:
            print(f'Nothing has been run to register {self.moving} at {self.z_um}x{self.xy_um}x{self.xy_um}.')


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

def point_in_polygon(x, y, points):
    """
    Check if a point (x, y) is inside a polygon defined by polygon_points.

    Parameters:
        x (float): X coordinate of the point.
        y (float): Y coordinate of the point.
        polygon_points (list of tuple): List of (x, y) tuples defining the polygon.

    Returns:
        bool: True if the point is inside the polygon, False otherwise.
    """
    point = Point(x, y)
    polygon = Polygon(points)
    return polygon.contains(point)


def elastix_to_affine_4x4(filepath):
    """
    Converts an Elastix TransformParameters.0.txt affine file into a 4x4 numpy matrix in ZYX order.
    """
    import re
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract TransformParameters
    params_match = re.search(r'\(TransformParameters\s+(.*?)\)', content, re.DOTALL)
    if not params_match:
        raise ValueError("TransformParameters not found in the file.")
    params = list(map(float, params_match.group(1).split()))

    if len(params) != 12:
        raise ValueError(f"Expected 12 TransformParameters, got {len(params)}")

    # Extract CenterOfRotationPoint
    center_match = re.search(r'\(CenterOfRotationPoint\s+(.*?)\)', content)
    if not center_match:
        raise ValueError("CenterOfRotationPoint not found in the file.")
    center = np.array(list(map(float, center_match.group(1).split())))

    # Build the affine matrix A and translation vector t
    A = np.array(params[:9]).reshape(3, 3)
    t = np.array(params[9:])

    # Apply the formula: M = A·(x - c) + t + c  => Full affine:
    # M = A x + (t - A·c + c) = A x + offset
    offset = t - A @ center + center


    # Compose into 4x4 affine matrix
    affine = np.eye(4)
    affine[:3, :3] = A
    affine[:3, 3] = t
    return affine

    # Reorder from XYZ to ZYX by permuting rows and columns
    reorder = [2, 1, 0]  # Z, Y, X
    affine_zyx = np.eye(4)
    affine_zyx[:3, :3] = affine[:3, :3][reorder, :][:, reorder]
    affine_zyx[:3, 3] = affine[:3, 3][reorder]

    return affine_zyx

def create_affine_matrix_from_mat(filepath):
    from scipy.io import loadmat

    transfo_dict = loadmat(filepath)
    lps2ras = np.diag([-1, -1, 1])

    rot = transfo_dict['AffineTransform_float_3_3'][0:9].reshape((3, 3))
    trans = transfo_dict['AffineTransform_float_3_3'][9:12]
    offset = transfo_dict['fixed']
    r_trans = (np.dot(rot, offset) - offset - trans).T * [1, 1, -1]

    matrix = np.eye(4)
    matrix[0:3, 3] = r_trans
    matrix[:3, :3] = np.dot(np.dot(lps2ras, rot), lps2ras)

    translation = (matrix[..., 3][0:3])
    #translation = 0
    return matrix


def linear_stretch(old_min, old_max, x, stretch):
    new_max = old_max * stretch
    new_min = old_min
    return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def apply_affine_to_points(points, affine):
    """
    Applies a 4x4 affine transformation to a list of (x, y, z) points.

    Parameters:
        affine: np.ndarray of shape (4, 4)
        points: tuple (x,y,z)

    Returns:
        transformed points: np.ndarray of shape (N, 3)
    """
    points = np.array((points))
    points = points.reshape(1,3)
    print(f'shape={points.shape} type of points={type(points)}')
    if points.shape[1] != 3:
        print("Points array must be of shape (N, 3)")
        exit(1)
    
    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    homogeneous_points = np.hstack([points, ones])
    
    # Apply affine transformation
    transformed = homogeneous_points @ affine.T
    return transformed[:, :3][0].tolist()  # Return as list of tuples

def transform_points(points_xyz, transform):
    """
    Apply a SimpleITK transform to a list of (x, y, z) points.

    :param points_xyz: Nx3 numpy array of points
    :param transform: SimpleITK.Transform object
    :return: Nx3 numpy array of transformed points
    """
    transformed_points = [transform.TransformPoint(p) for p in points_xyz]
    return np.array(transformed_points)
