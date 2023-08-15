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
import sys
import numpy as np
from skimage import io
from skimage.exposure import rescale_intensity
from tqdm import tqdm
import SimpleITK as sitk
from taskqueue import LocalTaskQueue
import igneous.task_creation as tc
import pandas as pd
import cv2
#from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache


from library.controller.polygon_sequence_controller import PolygonSequenceController
from library.controller.sql_controller import SqlController
from library.image_manipulation.neuroglancer_manager import NumpyToNeuroglancer
from library.image_manipulation.filelocation_manager import FileLocationManager, data_path
from library.utilities.utilities_mask import normalize8, smooth_image
from library.utilities.utilities_process import read_image
from library.controller.annotation_session_controller import AnnotationSessionController
from library.controller.structure_com_controller import StructureCOMController

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

class VolumeRegistration:
    """This class takes a downsampled image stack and registers it to the Allen volume
    Data resides in /net/birdstore/Active_Atlas_Data/data_root/brains_info/registration
    Volumes have the following naming convention:
        1. image stack = {animal}_{um}um_{orientation}.tif -> MD589_25um_sagittal.tif
        2. registered volume = {moving}_{fixed}_{um}um_{orientation}.tif -> MD585_MD589_25um_sagittal.tif
        3. point file = {animal}_{um}um_{orientation}.pts -> MD589_25um_sagittal.pts
        
    """

    def __init__(self, moving, channel=1, um=25, fixed='Allen', orientation='sagittal', debug=False):
        DATA = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
        self.data = DATA
        self.moving = moving
        self.animal = moving
        self.debug = debug
        self.fixed = fixed
        self.um = um
        self.mask_color = 254
        self.channel = f'CH{channel}'
        self.orientation = orientation
        self.output_dir = f'{moving}_{fixed}_{um}um_{orientation}'
        self.scaling_factor = 64 # This is the downsampling factor used to create the aligned volume
        self.fileLocationManager = FileLocationManager(self.moving)
        self.thumbnail_aligned = os.path.join(self.fileLocationManager.prep, self.channel, 'thumbnail_aligned')
        self.moving_volume_path = os.path.join(self.data, f'{self.moving}_{um}um_{orientation}.tif' )
        self.fixed_volume_path = os.path.join(self.data, f'{self.fixed}_{um}um_{orientation}.tif' )
        
        self.registration_output = os.path.join(self.data, self.output_dir)
        self.elastix_output = os.path.join(self.registration_output, 'elastix_output')
        self.reverse_elastix_output = os.path.join(self.registration_output, 'reverse_elastix_output')
        
        self.registered_point_file = os.path.join(self.registration_output, 'outputpoints.txt')
        self.unregistered_point_file = os.path.join(self.data, f'{self.animal}_{um}um_{orientation}_unregistered.pts')
        self.neuroglancer_data_path = os.path.join(self.fileLocationManager.neuroglancer_data, f'{self.channel}_{self.fixed}{um}um')
        self.number_of_sampling_attempts = "10"
        if self.debug:
            iterations = "100"
            self.number_of_resolutions = "4"
            self.rigidIterations = iterations
            self.affineIterations = iterations
            self.bsplineIterations = iterations
        else:
            self.number_of_resolutions = "6"
            self.rigidIterations = "1000"
            self.affineIterations = "2500"
            self.bsplineIterations = "15000"

        if not os.path.exists(self.fixed_volume_path):
            print(f'{self.fixed_volume_path} does not exist, exiting.')
            sys.exit()        
 


    def setup_transformix(self, outputpath):
        """Method used to transform volumes and points
        """
        
        os.makedirs(self.registration_output, exist_ok=True)

        transformixImageFilter = sitk.TransformixImageFilter()
        parameterMap0 = sitk.ReadParameterFile(os.path.join(outputpath, 'TransformParameters.0.txt'))
        parameterMap1 = sitk.ReadParameterFile(os.path.join(outputpath, 'TransformParameters.1.txt'))
        parameterMap2 = sitk.ReadParameterFile(os.path.join(outputpath, 'TransformParameters.2.txt'))
        #parameterMap3 = sitk.ReadParameterFile(os.path.join(outputpath, 'TransformParameters.3.txt'))
        transformixImageFilter.SetTransformParameterMap(parameterMap0)
        transformixImageFilter.AddTransformParameterMap(parameterMap1)
        transformixImageFilter.AddTransformParameterMap(parameterMap2)
        #transformixImageFilter.AddTransformParameterMap(parameterMap3)
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
        sitk.WriteImage(transformed, os.path.join(self.registration_output, 'result.tif'))

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


    def transformix_points(self):
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
        
        transformixImageFilter = self.setup_transformix(self.reverse_elastix_output)
        transformixImageFilter.SetFixedPointSetFileName(self.unregistered_point_file)
        transformixImageFilter.Execute()


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
        structureController = StructureCOMController(self.moving)
        coms = structureController.get_coms(self.moving, annotator_id=com_annotator_id)

        point_or_index = 'OutputPoint'
        source='COMPUTER'
        self.moving = 'Atlas'
        sessionController = AnnotationSessionController(self.moving)

        with open(self.registered_point_file, "r") as f:                
            lines=f.readlines()
            f.close()

        assert len(lines) == len(coms), f'Length of {self.registered_point_file}={len(lines)} != length of coms={len(coms)}'

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
                annotation_session = sessionController.get_annotation_session(self.moving, brain_region.id, 1)
                entry = {'source': source, 'FK_session_id': annotation_session.id, 'x': x, 'y':y, 'z': z}
                sessionController.upsert_structure_com(entry)
            else:
                print(f'No brain region found for {structure}')

            if self.debug and brain_region is not None:
                #lf = [round(l) for l in lf]
                print(annotation_session.id, self.moving, brain_region.id, source, structure,  int(x), int(y), int(z))


    def transformix_polygons(self):
        sqlController = SqlController(self.moving)
        polygon = PolygonSequenceController(animal=self.moving)        
        scale_xy = sqlController.scan_run.resolution
        z_scale = sqlController.scan_run.zresolution
        #input_points = itk.PointSet[itk.F, 3].New()
        input_points = None
        df_L = polygon.get_volume(self.moving, 38, 12)
        df_R = polygon.get_volume(self.moving, 38, 13)
        frames = [df_L, df_R]
        df = pd.concat(frames)
        len_L = df_L.shape[0]
        len_R = df_R.shape[0]
        len_total = df.shape[0]
        assert len_L + len_R == len_total, "Lengths of dataframes do not add up."
        
        TRANSFORMIX_POINTSET_FILE = os.path.join(self.registration_output,"transformix_input_points.txt")        
        #df = polygon.get_volume(self.moving, 3, 33)

        for idx, (_, row) in enumerate(df.iterrows()):
            x = row['coordinate'][0]/scale_xy/self.scaling_factor
            y = row['coordinate'][1]/scale_xy/self.scaling_factor
            z = row['coordinate'][2]/z_scale
            point = [x,y,z]
            input_points.GetPoints().InsertElement(idx, point)
        del df
        
        with open(TRANSFORMIX_POINTSET_FILE, "w") as f:
            f.write("point\n")
            f.write(f"{input_points.GetNumberOfPoints()}\n")
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
            for idx in range(input_points.GetNumberOfPoints()):
                point = input_points.GetPoint(idx)
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
                
        transformixImageFilter = self.setup_transformix(self.reverse_elastix_output)
        transformixImageFilter.SetFixedPointSetFileName(TRANSFORMIX_POINTSET_FILE)
        transformixImageFilter.Execute()
        
        polygons = defaultdict(list)
        with open(self.registration_point_file, "r") as f:                
            lines=f.readlines()
            f.close()

        point_or_index = 'OutputPoint'
        for i in range(len(lines)):        
            lx=lines[i].split()[lines[i].split().index(point_or_index)+3:lines[i].split().index(point_or_index)+6] #x,y,z
            lf = [float(f) for f in lx]
            x = lf[0]
            y = lf[1]
            z = lf[2]
            section = int(np.round(z))
            polygons[section].append((x,y))
        resultImage = io.imread(os.path.join(self.registration_output, 'result.tif'))
        resultImage = normalize8(resultImage)
        
        for section, points in polygons.items():
            points = sort_from_center(points)
            points = np.array(points)
            points = points.astype(np.int32)
            cv2.fillPoly(resultImage[section,:,:], pts = [points], color = self.mask_color)
            cv2.polylines(resultImage[section,:,:], [points], isClosed=True, color=(self.mask_color), 
                          thickness=4)
        
        #for i in range(resultImage.shape[0]):
        #    section = int(points[0][2])
        #    x = int(points[0][0])
        #    y = int(points[0][1])
        #    if i == section:
        #        print(x,y,section)
        #        cv2.circle(resultImage[section,:,:], (x,y), 12, 254, thickness=3)
        outpath = os.path.join(self.registration_output, 'annotated.tif')
        io.imsave(outpath, resultImage)
        print(f'Saved a 3D volume {outpath} with shape={resultImage.shape} and dtype={resultImage.dtype}')


    def transformix_coms(self):
        com_annotator_id = 2
        controller = StructureCOMController(self.moving)
        coms = controller.get_COM(self.moving, annotator_id=com_annotator_id)

        
        TRANSFORMIX_POINTSET_FILE = os.path.join(self.registration_output,"transformix_input_points.txt")        
        with open(TRANSFORMIX_POINTSET_FILE, "w") as f:
            f.write("point\n")
            f.write(f"{len(coms)}\n")

            for idx, (structure, (x,y,z)) in enumerate(coms.items()):
                x /= self.um
                y /= self.um
                z /= self.um
                print(idx, structure, x,y,z)
                f.write(f"{x} {y} {z}\n")
                
        transformixImageFilter = self.setup_transformix(self.reverse_elastix_output)
        transformixImageFilter.SetFixedPointSetFileName(TRANSFORMIX_POINTSET_FILE)
        transformixImageFilter.ExecuteInverse()


    def fill_contours(self):
        sqlController = SqlController(self.moving)
        # vars
        INPUT = os.path.join(self.movingLocationManager.prep, 'CH1', 'thumbnail_aligned')
        OUTPUT = os.path.join(self.movingLocationManager.prep, 'CH1', 'thumbnail_merged')
        os.makedirs(OUTPUT, exist_ok=True)
        polygon = PolygonSequenceController(animal=self.moving)        
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
            inpath = os.path.join(INPUT, file)
            if not os.path.exists(inpath):
                print(f'{inpath} does not exist')
                continue
            img = cv2.imread(inpath, cv2.IMREAD_GRAYSCALE)
            points = np.array(points)
            points = points.astype(np.int32)
            cv2.fillPoly(img, pts = [points], color = color)
            outpath = os.path.join(OUTPUT, file)
            cv2.imwrite(outpath, img)

        files = sorted(os.listdir(INPUT))
        for file in tqdm(files):
            inpath = os.path.join(INPUT, file)
            img = cv2.imread(inpath, cv2.IMREAD_GRAYSCALE)
            outpath = os.path.join(OUTPUT, file)
            if not os.path.exists(outpath):
                cv2.imwrite(outpath, img)


    def get_file_information(self):
        """Get information about the mid file in the image stack

        :return files: list of files in the directory
        :return volume_size: tuple of numpy shape
        """

        files = sorted(os.listdir(self.thumbnail_aligned))
        midpoint = len(files) // 20
        midfilepath = os.path.join(self.thumbnail_aligned, files[midpoint])
        midfile = read_image(midfilepath)
        rows = midfile.shape[0]
        columns = midfile.shape[1]
        volume_size = (rows, columns, len(files))
        return files, volume_size, midfile.dtype


    def create_volume(self):
        """Create a 3D volume of the image stack
        """
        #from skimage.filters import gaussian        
        #from skimage.exposure import rescale_intensity
        files, volume_size, dtype = self.get_file_information()
        image_stack = np.zeros(volume_size)
        file_list = []
        #clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(4, 4))
        for ffile in tqdm(files):
            fpath = os.path.join(self.thumbnail_aligned, ffile)
            farr = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            #blur = cv2.GaussianBlur(farr, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
            #farr = rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255))
            #farr = normalize8(farr)
            #print(farr.dtype)
            #farr = clahe.apply(farr)
            farr[farr == 255] = 0
            farr = smooth_image(farr)


            #blur = cv2.blur(farr,(9,9))
            #blur2 = cv2.GaussianBlur(farr,(3,3),0)
            #farr = cv2.equalizeHist(cv2.absdiff(blur2,blur))
            #farr = cv2.GaussianBlur(farr,(3,3),0)
            #farr = farr[200:-200,200:-200]
            file_list.append(farr)
        image_stack = np.stack(file_list, axis = 0)
        io.imsave(self.moving_volume_path, image_stack.astype(dtype))
        print(f'Saved a 3D volume {self.moving_volume_path} with shape={image_stack.shape} and dtype={image_stack.dtype}')

    def create_precomputed(self):
        chunk = 64
        chunks = (chunk, chunk, chunk)
        volumepath = os.path.join(self.registration_output, 'result.tif')
        if not os.path.exists(volumepath):
            print(f'{volumepath} does not exist, exiting.')
            sys.exit()
            
        PRECOMPUTED = self.neuroglancer_data_path
        scale = self.um * 1000
        scales = (scale, scale, scale)
        os.makedirs(PRECOMPUTED, exist_ok=True)
        volume = read_image(volumepath)
        volume = np.swapaxes(volume, 0, 2)
        num_channels = 1
        volume_size = volume.shape
        print(f'volume shape={volume.shape} dtype={volume.dtype}')
        #volume = normalize16(volume)
        print(f'volume shape={volume.shape} dtype={volume.dtype}')

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

        os.makedirs(self.elastix_output, exist_ok=True)

        elastixImageFilter = self.setup_registration(self.fixed, self.moving)
        elastixImageFilter.SetOutputDirectory(self.elastix_output)
        resultImage = elastixImageFilter.Execute()         
        resultImage = sitk.Cast(sitk.RescaleIntensity(resultImage), sitk.sitkUInt8)
        savepath = os.path.join(self.elastix_output, 'result.tif')
        sitk.WriteImage(resultImage, savepath)
        print(f'Saved img to {savepath}')

    def reverse_register_volume(self):
        """This method also uses an affine and a bspline registration process, but it does 
        it in reverse. The fixed and moving images get switched so we can get the transformation
        for the points to get registered to the atlas. 
        """
       
        os.makedirs(self.reverse_elastix_output, exist_ok=True)

        elastixImageFilter = self.setup_registration(self.moving, self.fixed)
        elastixImageFilter.SetOutputDirectory(self.reverse_elastix_output)
        elastixImageFilter.Execute()
        print(f'Done performing inverse')

    def setup_registration(self, fixed, moving):
        
        fixed_path = os.path.join(self.data, f'{fixed}_{self.um}um_{self.orientation}.tif' )
        moving_path = os.path.join(self.data, f'{moving}_{self.um}um_{self.orientation}.tif' )

        if not os.path.exists(fixed_path):
            print(f'{fixed_path} does not exist')
            return
        if not os.path.exists(moving_path):
            print(f'{moving_path} does not exist')
            return
        # set point paths
        fixed_point_path = os.path.join(self.data, f'{fixed}_{self.um}um_{self.orientation}.pts')
        moving_point_path = os.path.join(self.data, f'{moving}_{self.um}um_{self.orientation}.pts')
        print(f'moving volume path={moving_path}')
        print(f'fixed volume path={fixed_path}')
        print(f'moving point path={moving_point_path}')
        print(f'fixed point path={fixed_point_path}')
        
        fixedImage = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
        movingImage = sitk.ReadImage(moving_path, sitk.sitkFloat32)
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixedImage)
        elastixImageFilter.SetMovingImage(movingImage)

        transParameterMap = sitk.GetDefaultParameterMap('translation')
        rigidParameterMap = sitk.GetDefaultParameterMap('rigid')
        rigidParameterMap["NumberOfResolutions"] = ["6"] # Takes lots of RAM
        rigidParameterMap["MaximumNumberOfIterations"] = [self.rigidIterations] 

        affineParameterMap = sitk.GetDefaultParameterMap('affine')
        affineParameterMap["UseDirectionCosines"] = ["true"]
        affineParameterMap["MaximumNumberOfIterations"] = [self.affineIterations] # 250 works ok
        affineParameterMap["MaximumNumberOfSamplingAttempts"] = [self.number_of_sampling_attempts]
        affineParameterMap["NumberOfResolutions"]= [self.number_of_resolutions] # Takes lots of RAM
        affineParameterMap["WriteResultImage"] = ["false"]

        bsplineParameterMap = sitk.GetDefaultParameterMap('bspline')
        bsplineParameterMap["MaximumNumberOfIterations"] = [self.bsplineIterations] # 250 works ok
        if not self.debug:
            bsplineParameterMap["WriteResultImage"] = ["false"]
            bsplineParameterMap["UseDirectionCosines"] = ["true"]
            bsplineParameterMap["FinalGridSpacingInVoxels"] = [f"{self.um}"]
            bsplineParameterMap["MaximumNumberOfSamplingAttempts"] = [self.number_of_sampling_attempts]
            bsplineParameterMap["NumberOfResolutions"]= ["6"]
            bsplineParameterMap["GridSpacingSchedule"] = ["6.219", "4.1", "2.8", "1.9", "1.4", "1.0"]
            del bsplineParameterMap["FinalGridSpacingInPhysicalUnits"]

        elastixImageFilter.SetParameterMap(transParameterMap)
        elastixImageFilter.AddParameterMap(rigidParameterMap)
        elastixImageFilter.AddParameterMap(affineParameterMap)
        if os.path.exists(fixed_point_path) and os.path.exists(moving_point_path):
            with open(fixed_point_path, 'r') as fp:
                fixed_count = len(fp.readlines())
            with open(moving_point_path, 'r') as fp:
                moving_count = len(fp.readlines())
            assert fixed_count == moving_count, f'Error, the number of fixed points in {fixed_point_path} do not match {moving_point_path}'

            elastixImageFilter.SetParameter("Registration", ["MultiMetricMultiResolutionRegistration"])
            elastixImageFilter.SetParameter("Metric",  ["AdvancedMattesMutualInformation", "CorrespondingPointsEuclideanDistanceMetric"])
            elastixImageFilter.SetParameter("Metric0Weight", ["0.5"]) # the weight of 1st metric
            elastixImageFilter.SetParameter("Metric1Weight",  ["0.5"]) # the weight of 2nd metric

            elastixImageFilter.SetFixedPointSetFileName(fixed_point_path)
            elastixImageFilter.SetMovingPointSetFileName(moving_point_path)
        elastixImageFilter.AddParameterMap(bsplineParameterMap)
        elastixImageFilter.SetLogToFile(True)
        elastixImageFilter.LogToConsoleOff()


        elastixImageFilter.SetParameter("WriteIterationInfo",["true"])
        elastixImageFilter.SetLogFileName('elastix.log')

        return elastixImageFilter


    def create_average_volume(self):
        brains = ['MD589_25um_sagittal.tif', 'MD585_MD589.tif', 'MD594_MD589.tif']
        volumes = []
        for brain in brains:
            brainpath = os.path.join(self.fixed_path, brain)
            if not os.path.exists(brainpath):
                print(f'{brainpath} does not exist, exiting.')
                sys.exit()
            brainimg = read_image(brainpath)
            if brainimg.dtype == np.uint8:
                brainimg = brainimg.astype(np.float32)
            volumes.append(brainimg)

        merged_volume = np.sum(volumes, axis=0)
        average_volume = merged_volume / float(len(volumes))
        average_volume = normalize8(average_volume)
        savepath = os.path.join(self.fixed_path, f'Atlas_{self.um}um_{self.orientation}.tif')
        print(f'Saving img to {savepath}')
        io.imsave(savepath, average_volume)


    def check_registration(self):
        """Starter method to check for existing directories and files
        """
        status = []
        
        if os.path.exists(self.fixed_volume_path):
            status.append(f'\tFixed volume at {self.fixed_volume_path}')

        if os.path.exists(self.moving_volume_path):
            status.append(f'\tMoving volume at {self.moving_volume_path}')

        result_path = os.path.join(self.registration_output, 'result.tif')
        if os.path.exists(result_path):
            status.append(f'\tRegistered volume at {result_path}')

        reverse_transformation_pfile = os.path.join(self.reverse_elastix_output, 'TransformParameters.3.txt')
        if os.path.exists(reverse_transformation_pfile):
            status.append(f'\tTransformParameters file to register points at: {reverse_transformation_pfile}')

        if os.path.exists(self.neuroglancer_data_path):
            status.append(f'\tPrecomputed data at: {self.neuroglancer_data_path}')

        if os.path.exists(self.unregistered_point_file):
            status.append(f'\tUnnregisted points at: {self.unregistered_point_file}')


        fixed_point_path = os.path.join(self.data, f'{self.fixed}_{self.um}um_{self.orientation}.pts')
        moving_point_path = os.path.join(self.data, f'{self.moving}_{self.um}um_{self.orientation}.pts')

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
            print(f'Nothing has been run to register {self.moving} to {self.fixed} with channel {self.channel}.')
