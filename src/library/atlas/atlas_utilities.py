import os
from pathlib import Path
import sys
import numpy as np
import SimpleITK as sitk
from scipy.spatial.transform import Rotation as R
from skimage.filters import gaussian
from scipy.ndimage import affine_transform

from library.controller.sql_controller import SqlController
from library.registration.algorithm import umeyama
from library.utilities.utilities_process import M_UM_SCALE, SCALING_FACTOR



ORIGINAL_ATLAS = 'AtlasV7'
NEW_ATLAS = 'AtlasV8'
RESOLUTION = 0.452
ALLEN_UM = 10


def apply_affine_transform(point: list, matrix) -> np.ndarray:
    """
    Applies an affine transformation to a 3D point.

    Parameters:
    point (tuple or list): A tuple (x, y, z) representing the 3D point.
    matrix (numpy array): A 4x4 affine transformation matrix.

    Returns:
    numpy array: Transformed (x', y', z') coordinates as a numpy array.
    """


    if len(point) != 3:
        raise ValueError("Point must be a 3-element tuple or list (x, y, z)")

    if matrix.shape != (4, 4):
        raise ValueError("Matrix must be a 4x4 numpy array")

    # Convert the point to homogeneous coordinates
    homogeneous_point = np.array([point[0], point[1], point[2], 1])

    # Apply the transformation
    transformed_point = np.dot(matrix, homogeneous_point)

    # Return the transformed x, y, z coordinates (ignoring the homogeneous coordinate)
    return transformed_point[:3]

def affine_transform_volume(volume, matrix):
    """Apply an affine transformation to a 3D volume."""
    if matrix.shape != (4, 4):
        raise ValueError("Matrix must be a 4x4 numpy array")
    translation = (matrix[..., 3][0:3])
    translation = 0
    matrix = matrix[:3, :3]
    transformed_volume = affine_transform(volume, matrix, offset=translation, order=1)
    return transformed_volume



def list_coms(animal, scaling_factor=1):
    """
    Lists the COMs from the annotation session table. The data
    is stored in meters and is then converted to micrometers.
    """
    sqlController = SqlController(animal)

    coms = {}
    com_dictionaries = sqlController.get_com_dictionary(prep_id=animal)
    if len(com_dictionaries.keys()) == 0:
        return coms
    for k, v in com_dictionaries.items():

        com = [i* M_UM_SCALE/scaling_factor for i in v]
        coms[k] = com

    if len(coms.keys()) == 0:
        coms = fetch_coms(animal, scaling_factor=scaling_factor)

    return coms


def fetch_coms(animal, scaling_factor=1):
    """
    Fetches the COMs from disk. The data is stored in micrometers.
    """
    
    coms = {}
    dirpath = f'/net/birdstore/Active_Atlas_Data/data_root/atlas_data/{animal}/com'
    if not os.path.exists(dirpath):
        return coms
    files = sorted(os.listdir(dirpath))
    for file in files:
        structure = Path(file).stem
        filepath = os.path.join(dirpath, file)
        com = np.loadtxt(filepath)
        com /= scaling_factor 
        coms[structure] = com
    return coms

def compute_affine_transformation_centroid(source_points, target_points):
    """
    Computes the affine transformation (scale, shear, rotation, translation) between two sets of 3D points.
    
    Parameters:
        set1: np.ndarray of shape (N, 3) - Source set of 3D points
        set2: np.ndarray of shape (N, 3) - Target set of 3D points
    
    Returns:
        A: np.ndarray of shape (3, 3) - Linear transformation matrix
        t: np.ndarray of shape (3,)   - Translation vector
    """
    if source_points.shape != target_points.shape or source_points.shape[1] != 3:
        raise ValueError("Input point sets must have the same shape (Nx3).")


    set1 = np.array(source_points)
    set2 = np.array(target_points)
    
    assert set1.shape == set2.shape, "Input point sets must have the same shape"
    assert set1.shape[1] == 3, "Point sets must have 3D coordinates"
    
    # Compute centroids
    centroid1 = np.mean(set1, axis=0)
    centroid2 = np.mean(set2, axis=0)
    
    # Center the points
    centered1 = set1 - centroid1
    centered2 = set2 - centroid2
    
    # Compute the affine transformation matrix using least squares
    A, residuals, rank, s = np.linalg.lstsq(centered1, centered2, rcond=None)
    
    # Compute the translation vector
    t = centroid2 - np.dot(centroid1, A)
    t = t.reshape(3,1)
    # Convert to 4x4 matrix
    transformation_matrix = A.copy()
    transformation_matrix = np.hstack( [transformation_matrix, t ])
    transformation_matrix = np.vstack([transformation_matrix, np.array([0, 0, 0, 1])])

    return A, t, transformation_matrix

def compute_affine_transformation(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """
    Computes the affine transformation matrix that maps source_points to target_points in 3D.
    
    Parameters:
    source_points (numpy.ndarray): Nx3 array of source 3D points.
    target_points (numpy.ndarray): Nx3 array of target 3D points.
    
    Returns:
    numpy.ndarray: 4x4 affine transformation matrix.
    """
    if isinstance(source_points, np.ndarray) and source_points.shape[0] == 0:
        return  None

    if source_points.shape != target_points.shape or source_points.shape[1] != 3:
        print("Input point sets must have the same shape (Nx3).")
        return None
    
    # Append a column of ones to the source points (homogeneous coordinates)
    ones = np.ones((source_points.shape[0], 1))
    source_h = np.hstack([source_points, ones])
    
    # Solve for the affine transformation using least squares
    affine_matrix, _, _, _ = np.linalg.lstsq(source_h, target_points, rcond=None)
    
    # Convert to 4x4 matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :] = affine_matrix.T
    
    return transformation_matrix


def get_affine_transformation(moving_name, fixed_name='Allen', scaling_factor=1):
        """This fetches data from the DB and returns the data in micrometers
        Adjust accordingly
        """

        moving_all = list_coms(moving_name, scaling_factor=scaling_factor)
        fixed_all = list_coms(fixed_name, scaling_factor=scaling_factor)

        bad_keys = ('RtTg', 'AP')

        common_keys = list(moving_all.keys() & fixed_all.keys())
        good_keys = set(common_keys) - set(bad_keys)

        moving_src = np.array([moving_all[s] for s in good_keys])
        fixed_src = np.array([fixed_all[s] for s in good_keys])


        return compute_affine_transformation(moving_src, fixed_src)


def get_umeyama(source_points, target_points, scaling=False):
    """
    Get the umeyama transformation matrix between the Allen
    and animal brains.
    """

    if source_points.shape != target_points.shape or source_points.shape[1] != 3:
        raise ValueError("Input point sets must have the same shape (Nx3).")


    A, t = umeyama(source_points.T, target_points.T, with_scaling=scaling)
    transformation_matrix = np.hstack( [A, t ])
    transformation_matrix = np.vstack([transformation_matrix, np.array([0, 0, 0, 1])])

    return transformation_matrix

def resample_image(image, reference_image):
    """
    Resamples an image to match the reference image in size, spacing, and direction.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkLinear)  # Linear interpolation for resampling
    resampler.SetDefaultPixelValue(0)  # Fill with zero if needed
    return resampler.Execute(image)

def average_images(volumes, structure):
    images = [sitk.GetImageFromArray(img.astype(np.float32)) for img in volumes]
    reference_image = max(images, key=lambda img: np.prod(img.GetSize()))
    resampled_images = [resample_image(img, reference_image) for img in images]
    registered_images = [register_volume(img, reference_image, structure) for img in resampled_images if img != reference_image]
    avg_array = np.mean(registered_images, axis=0)
    return avg_array


def register_volume(movingImage, fixedImage, structure):

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)

    bspline_params = elastixImageFilter.GetDefaultParameterMap("bspline")
    rigid_params = elastixImageFilter.GetDefaultParameterMap("affine")
    rigid_params["AutomaticTransformInitialization"] = ["true"]
    rigid_params["AutomaticTransformInitializationMethod"] = ["GeometricalCenter"]
    rigid_params["FixedInternalImagePixelType"] = ["float"]
    rigid_params["MovingInternalImagePixelType"] = ["float"]
    rigid_params["FixedImageDimension"] = ["3"]
    rigid_params["MovingImageDimension"] = ["3"]
    rigid_params["UseDirectionCosines"] = ["false"]
    rigid_params["HowToCombineTransforms"] = ["Compose"]
    rigid_params["DefaultPixelValue"] = ["0"]
    rigid_params["WriteResultImage"] = ["false"]    
    rigid_params["WriteIterationInfo"] = ["false"]
    rigid_params["Resampler"] = ["DefaultResampler"]
    rigid_params["MaximumNumberOfIterations"] = ["150"] # 250 works ok

    elastixImageFilter.SetParameterMap(rigid_params)
    #elastixImageFilter.AddParameterMap(bspline_params)
    #elastixImageFilter.SetParameter("Registration", ["MultiResolutionRegistration"])
    #elastixImageFilter.SetParameter("Metric",  ["AdvancedImageToImageMetric", "CorrespondingPointsEuclideanDistanceMetric"])
    elastixImageFilter.SetLogToFile(False)
    elastixImageFilter.LogToConsoleOff()

    elastixImageFilter.SetParameter("WriteIterationInfo",["false"])
    elastixImageFilter.SetOutputDirectory('/tmp')
    try:
        resultImage = elastixImageFilter.Execute() 
    except Exception as e:
        print(f'{structure} in registration')
        print(e)
        sys.exit()
        #return sitk.GetArrayFromImage(movingImage)

    return sitk.GetArrayFromImage(resultImage)

def get_min_max_mean(coords):

    if not isinstance(coords, list) or len(coords) == 0:
        return None, None, None

    if len(coords) > 0 and isinstance(coords[0], list):
        new_coords = [coord for sublist in coords for coord in sublist]

    try:
        coords = [tuple(map(float, coord)) for coord in new_coords]
    except:
        coords = [tuple(coord) for coord in coords]
    
    x_vals, y_vals = zip(*coords)
    
    min_vals = (min(x_vals), min(y_vals))
    max_vals = (max(x_vals), max(y_vals))
    mean_vals = (
        sum(x_vals) / len(coords),
        sum(y_vals) / len(coords),
    )
    
    return min_vals, max_vals, mean_vals

def adjust_volume(volume, allen_id):
    """
    The commands below produce really nice STLs
    """
    upper = 150
    volume = gaussian(volume, 4.0)            
    volume[(volume > upper) ] = allen_id
    volume[(volume != allen_id)] = 0
    volume = volume.astype(np.uint32)
    return volume


def average_imagesV1(volumes, structure):
    images = [sitk.GetImageFromArray(img.astype(np.float32)) for img in volumes]
    reference_image_index, reference_image = max(enumerate(images), key=lambda img: np.prod(img[1].GetSize()))
    #max_index = images.index(reference_image_index)
    #del images[max_index]
    # Resample all images to the reference
    resampled_images = [resample_image(img, reference_image) for img in images]
    registered_images = [register_volume(img, reference_image, structure) for img in resampled_images if img != reference_image]
    # Convert images to numpy arrays and compute the average
    #registered_images = [sitk.GetArrayFromImage(img) for img in resampled_images]
    #registered_images = [sitk.GetArrayFromImage(img) for img in registered_images]
    avg_array = np.mean(registered_images, axis=0)
    # Convert back to SimpleITK image
    #avg_image = sitk.GetImageFromArray(avg_array)
    #avg_image.CopyInformation(reference_image)  # Copy metadata
    return avg_array
    #return sitk.GetArrayFromImage(avg_array)


def euler_to_rigid_transform(euler_transform, rotation_order='xyz', degrees=False):
    """
    Converts a 6-variable Euler transform to a 4x4 rigid transformation matrix.
    
    Parameters:
        euler_transform (list or np.ndarray): A list or array of 6 values.
                                              The first 3 are rotation (rx, ry, rz),
                                              the last 3 are translation (tx, ty, tz).
        rotation_order (str): Order of Euler rotations (default 'xyz').
        degrees (bool): Whether the input rotation angles are in degrees. Default is radians.

    Returns:
        np.ndarray: A 4x4 rigid transformation matrix.
    """
    assert len(euler_transform) == 6, "Euler transform must have 6 elements"

    rot_angles = euler_transform[:3]
    translation = euler_transform[3:]

    # Create rotation matrix
    rotation = R.from_euler(rotation_order, rot_angles, degrees=degrees)
    rot_matrix = rotation.as_matrix()

    # Construct 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = translation

    return transform
