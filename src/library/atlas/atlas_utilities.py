import numpy as np
from library.controller.sql_controller import SqlController
from library.utilities.utilities_process import M_UM_SCALE

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


def list_coms(animal):
    """
    Lists the COMs from the annotation session table. The data
    is stored in meters so you will want to convert it to micrometers
    and then by the resolution of the scan run.
    """
    sqlController = SqlController(animal)
    xy_resolution = sqlController.scan_run.resolution
    z_resolution = sqlController.scan_run.zresolution

    coms = {}
    annotator_id = 1 # Hardcoded to edward
    com_dictionaries = sqlController.get_com_dictionary(prep_id=animal, annotator_id=annotator_id)
    for k, v in com_dictionaries.items():
        x = round(v[0] * M_UM_SCALE / xy_resolution, 2)
        y = round(v[1] * M_UM_SCALE / xy_resolution, 2)
        z = round(v[2] * M_UM_SCALE / z_resolution, 2)
        #x = round(v[0] * M_UM_SCALE / 10, 2)
        #y = round(v[1] * M_UM_SCALE / 10, 2)
        #z = round(v[2] * M_UM_SCALE / 10, 2)
        coms[k] = (x,y,z)

    return coms


def compute_affine_transformation_centroid(set1, set2):
    """
    Computes the affine transformation (scale, shear, rotation, translation) between two sets of 3D points.
    
    Parameters:
        set1: np.ndarray of shape (N, 3) - Source set of 3D points
        set2: np.ndarray of shape (N, 3) - Target set of 3D points
    
    Returns:
        A: np.ndarray of shape (3, 3) - Linear transformation matrix
        t: np.ndarray of shape (3,)   - Translation vector
    """
    set1 = np.array(set1)
    set2 = np.array(set2)
    
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
    #transformation_matrix[:3, :] = A
    #transformation_matrix[... , 4] = t
    transformation_matrix = np.hstack( [transformation_matrix, t ])
    transformation_matrix = np.vstack([transformation_matrix, np.array([0, 0, 0, 1])])

    
    return A, t, transformation_matrix

def compute_affine_transformation(source_points, target_points):
    """
    Computes the affine transformation matrix that maps source_points to target_points in 3D.
    
    Parameters:
    source_points (numpy.ndarray): Nx3 array of source 3D points.
    target_points (numpy.ndarray): Nx3 array of target 3D points.
    
    Returns:
    numpy.ndarray: 4x4 affine transformation matrix.
    """

    if source_points.shape != target_points.shape or source_points.shape[1] != 3:
        raise ValueError("Input point sets must have the same shape (Nx3).")
    
    # Append a column of ones to the source points (homogeneous coordinates)
    ones = np.ones((source_points.shape[0], 1))
    source_h = np.hstack([source_points, ones])
    
    # Solve for the affine transformation using least squares
    affine_matrix, _, _, _ = np.linalg.lstsq(source_h, target_points, rcond=None)
    
    # Convert to 4x4 matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :] = affine_matrix.T
    
    return transformation_matrix

def get_affine_transformation(animal='Atlas'):
    """
    Get the affine transformation matrix between the Allen
    and animal brains.
    """

    atlas_structures = list_coms(animal)
    allen_structures = list_coms('Allen')

    common_keys = atlas_structures.keys() & allen_structures.keys()
    atlas_src = np.array([atlas_structures[s] for s in common_keys])
    allen_src = np.array([allen_structures[s] for s in common_keys])

    A, t, transformation = compute_affine_transformation_centroid(atlas_src, allen_src)

    return transformation   
