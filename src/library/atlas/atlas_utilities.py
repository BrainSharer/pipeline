from collections import defaultdict
import os
from pathlib import Path
import sys
import numpy as np
import SimpleITK as sitk
from scipy.spatial import Delaunay

from scipy.spatial.transform import Rotation as R
from skimage.filters import gaussian
from scipy.ndimage import affine_transform, binary_erosion, zoom, center_of_mass, shift

from scipy.interpolate import splprep, splev

import shapely
from shapely.geometry import LineString, Polygon, MultiPoint
from shapely.ops import triangulate,  unary_union, polygonize
from skimage.draw import polygon as sk_polygon

from library.controller.sql_controller import SqlController
from library.registration.algorithm import umeyama
from library.utilities.utilities_process import M_UM_SCALE



ORIGINAL_ATLAS = 'AtlasV7'
NEW_ATLAS = 'AtlasV8'
RESOLUTION = 0.452
ALLEN_UM = 10

def order_points_concave_hull(points, alpha=1.0):
    """Return points ordered along a concave hull using alpha shape."""
    pts = np.array(points)
    if len(pts) < 4:
        return pts
    
    tri = Delaunay(pts)
    edges = set()
    for ia, ib, ic in tri.simplices:
        pa, pb, pc = pts[ia], pts[ib], pts[ic]
        a = np.linalg.norm(pa - pb)
        b = np.linalg.norm(pb - pc)
        c = np.linalg.norm(pc - pa)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < 1.0 / alpha:
            edges.add(tuple(sorted((ia, ib))))
            edges.add(tuple(sorted((ib, ic))))
            edges.add(tuple(sorted((ic, ia))))

    m = MultiPoint([tuple(p) for p in pts])
    edge_lines = [LineString([pts[i], pts[j]]) for i, j in edges]
    m = unary_union(edge_lines)
    polygons = list(polygonize(m))
    if not polygons:
        return pts
    boundary = polygons[0].exterior.coords[:-1]  # drop duplicate last point
    return np.array(boundary)

def interpolate_points(points, new_len):
    points = np.array(points)
    pu = points.astype(int)
    indexes = np.unique(pu, axis=0, return_index=True)[1]
    points = np.array([points[index] for index in sorted(indexes)])
    addme = points[0].reshape(1, 2)
    if points.shape[0] < 3:
        return points
    points = np.concatenate((points, addme), axis=0)
    tck, u = splprep(points.T, u=None, s=3, per=1)
    u_new = np.linspace(u.min(), u.max(), new_len)
    x_array, y_array = splev(u_new, tck, der=0)
    arr_2d = np.concatenate([x_array[:, None], y_array[:, None]], axis=1)
    return list(map(tuple, arr_2d.astype(np.int32)))


def alpha_shape_polygon(points_xy, alpha=1.5):
    """
    Build a (multi)polygon concave hull from unordered 2D points using an alpha-shape.
    Returns a list of shapely Polygons. Requires shapely.
    """
    if len(points_xy) < 4:
        return []  # too few to form area

    mp = MultiPoint(points_xy)
    # Start with convex hull if alpha very small or points small:
    hull = mp.convex_hull
    if alpha <= 0:
        return [hull] if isinstance(hull, Polygon) else []

    # Triangulate and keep triangles with circumradius small enough
    tris = triangulate(mp)
    keep = []
    for tri in tris:
        # triangle as 3 points
        x, y = tri.exterior.coords.xy
        pts = np.column_stack([x[:-1], y[:-1]])  # 3 vertices
        # edge lengths
        a = np.linalg.norm(pts[0] - pts[1])
        b = np.linalg.norm(pts[1] - pts[2])
        c = np.linalg.norm(pts[2] - pts[0])
        s = 0.5 * (a + b + c)
        area = max(s * (s - a) * (s - b) * (s - c), 0.0) ** 0.5
        if area == 0:
            continue
        R = (a * b * c) / (4.0 * area)  # circumradius
        if R < (1.0 / alpha):
            keep.append(tri)

    if not keep:
        # fallback: convex hull
        return [hull] if isinstance(hull, Polygon) else []

    # Union kept triangles -> concave hull
    concave = shapely.union_all(keep) if hasattr(shapely, "union_all") else keep[0].union(keep[1:])
    # Normalize to list of Polygons
    if isinstance(concave, Polygon):
        return [concave]
    elif hasattr(concave, "geoms"):
        return [g for g in concave.geoms if isinstance(g, Polygon)]
    else:
        return []


def nearest_neighbor_polygon(points_xy):
    """
    Very simple NN chain to order unordered boundary points into a closed polygon.
    Works surprisingly well for dense boundaries but is heuristic.
    Returns a (N,2) array of ordered vertices.
    """
    pts = np.asarray(points_xy, dtype=float)
    if len(pts) < 3:
        return None
    # start at the leftmost (then lowest y) point
    start_idx = np.lexsort((pts[:,1], pts[:,0]))[0]
    used = np.zeros(len(pts), dtype=bool)
    order = [start_idx]
    used[start_idx] = True
    for _ in range(len(pts) - 1):
        last = pts[order[-1]]
        # squared distances to unused
        diffs = pts[~used] - last
        d2 = np.einsum('ij,ij->i', diffs, diffs)
        next_rel = np.argmin(d2)
        # map back to absolute index
        cand_idx = np.flatnonzero(~used)[next_rel]
        order.append(cand_idx)
        used[cand_idx] = True
    ordered = pts[order]
    return ordered


def rasterize_polygon_to_mask(poly, H, W):
    """
    Fill polygon (with holes) into a 2D mask of shape (H, W).
    poly: shapely Polygon
    """
    mask = np.zeros((H, W), dtype=bool)
    # exterior
    ex = np.array(poly.exterior.coords, dtype=float)
    rr, cc = sk_polygon(ex[:,1], ex[:,0], shape=(H, W))
    mask[rr, cc] = True
    # holes
    for interior in poly.interiors:
        hole = np.array(interior.coords, dtype=float)
        rr_h, cc_h = sk_polygon(hole[:,1], hole[:,0], shape=(H, W))
        mask[rr_h, cc_h] = False
    return mask


def create_subvolume_from_boundary_vertices(shape_zyx,
                                            coords_xyz,
                                            dtype=np.uint8,
                                            fill_value=255,
                                            alpha=1.5,
                                            voxel_spacing_xyz=None):
    """
    Build a 3D volume with a filled subvolume defined by boundary vertices (x,y,z).
    Non-convex shapes supported (per-slice concave hull). Coordinates can be floats.
    
    Parameters
    ----------
    shape_zyx : tuple (Z, Y, X)
        Output volume shape.
    coords_xyz : array-like of (x, y, z)
        Unordered boundary points. z should index slices in [0, Z-1].
    dtype : numpy dtype
        Output dtype.
    fill_value : int
        Value assigned inside the filled subvolume; 0 elsewhere.
    alpha : float
        Alpha-shape parameter (larger -> more detail; smaller -> smoother/closer to convex).
        Only used if shapely is available.
    voxel_spacing_xyz : (sx, sy, sz) or None
        If provided, points are rescaled so alpha operates in physical units per slice.
        Commonly not needed; useful when x,y are anisotropic.

    Returns
    -------
    vol : np.ndarray (Z, Y, X)
        Binary-ish mask with fill_value inside subvolume and 0 outside.
    """
    Z, Y, X = map(int, shape_zyx)
    vol = np.zeros((Z, Y, X), dtype=dtype)

    coords = np.asarray(coords_xyz, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords_xyz must be an array of shape (N, 3) with (x, y, z).")

    # Optionally rescale to physical units for alpha-shape (x,y only matter per slice)
    if voxel_spacing_xyz is not None:
        sx, sy, sz = voxel_spacing_xyz
    else:
        sx = sy = sz = 1.0

    # Group by slice index (round to nearest int robustly)
    by_slice = defaultdict(list)
    for x, y, z in coords:
        zi = int(round(z / sz))
        if 0 <= zi < Z:
            by_slice[zi].append((x / sx, y / sy))  # store rescaled coords

    for zi, pts in by_slice.items():
        if len(pts) < 3:
            continue

        H, W = Y, X  # mask height, width

        # Concave hull via alpha-shape; may produce MultiPolygons.
        polys = alpha_shape_polygon(pts, alpha=alpha)
        if not polys:
            # Fallback to NN ordering if alpha-shape fails to make area
            ordered = nearest_neighbor_polygon(pts)
            if ordered is None:
                continue
            rr, cc = sk_polygon(ordered[:,1], ordered[:,0], shape=(H, W))
            slice_mask = np.zeros((H, W), dtype=bool)
            slice_mask[rr, cc] = True
        else:
            slice_mask = np.zeros((H, W), dtype=bool)
            for poly in polys:
                if poly.is_empty or not isinstance(poly, Polygon):
                    continue
                slice_mask |= rasterize_polygon_to_mask(poly, H, W)

        vol[zi, slice_mask] = fill_value

    return vol



def load_transformation(animal: str, xy_um: float, z_um: float) -> sitk.Transform:
    reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
    transform = f'{animal}_Allen_{z_um}x{xy_um}x{xy_um}um.tfm'
    transform_path = os.path.join(reg_path, animal, transform)
    if not os.path.exists(transform_path):
        print(f"Transformation file not found: {transform_path}")
        return None
    else:
        print(f"Loading transformation file: {transform_path}")

    return sitk.ReadTransform(transform_path)

def get_physical_center(image):
    size = np.array(image.GetSize())
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    center_index = size / 2.0
    center_physical = origin + direction @ (spacing * center_index)
    return center_physical



def create_convex_hull(xyz_array: np.ndarray, volume: np.ndarray, shape: tuple) -> np.ndarray:
    hull = Delaunay(xyz_array)
    # Create a grid of all voxel indices
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
        indexing='ij'
    )
    grid_points = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
    # Find which voxels lie inside the convex hull
    mask = hull.find_simplex(grid_points) >= 0
    # Fill in those voxels with 255
    volume_flat = volume.ravel()
    volume_flat[mask] = 255
    volume = volume_flat.reshape(shape)
    return volume



def affine_transform_point(point: list, matrix: np.ndarray) -> np.ndarray:
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

def affine_transform_volume(volume: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply an affine transformation to a 3D volume.
    order = 0 for nearest neighbor interpolation, for binary masks
    """
    if matrix.shape != (4, 4):
        raise ValueError("Matrix must be a 4x4 numpy array")
    #translation = (matrix[..., 3][0:3])
    translation = (0,0,0)
    matrix = matrix[:3, :3]
    transformed = affine_transform(volume.astype(np.float32), matrix, offset=translation, order=0, mode='constant', cval=0)
    return (transformed > 0.05).astype(np.uint8)

def affine_transform_points(polygons: defaultdict, matrix: np.ndarray) -> defaultdict:
    """
    Applies an affine transformation to a dictionary of 3D points in um
    the points will be in a defaultdict like points[section] = (x, y)
    where section = z

    Parameters:
        points (dictionary of lists): List of (x, y) coordinates per z.
        matrix (numpy.ndarray): A 4x4 affine transformation matrix.

    Returns:
        transformed_points (list of tuples): Transformed (x, y, z) coordinates.
    """
    if matrix.shape != (4, 4):
        raise ValueError("Transformation matrix must be 4x4.")
    
    affine_matrix = matrix[:3, :3]
    transformed = defaultdict(list)
    for z, points in polygons.items():
        for x, y in points:
            vec = np.array([x, y, z])
            x_new, y_new, z_new = affine_matrix @ vec
            z_new = int(round(z_new))
            transformed[z_new].append((x_new, y_new))

    return transformed


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

    return coms

def list_raw_coms(animal, scaling_factor=1):
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
        coms[k] = v

    return coms

def get_origins(animal, scale=1):
    """
    Fetches the origins from disk. The data is already scaled to 10um.
    """
    
    origins = {}
    dirpath = f'/net/birdstore/Active_Atlas_Data/data_root/atlas_data/{animal}/origin'
    if not os.path.exists(dirpath):
        return origins
    files = sorted(os.listdir(dirpath))
    for file in files:
        structure = Path(file).stem
        filepath = os.path.join(dirpath, file)
        origin = np.loadtxt(filepath)
        origins[structure] = origin / scale
    return origins


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
    A, res, rank, s = np.linalg.lstsq(source_h, target_points, rcond=None)

    # A is 4x3, so we transpose it and add a row for homogeneous coordinates
    affine_matrix = np.vstack([A.T, [0, 0, 0, 1]])  # (4 x 4)
    
    return affine_matrix


def get_affine_transformation(moving_name, fixed_name='Allen', scaling_factor=1):
        """This fetches data from the DB and returns the data in micrometers
        Adjust accordingly
        """

        moving_all = fetch_coms(moving_name, scaling_factor=scaling_factor)
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

def center_images_to_largest_volume(images):
    """
    Centers a list of 3D SimpleITK images using the largest-volume image as the reference.

    Parameters:
        images (List[sitk.Image]): List of 3D SimpleITK Image objects.

    Returns:
        List[sitk.Image]: List of centered images (same order as input).
    """
    if not images:
        raise ValueError("No images provided.")

    # Compute volumes and find reference image
    volumes = [img.GetSize()[0] * img.GetSize()[1] * img.GetSize()[2] * 
               img.GetSpacing()[0] * img.GetSpacing()[1] * img.GetSpacing()[2] for img in images]
    reference_index = volumes.index(max(volumes))
    reference_image = images[reference_index]

    centered_images = []
    for i, img in enumerate(images):
        if i == reference_index:
            centered_images.append(img)
            continue

        # Calculate center transform
        try:
            transform = sitk.CenteredTransformInitializer(reference_image, img,
                                                        sitk.Euler3DTransform(), 
                                                        sitk.CenteredTransformInitializerFilter.GEOMETRY)
        except Exception as e:
            print(f"Error initializing transform for image {i}: {e}")
            exit(1)

        # Resample image
        resampled = sitk.Resample(img,
                                  reference_image,
                                  transform,
                                  sitk.sitkNearestNeighbor,
                                  0.0,
                                  img.GetPixelID())

        centered_images.append(resampled)

    return centered_images


def resample_image(image, reference_image):
    """
    Resamples an image to match the reference image in size, spacing, and direction.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkLinear)  # Linear interpolation for resampling
    resampler.SetDefaultPixelValue(0)  # Fill with zero if needed
    resultImage = resampler.Execute(image)
    #return sitk.GetArrayFromImage(resultImage)
    return resultImage

def resize_image(image, new_size):
    """
    Resamples an image to match the reference image in size, spacing, and direction.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)  # Linear interpolation for resampling
    resampler.SetDefaultPixelValue(0)  # Fill with zero if needed
    resultImage = resampler.Execute(image)
    #return sitk.GetArrayFromImage(resultImage)
    return resultImage


def get_image_center(image):
    """Compute the physical center of a SimpleITK image."""
    size = np.array(image.GetSize())
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    
    center_index = (size - 1) / 2.0
    center_physical = origin + direction @ (center_index * spacing)
    
    return center_physical

def compute_translation_transform(from_center, to_center):
    """Create a translation transform that moves from_center to to_center."""
    translation_vector = to_center - from_center
    return sitk.TranslationTransform(3, translation_vector)

def align_images_to_common_center(images):
    """
    Given a list of SimpleITK 3D images, return a list of translated images
    aligned to their common center.
    """
    centers = [get_image_center(img) for img in images]
    common_center = np.mean(centers, axis=0)
    
    aligned_images = []

    for img, center in zip(images, centers):
        transform = compute_translation_transform(center, common_center)
        resampled = sitk.Resample(
            img,
            img.GetSize(),
            transform,
            sitk.sitkLinear,
            img.GetOrigin(),
            img.GetSpacing(),
            img.GetDirection(),
            0.0,  # default pixel value
            img.GetPixelID()
        )
        aligned_images.append(resampled)
    
    return aligned_images


def create_average_binary_maskXXX(volumes):
    if len(volumes) == 1:
        avg_volume = volumes[0]
    else:
        images = [sitk.GetImageFromArray(img.astype(np.uint8)) for img in volumes]
        resampled_images = center_images_to_largest_volume(images)
        avg_volume = np.mean([sitk.GetArrayFromImage(vol) for vol in resampled_images], axis=0)
        #avg_volume = adjust_volume(avg_volume, allen_id=255)
    return avg_volume.astype(np.uint8)


def create_average_nii(images, structure=None):
    if len(images) == 0:
        raise ValueError("No volumes provided for averaging.")
    elif len(images) == 1:
        volume = images[0]
    else:
        pixel_type = sitk.sitkUInt8
        reference_image_index, reference_image = max(enumerate(images), key=lambda img: np.prod(img[1].GetSize()))
        # Accumulate pixel values
        #resampled_images = [resample_image(img, reference_image) for img in images]
        origins = []
        for image in images:
            origin = image.GetOrigin()
            origins.append(origin)
        mean_origin = np.mean(origins, axis=0)
        resampled_images = center_images_to_largest_volume(images)
        registered_images = [register_volume(img, reference_image, structure) for img in resampled_images if img != reference_image]
        volume = np.mean(registered_images, axis=0)
        volume = sitk.GetImageFromArray(volume.astype(np.uint8))
        volume = sitk.Cast(volume, pixel_type)
        volume = sitk.BinaryThreshold(volume, lowerThreshold=1, upperThreshold=255, insideValue=255, outsideValue=0)
        #num_images = len(images)
        #volume = sum_image / num_images
        volume.SetSpacing(reference_image.GetSpacing())
        volume.SetOrigin(mean_origin)
        volume.SetDirection(reference_image.GetDirection())

    return volume

def create_average_binary_mask(volumes, structure=None):
    if len(volumes) == 0:
        raise ValueError("No volumes provided for averaging.")
    elif len(volumes) == 1:
        volume = volumes[0]
    else:
        images = [sitk.GetImageFromArray(img.astype(np.float32)) for img in volumes]
        reference_image_index, reference_image = max(enumerate(images), key=lambda img: np.prod(img[1].GetSize()))
        # Resample all images to the reference
        resampled_images = [resample_image(img, reference_image) for img in images]
        registered_images = [register_volume(img, reference_image, structure) for img in resampled_images if img != reference_image]
        volume = np.mean(registered_images, axis=0)
        upper = 100
        volume = gaussian(volume, 2.0)
        volume[(volume > upper) ] = 255
        volume[(volume != 255)] = 0
        dtype = np.uint8
        volume = volume.astype(dtype)
        
    return volume

def rigid_registration_get_matrix_translation(fixed_image, moving_image):
    """
    Perform rigid registration between two 3D images and extract the rotation matrix and translation vector.

    Args:
        fixed_image (sitk.Image): The reference image.
        moving_image (sitk.Image): The image to be transformed.

    Returns:
        rotation_matrix (np.ndarray): A 3x3 rotation matrix.
        translation_vector (np.ndarray): A 3x1 translation vector.
    """
    # Initialize the registration method
    registration_method = sitk.ImageRegistrationMethod()
    
    # Use Mutual Information for multi-modal or Mattes MI for mono-modal
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    
    # Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-8
    )

    # Set initial transform (rigid)
    initial_transform = sitk.VersorRigid3DTransform()
    initial_transform.SetCenter(fixed_image.TransformContinuousIndexToPhysicalPoint(np.array(fixed_image.GetSize()) / 2.0))
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Execute registration
    final_transform = registration_method.Execute(fixed_image, moving_image)


    # Extract matrix and translation
    matrix = np.array(final_transform.GetMatrix()).reshape((3, 3))
    translation = np.array(final_transform.GetTranslation()).reshape((3, 1))
    print(f"Final transform matrix:\n{matrix}\nTranslation vector:\n{translation}")

    return matrix, translation



def register_volume(movingImage, fixedImage, structure=None):

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)

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
    rigid_params["MaximumNumberOfIterations"] = ["250"] # 250 works ok
    rigid_params["WriteIterationInfo"] = ["false"]

    elastixImageFilter.SetParameterMap(rigid_params)
    elastixImageFilter.SetLogToFile(False)
    elastixImageFilter.LogToConsoleOff()

    try:
        resultImage = elastixImageFilter.Execute() 
    except Exception as e:
        print(f'Exception in registration with {structure=}')
        print(e)
        return sitk.GetArrayFromImage(movingImage)

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
    upper = np.quantile(volume, 0.85)
    try:
        volume = gaussian(volume, 2.0)
    except Exception as e:
        print(f"Gaussian smoothing failed: {e}")
        pass
    volume[(volume > upper) ] = allen_id
    volume[(volume != allen_id)] = 0
    if allen_id == 255:
        dtype = np.uint8
    else:
        dtype = np.uint32
    volume = volume.astype(dtype)
    return volume



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

def itk_rigid_euler():
    params = [0.066036, 0.006184, 0.476529, -235.788051, -169.603207, 36.877408]
    theta_x = params[0]
    theta_y = params[1]
    theta_z = params[2]
    translation = np.array(params[3:6])
    center = np.array([0,0,0,])
    rigid_euler = sitk.Euler3DTransform(center, theta_x, theta_y, theta_z, translation)
    R = np.asarray(rigid_euler.GetMatrix()).reshape(3,3)
    t = np.asarray(rigid_euler.GetTranslation())
    return R, t

import numpy as np
import cv2

def get_evenly_spaced_vertices_from_volume(mask, num_points=20):
    """
    This function was made entirely from ChatGTP
    Given a binary mask, extract the outer contour and return evenly spaced vertices along the edge.

    Parameters:
    - mask: 2D numpy array (binary mask)
    - num_points: Number of evenly spaced points to return

    Returns:
    - List of (x, y) coordinates of vertices
    """
    # Ensure mask is uint8
    mask = mask.astype(np.uint8)

    # Find contours (external only)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    # Choose the largest contour (in case there are multiple)
    contour = max(contours, key=cv2.contourArea).squeeze()

    # Calculate arc length (perimeter)
    #arc_length = cv2.arcLength(contour, True)
    """
    try:
        arc_length = cv2.arcLength(contour, True)
    except Exception as e:
        print(f"Error calculating arc length: {e}")
        return []
    """
    # Calculate the cumulative arc lengths
    distances = [0]
    for i in range(1, len(contour)):
        d = np.linalg.norm(contour[i] - contour[i - 1])
        distances.append(distances[-1] + d)
    distances = np.array(distances)

    # Sample points at regular intervals
    desired_distances = np.linspace(0, distances[-1], num_points, endpoint=False)
    vertices = []
    j = 0
    for d in desired_distances:
        while j < len(distances) - 1 and distances[j+1] < d:
            j += 1
        # Linear interpolation between points j and j+1
        t = (d - distances[j]) / (distances[j+1] - distances[j])
        pt = (1 - t) * contour[j] + t * contour[j + 1]
        try:
            vertices.append(tuple(pt.astype(int)))
        except TypeError as e:
            continue

    return vertices

def get_edge_coordinates(array):
    """
    Returns the coordinates of non-zero edge pixels in a 2D binary array.
    """
    # Ensure the input is a binary array
    binary = array > 0

    # Erode the binary mask
    eroded = binary_erosion(binary)

    # Subtract eroded version from original to get edges
    edges = binary & ~eroded

    # Get coordinates of edge pixels
    edge_coords = np.column_stack(np.nonzero(edges))

    return edge_coords

def get_evenly_spaced_vertices_from_slice(mask):
    """
    Given a binary mask, extract the outer contour and return evenly spaced vertices along the edge.

    Parameters:
    - mask: 2D numpy array (binary mask)
    - num_points: Number of evenly spaced points to return

    Returns:
    - List of (x, y) coordinates of vertices
    """
    # Ensure mask is uint8
    mask = mask.astype(np.uint8)

    # Find contours (external only)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    # Choose the largest contour (in case there are multiple)
    contour = max(contours, key=cv2.contourArea).squeeze()

    # Calculate arc length (perimeter)
    arc_length = cv2.arcLength(contour, True)
    if arc_length < 10:
        num_points = 4
    elif arc_length >= 10 and arc_length < 100:
        num_points = 10
    elif arc_length >= 100 and arc_length < 300:
        num_points = 20
    else:
        num_points = 30

    #print(f"Arc length: {arc_length} num_points: {num_points}")

    # Calculate the cumulative arc lengths
    distances = [0]
    for i in range(1, len(contour)):
        d = np.linalg.norm(contour[i] - contour[i - 1])
        distances.append(distances[-1] + d)
    distances = np.array(distances)

    # Sample points at regular intervals
    desired_distances = np.linspace(0, distances[-1], num_points, endpoint=False)
    vertices = []
    j = 0
    for d in desired_distances:
        while j < len(distances) - 1 and distances[j+1] < d:
            j += 1
        # Linear interpolation between points j and j+1
        t = (d - distances[j]) / (distances[j+1] - distances[j])
        pt = (1 - t) * contour[j] + t * contour[j + 1]
        vertices.append(tuple(pt.astype(int)))

    return vertices


def get_evenly_spaced_vertices(vertices: list, num_points=20) -> np.ndarray:
    """
    Returns a specified number of evenly spaced points along the perimeter of a polygon.

    Args:
        vertices (list of tuple): List of (x, y) tuples representing the polygon vertices.
        num_points (int): The number of evenly spaced points to return.

    Returns:
        list of tuple: List of (x, y) tuples representing the evenly spaced points.
    """
    # Close the polygon if it's not already closed


    if not isinstance(vertices, list):
        if isinstance(vertices, np.ndarray):
            #non_zero_coords = np.argwhere(vertices != 0)
            non_zero_coords = get_edge_coordinates(vertices)
            vertices = [tuple(row) for row in non_zero_coords]
            #return vertices


    if not isinstance(vertices[0], tuple) and len(vertices[0]) != 2:
        print("Vertices[0] should be a list of tuples.")
        print(type(vertices[0]), len(vertices[0]))
        exit(1)
    
    if vertices[0] != vertices[-1]:
        vertices.append(vertices[0])

    # Calculate distances between consecutive vertices
    distances = [np.linalg.norm(np.subtract(vertices[i+1], vertices[i])) for i in range(len(vertices)-1)]
    perimeter = sum(distances)

    # Total length between each evenly spaced point
    try:
        step = perimeter / num_points
    except TypeError as te:
        print(f"Error in calculating step size: {te}")
        print(f"perimeter = {perimeter}")
        print(f"num_points = {num_points}")
        exit(1)

    # Generate points
    result = []
    current_distance = 0
    i = 0
    while len(result) < num_points:
        start = np.array(vertices[i])
        end = np.array(vertices[i+1])
        segment_length = distances[i]

        while current_distance + segment_length >= len(result) * step:
            t = ((len(result) * step) - current_distance) / segment_length
            point = start + t * (end - start)
            result.append(tuple(point))

            if len(result) == num_points:
                break

        current_distance += segment_length
        i += 1
        if i >= len(distances):  # Safety break in case of rounding issues
            break

    return result

"""Below are some functions that might come in handy later"""
def filter_top_n_values(volume: np.ndarray, n: int, set_value: int = 1) -> np.ndarray:
    """
    Get the `n` most common unique values from a numpy volume.
    Sets those values to `set_value` and the rest to 0.

    Parameters:
        volume (np.ndarray): Input 3D volume.
        n (int): Number of most common unique values to retain.
        set_value (int, optional): The value to assign to the most common values. Defaults to 1.

    Returns:
        np.ndarray: Transformed volume.
    """
    from collections import Counter    
    # Flatten the volume and count occurrences of unique values
    values, counts = np.unique(volume[volume != 0], return_counts=True)
    
    # Get the top `n` most common values
    top_n_values = [val for val, _ in Counter(dict(zip(values, counts))).most_common(n)]
    print(f'top {n} {top_n_values=}')
    
    # Create a mask where only top N values are retained
    mask = np.isin(volume, top_n_values)
    
    # Set the selected values to `set_value` and the rest to 0
    result = np.where(mask, set_value, 0)
    
    return result


def center_3d_volume(volume: np.ndarray) -> np.ndarray:
    """
    Centers a 3D volume by shifting its center of mass to the geometric center.

    Parameters:
    volume (np.ndarray): A 3D numpy array representing the volume.

    Returns:
    np.ndarray: The centered 3D volume.
    """

    if volume.ndim != 3:
        raise ValueError("Input volume must be a 3D numpy array")
    
    # Compute the center of mass
    com = np.array(center_of_mass(volume))
    
    # Compute the geometric center
    shape = np.array(volume.shape)
    geometric_center = (shape - 1) / 2
    
    # Compute the shift required
    shift_values = geometric_center - com
    
    # Apply shift
    centered_volume = shift(volume, shift_values, mode='constant', cval=0)
    
    return centered_volume

def crop_nonzero_3d(volume):
    """
    Crops a 3D volume to remove all-zero regions.
    
    Parameters:
        volume (numpy.ndarray): A 3D NumPy array.
        
    Returns:
        numpy.ndarray: The cropped 3D volume.
    """
    if volume.ndim != 3:
        raise ValueError("Input volume must be a 3D NumPy array")
    
    # Find nonzero elements
    nonzero_coords = np.argwhere(volume)
    
    # Get bounding box of nonzero elements
    min_coords = nonzero_coords.min(axis=0)
    max_coords = nonzero_coords.max(axis=0) + 1  # Add 1 to include the max index
    
    # Crop the volume
    cropped_volume = volume[min_coords[0]:max_coords[0],
                            min_coords[1]:max_coords[1],
                            min_coords[2]:max_coords[2]]
    
    return cropped_volume

"""Tests using PCA to get an average volume"""
def mask_pca_axes(mask):
    """
    Compute centroid and principal axes (3x3 matrix whose columns are eigenvectors)
    from non-zero voxels of a binary mask.
    Returns centroid (3,), axes (3,3) (columns are principal axes), and 'size_metric' (mean std along axes).
    """
    coords = np.column_stack(np.nonzero(mask))  # shape (N, 3)
    if coords.shape[0] == 0:
        raise ValueError("Mask is empty.")
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
    # Reorder to descending
    order = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, order]
    eigvals = eigvals[order]
    # Ensure right-handed coordinate system (determinant = +1)
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] *= -1
    # Size metric: mean of std along principal axes
    proj = centered.dot(eigvecs)  # coords in PCA frame
    stds = proj.std(axis=0)
    size_metric = stds.mean()
    return centroid, eigvecs, size_metric


def average_rotation_from_axes_list(axes_list):
    """
    Given a list of axes matrices (3x3, columns are eigenvectors), compute average rotation.
    Uses quaternion averaging (make sure quaternions are consistently oriented).
    Returns a 3x3 rotation matrix.
    """
    rots = [R.from_matrix(a) for a in axes_list]
    quats = np.array([r.as_quat() for r in rots])  # xyzw order (scipy uses x,y,z,w)
    # Ensure hemisphere alignment: flip quaternions so scalar parts have same sign (or dot with first > 0)
    base = quats[0]
    for i in range(1, len(quats)):
        if np.dot(base, quats[i]) < 0:
            quats[i] = -quats[i]
    mean_quat = quats.mean(axis=0)
    mean_quat /= np.linalg.norm(mean_quat)
    mean_rot = R.from_quat(mean_quat).as_matrix()
    return mean_rot


def transform_mask_affine(mask, A, center_in, center_out, out_shape=None, order=1):
    """
    Apply affine transform (3x3 matrix A) and translation to resample mask into an output grid.
    We want: x_in = inv(A) @ (x_out - center_out) + center_in
    So for scipy.ndimage.affine_transform we provide matrix = inv(A) and offset = center_in - inv(A) @ center_out
    mask: input 3D array
    A: 3x3 transform applied to coordinates relative to centers (A = scale * rotation)
    center_in: center in input index coords (3,)
    center_out: desired center in output grid (3,)
    out_shape: shape of output volume (if None, use mask.shape)
    order: interpolation order (0 nearest, 1 trilinear)
    """
    if out_shape is None:
        out_shape = mask.shape
    A = np.asarray(A)
    invA = np.linalg.inv(A)
    # scipy affine_transform expects matrix mapping output coords to input coords:
    matrix = invA
    offset = center_in - invA.dot(center_out)
    # Use affine_transform on float array
    res = affine_transform(mask.astype(np.float32),
                                   matrix=matrix,
                                   offset=offset,
                                   output_shape=out_shape,
                                   order=order, mode='constant', cval=0.0)
    # Threshold back to binary (0.5)
    return (res >= 0.5).astype(np.uint8)


def align_masks_numpy(masks):
    """
    masks: list of 3D numpy arrays (binary)
    returns aligned_masks: list of aligned 3D numpy arrays (same shape)
    """
    n = len(masks)
    if n == 0:
        return []

    # 1) Compute PCA axes and centroids/sizes for each mask
    props = []
    for i, m in enumerate(masks):
        centroid, axes, size_metric = mask_pca_axes(m)
        props.append({'centroid': centroid, 'axes': axes, 'size': size_metric})

    axes_list = [p['axes'] for p in props]
    mean_axes = average_rotation_from_axes_list(axes_list)  # 3x3

    # 2) Compute average size (we will scale isotropically)
    sizes = np.array([p['size'] for p in props])
    avg_size = sizes.mean()

    # 3) Prepare output grid parameters (use same shape as inputs)
    buffer = 0
    mx = max(s.shape[0] for s in masks) + buffer
    my = max(s.shape[1] for s in masks) + buffer
    mz = max(s.shape[2] for s in masks) + buffer
    
    out_shape = (mx,my,mz)
    center_out = np.array(out_shape) / 2.0

    aligned = []
    for i, m in enumerate(masks):
        centroid = props[i]['centroid']
        axes = props[i]['axes']
        size_i = props[i]['size']
        # Rotation to map mask's axes into mean_axes:
        # R_needed = mean_axes @ axes.T
        R_needed = mean_axes.dot(axes.T)
        # scale factor (isotropic)
        if size_i == 0:
            scale = 1.0
        else:
            scale = (avg_size / size_i)
        #scale = 1
        A = scale * R_needed  # combined affine (applied about centers)
        # transform
        res = transform_mask_affine(m, A=A,
                                    center_in=centroid,
                                    center_out=center_out,
                                    out_shape=out_shape,
                                    order=1)
        aligned.append(res)
    return aligned


def numpy_to_sitk(arr):
    """Convert numpy 3D array to SimpleITK image."""
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing((1.0, 1.0, 1.0))
    return img

def sitk_to_numpy(img):
    """Convert SimpleITK image back to numpy."""
    return sitk.GetArrayFromImage(img)

def rigid_register_elastix(fixed, moving):
    """Perform rigid registration using mutual information."""
    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(fixed)
    elastix.SetMovingImage(moving)

    # Rigid registration parameters
    params = sitk.GetDefaultParameterMap("rigid")
    params["Metric"] = ["AdvancedMattesMutualInformation"]
    params["MaximumNumberOfIterations"] = ["128"]
    params["AutomaticTransformInitialization"] = ["true"]
    params["AutomaticTransformInitializationMethod"] = ["GeometricalCenter"]

    params["AutomaticScalesEstimation"] = ["true"]
    params["NumberOfSamplesForExactGradient"] = ["10000"]
    elastix.SetParameterMap(params)
    elastix.LogToConsoleOff()
    elastix.Execute()
    result = elastix.GetResultImage()
    transform = elastix.GetTransformParameterMap()
    return result, transform

def apply_transform(moving, transform):
    """Apply an existing transform to an image."""
    transformix = sitk.TransformixImageFilter()
    transformix.SetMovingImage(moving)
    transformix.SetTransformParameterMap(transform)
    transformix.LogToConsoleOff()
    transformix.Execute()
    return transformix.GetResultImage()



def rigid_register(fixed, moving):
    """Return the rigid transform aligning moving to fixed."""
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(32)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    ))
    registration_method.SetMetricSamplingPercentage(0.2, sitk.sitkWallClock)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetShrinkFactorsPerLevel([4,2,1])
    registration_method.SetSmoothingSigmasPerLevel([2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    transform = registration_method.Execute(fixed, moving)
    return transform

def average_transforms(transforms):
    """Average multiple 3D Euler transforms."""
    tx, ty, tz, rx, ry, rz = [], [], [], [], [], []
    for t in transforms:
        p = t.GetParameters()
        rx.append(p[0]); ry.append(p[1]); rz.append(p[2])
        tx.append(p[3]); ty.append(p[4]); tz.append(p[5])
    avg = sitk.Euler3DTransform()
    avg.SetParameters([
        np.mean(rx), np.mean(ry), np.mean(rz),
        np.mean(tx), np.mean(ty), np.mean(tz)
    ])
    return avg


def average_volumes_with_fiducials(volumes, fiducials_list, transform_type="rigid"):
    """
    Average multiple 3D volumes by aligning them with rigid transforms derived from fiducial points.

    Parameters
    ----------
    volumes : list of sitk.Image
        List of 3D image volumes to be averaged.
    fiducials : list of np.ndarray
        List of Nx3 numpy arrays containing corresponding fiducial points for each volume.
        fiducials[i][k] == the k-th fiducial in volume[i].
    reference_index : int
        Index of the volume to use as the reference space.

    Returns
    -------
    sitk.Image
        A 3D volume representing the average of all aligned volumes.
    """

    # Reference image and fiducials
    images = [sitk.ReadImage(vol) for vol in volumes]
    reference_index, reference_image = max(enumerate(images), key=lambda img: np.prod(img[1].GetSize()))
    #fixed_image = sitk.ReadImage(volumes[0])
    fixed_points = fiducials_list[0]

    # Accumulator volume
    acc = sitk.Image(reference_image.GetSize(), sitk.sitkFloat32)
    acc.CopyInformation(reference_image)

    initializer = sitk.LandmarkBasedTransformInitializerFilter()

    if transform_type == "rigid":
        tx = sitk.VersorRigid3DTransform()
    elif transform_type == "affine":
        tx = sitk.AffineTransform(3)
    else:
        raise ValueError("transform_type must be 'rigid' or 'affine'.")

    initializer.SetTransform(tx)
    # metric: sum of squared distances
    initializer.SetMetricAsEuclideanDistance()

    # optimizer
    initializer.SetOptimizerAsGradientDescent(learningRate=1.0,
                                      numberOfIterations=200,
                                      convergenceMinimumValue=1e-6,
                                      convergenceWindowSize=10)
    

    for i in range(1, len(volumes)):
        print(f"Processing volume {i} ...")
        moving_img = sitk.ReadImage(volumes[i])
        moving_points = fiducials_list[i]        
        final_tx = initializer.Execute(fixed_points, moving_points)
        # Resample image to reference grid
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(final_tx)

        moved_img = resampler.Execute(moving_img)

        # accumulate
        acc += sitk.GetArrayFromImage(moved_img)
        count += 1

    # ---------------------------------------
    # Return averaged volume in SimpleITK form
    # ---------------------------------------
    avg_array = acc / count
    avg_img = sitk.GetImageFromArray(avg_array)
    avg_img.CopyInformation(reference_image)

    return avg_img

def sitk_affine_to_matrix4x4(tx: sitk.Transform):
    params = list(tx.GetParameters())
    
    # First 9 values → affine matrix
    A = np.array(params[:9]).reshape(3,3)

    # Last 3 values → translation
    t = np.array(params[9:12]).reshape(3)

    # Build homogeneous 4×4
    M = np.eye(4)
    M[:3,:3] = A
    M[:3, 3] = t

    return M