'''
Trimmed down version of the original atlas utilities and also utilities_atlas_lite
'''
from colorsys import hsv_to_rgb
import random
import numpy as np
import vtk
from vtk.util import numpy_support

import mcubes # https://github.com/pmneila/PyMCubes

from library.controller.sql_controller import SqlController
from library.registration.algorithm import umeyama

singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']



def get_common_structure(brains):
    '''
    Finds the common structures between a brain and the atlas. These are used
    for the inputs to the rigid transformation.
    :param brains: a list (usually just one brain) of brain names
    '''
    sqlController = SqlController('MD594') # just to declare var
    common_structures = set()
    for brain in brains:
        common_structures = common_structures | set(sqlController.get_annotation_points_entry(brain).keys())
    common_structures = list(sorted(common_structures))
    return common_structures


def get_transformation(animal):
    '''
    Fetches the common structures between the atlas and and animal and creates
    the rigid transformation with the umemeya method. Returns the rotation
    and translation matrices.
    :param animal: string of the brain name
    '''
    sqlController = SqlController(animal) # just to declare var
    pointdata = sqlController.get_annotation_points_entry(animal)
    atlas_centers = sqlController.get_annotation_points_entry('Atlas', FK_input_id=1, person_id=16)
    common_structures = get_common_structure(['Atlas', animal])
    point_structures = sorted(pointdata.keys())
    
    dst_point_set = np.array([atlas_centers[s] for s in point_structures if s in common_structures and s in atlas_centers]).T
    point_set = np.array([pointdata[s] for s in point_structures if s in common_structures and s in atlas_centers]).T
    R, t = umeyama(point_set, dst_point_set)
    return R, t 


def save_mesh(polydata, filename):
    stlWriter = vtk.vtkSTLWriter()
    stlWriter.SetFileName(filename)
    stlWriter.SetInputData(polydata)
    stlWriter.Write()


def symmetricalize_volume(volume):
    """
    Replace the volume with the average of its left half and right half.
    """
    zcenter = volume.shape[2] // 2
    symmetrical_volume = volume.copy()
    left_half = volume[..., :zcenter]
    right_half = volume[..., -zcenter:]
    left_half_averaged = (left_half + right_half[..., ::-1]) // 2.
    symmetrical_volume[..., :zcenter] = left_half_averaged
    symmetrical_volume[..., -zcenter:] = left_half_averaged[..., ::-1]
    return symmetrical_volume


def volume_to_polygon(volume, origin, times_to_simplify=0, min_vertices=200):
    """
    Convert a volume to a mesh, either as vertices/faces tuple or a vtk.Polydata.

    Args:
        level (float): the level to threshold the input volume
        min_vertices (int): minimum number of vertices. Simplification will stop if the number of vertices drops below this value.
        return_vertex_face_list (bool): If True, return only (vertices, faces); otherwise, return polydata.
    """
    # need this otherwise the sides of volume will not close and expose the hollow inside of structures
    if volume is None:
        print('In volume_to_polygon, volume is None')
        return
    vol_padded = np.pad(volume, ((5, 5), (5, 5), (5, 5)), 'constant')
    # more than 5 times faster than skimage.marching_cube + correct_orientation
    vertices, faces = mcubes.marching_cubes(vol_padded, 0)
    buffer = 0
    vertices = vertices + origin - (buffer, buffer, buffer)
    polydata = mesh_to_polydata(vertices, faces)

    for _ in range(times_to_simplify):
        deci = vtk.vtkQuadricDecimation()
        deci.SetInputData(polydata)
        deci.SetTargetReduction(0.8)
        deci.Update()
        polydata = deci.GetOutput()

        if polydata.GetNumberOfPoints() < min_vertices:
            break
    return polydata

def find_merged_bounding_box(bounding_box=None):
    bounding_box = np.array(bounding_box)
    xmin, ymin, zmin = np.min(bounding_box[:, [0,2,4]], axis=0)
    xmax, ymax, zmax = np.max(bounding_box[:, [1,3,5]], axis=0)
    bbox = xmin, xmax, ymin, ymax, zmin, zmax
    return bbox

def bbox(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    #return img[rmin:rmax, cmin:cmax, zmin:zmax]
    return rmin, cmin, zmin

def crop_and_pad_volume(volumes, bounding_box, merged_bounding_box):
    """
    Crop and pad an volume.
    in_vol and in_bbox together define the input volume in a underlying space.
    out_bbox then defines how to crop the underlying space, which generates the output volume.
    Args:
        in_bbox ((6,) array): the bounding box that the input volume is defined on. If None, assume origin is at (0,0,0) of the input volume.
        in_origin ((3,) array): the input volume origin coordinate in the space. Used only if in_bbox is not specified. Default is (0,0,0), meaning the input volume is located at the origin of the underlying space.
        out_bbox ((6,) array): the bounding box that the output volume is defined on. If not given, each dimension is from 0 to the max reach of any structure.
    Returns:
        3d-array: cropped/padded volume
    """
    bounding_box = np.array(bounding_box).astype(np.int)
    xmin, xmax, ymin, ymax, zmin, zmax = bounding_box
    merged_bounding_box = np.array(merged_bounding_box).astype(np.int)
    merged_xmin, merged_xmax, merged_ymin, merged_ymax, merged_zmin, merged_zmax = merged_bounding_box
    merged_xdim = merged_xmax - merged_xmin + 1
    merged_ydim = merged_ymax - merged_ymin + 1
    merged_zdim = merged_zmax - merged_zmin + 1
    if merged_xmin > xmax or merged_xmax < xmin or merged_ymin > ymax or merged_ymax < ymin or merged_zmin > zmax or merged_zmax < zmin:
        return np.zeros((merged_xdim,merged_ydim, merged_zdim), np.int)
    if merged_xmax > xmax:
        volumes = np.pad(volumes, pad_width=[(0,merged_xmax-xmax),(0,0),(0,0)], mode='constant', constant_values=0)
    if merged_ymax > ymax:
        volumes = np.pad(volumes, pad_width=[(0,0),(0,merged_ymax-ymax),(0,0)], mode='constant', constant_values=0)
    if merged_zmax > zmax:
        volumes = np.pad(volumes, pad_width=[(0,0),(0,0),(0, merged_zmax-zmax)], mode='constant', constant_values=0)
    out_vol = np.zeros((merged_xdim , merged_ydim, merged_zdim), volumes.dtype)
    
    out_vol[xmin-merged_xmin:merged_xmax+1-merged_xmin,
            ymin-merged_ymin:merged_ymax+1-merged_ymin,
            zmin-merged_zmin:merged_zmax+1-merged_zmin] = volumes
    assert out_vol.shape[0] == merged_xdim
    assert out_vol.shape[1] == merged_ydim
    assert out_vol.shape[2] == merged_zdim
    return out_vol

def crop_and_pad_volumes(merged_bounding_box=None, bounding_box_volume=None):
    """
    Args:
        out_bbox ((6,)-array): the output bounding box, must use the same reference system as the vol_bbox input.
        vol_bbox 
    Returns:
        list of 3d arrays or dict {structure name: 3d array}
    """
    if isinstance(bounding_box_volume, dict):
        bounding_box_volume = list(bounding_box_volume.items())
    vols = [crop_and_pad_volume(volume, bounding_box, merged_bounding_box=merged_bounding_box) for (volume, bounding_box) in bounding_box_volume]
    # from atlas.BrainStructureManager import Brain
    # brain = Brain('MD585')
    # id = 0
    # axis = 1
    # brain.plotter.plot_3d_image_stack(bounding_box_volume[id][0],axis)
    # brain.plotter.plot_3d_image_stack(vols[id],axis)
    return vols


def mesh_to_polydata(vertices, faces, num_simplify_iter=0, smooth=False):
    """
    Args:
        vertices ((num_vertices, 3) arrays)
        faces ((num_faces, 3) arrays)
    """
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()

    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")
    colors.SetNumberOfTuples(len(faces))

    for pt_ind, (x,y,z) in enumerate(vertices):
        points.InsertPoint(pt_ind, x, y, z)

    if len(faces) > 0:
        cells = vtk.vtkCellArray()
        cell_arr = np.empty((len(faces)*4, ), int)
        cell_arr[::4] = 3
        cell_arr[1::4] = faces[:,0]
        cell_arr[2::4] = faces[:,1]
        cell_arr[3::4] = faces[:,2]
        cell_vtkArray = numpy_support.numpy_to_vtkIdTypeArray(cell_arr, deep=1)
        cells.SetCells(len(faces), cell_vtkArray)
        colors.InsertNextTuple3(10,255,0)

    polydata.SetPoints(points)

    if len(faces) > 0:
        polydata.SetPolys(cells)
        polydata.GetCellData().SetScalars(colors)
        polydata = simplify_polydata(polydata, num_simplify_iter, smooth)

    return polydata


def simplify_polydata(polydata, num_simplify_iter=0, smooth=False):
    for _ in range(num_simplify_iter):
        deci = vtk.vtkQuadricDecimation()
        deci.SetInputData(polydata)
        deci.SetTargetReduction(0.8)
        # 0.8 means each iteration causes the point number to drop to 20% the original
        deci.Update()
        polydata = deci.GetOutput()

        if smooth:
            smoother = vtk.vtkWindowedSincPolyDataFilter()
    #         smoother.NormalizeCoordinatesOn()
            smoother.SetPassBand(.1)
            smoother.SetNumberOfIterations(20)
            smoother.SetInputData(polydata)
            smoother.Update()
            polydata = smoother.GetOutput()

        if polydata.GetNumberOfPoints() < 200:
            break

    return polydata

def generate_random_rgb_color():
  """Generates a random color in RGB format (tuple of three integers)."""
  red = random.randint(0, 255)
  green = random.randint(0, 255)
  blue = random.randint(0, 255)
  return (red, green, blue)

def number_to_rgb(number):
    """
    Converts a number from 1 to 65000 to an RGB color.
    The number is mapped to the hue of an HSV color.
    """
    if not 1 <= number <= 65000:
        raise ValueError("Number must be between 1 and 65000")

    # Normalize the number to a value between 0 and 1
    # We subtract 1 from the numerator and denominator to handle the 1-max(allen ID) range.
    normalized = (number - 1) / 1000

    # Convert normalized value to a hue in the range [0, 1]
    # Keep saturation and value at 1 for the full color spectrum
    r, g, b = hsv_to_rgb(normalized, 1.0, 1.0)

    # Scale the RGB values from [0, 1] to [0, 255]
    red = int(r * 255)
    green = int(g * 255)
    blue = int(b * 255)

    return (red, green, blue)

def mask_to_mesh(mask: np.ndarray, origin: tuple, ply_filename: str, color=(255, 0, 0)):
    """
    This is a simpler way to create 3D meshes from volumes. Instead of creating stl files, it creates ply files.
    ply files can have color and it is much faster.
    Convert a 3D numpy mask to a mesh and export as PLY.
    
    Parameters
    ----------
    mask : np.ndarray
        3D numpy array, binary or integer labels.
    ply_filename : str
        Path to output .ply file.
    color : tuple of 3 ints
        RGB color for the mesh (0–255).
    """
    # Ensure binary
    spacing = (1.0, 1.0, 1.0)  # Set voxel spacing if needed
    mask = (mask > 0).astype(np.uint8)

    # Convert numpy array to VTK image
    depth_array = numpy_support.numpy_to_vtk(
        num_array=mask.ravel(order='F'),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR
    )
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(mask.shape)  # Note: VTK uses (x, y, z), our data is already in (x, y, z) order
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(origin)
    vtk_image.GetPointData().SetScalars(depth_array)

    # Extract surface using marching cubes
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vtk_image)
    mc.SetValue(0, 0.5)  # isosurface at 0.5 for binary mask
    mc.Update()

    # Use the windowed sinc filter for less shrinkage
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(mc.GetOutputPort())    
    smoother.SetNumberOfIterations(200)
    smoother.BoundarySmoothingOn()
    smoother.FeatureEdgeSmoothingOn()
    smoother.SetFeatureAngle(45.0)
    smoother.SetPassBand(0.1)  # Controls the filter's strength, 0.1 is strong
    smoother.Update()

    polydata = smoother.GetOutput()

    # Add color
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")
    for _ in range(polydata.GetNumberOfPoints()):
        colors.InsertNextTuple3(*color)
    polydata.GetPointData().SetScalars(colors)

    # Write to PLY
    writer = vtk.vtkPLYWriter()
    writer.SetFileName(ply_filename)
    writer.SetInputData(polydata)
    writer.SetFileTypeToBinary()
    writer.SetArrayName("Colors")
    writer.SetColorModeToDefault()
    writer.Update()
    writer.Write()


#    'Pn_L': 771,
#    'Pn_R': 771,

allen_structures = {
    'SC': [851],
    'IC': [811],
    'AP': 207,
    'RtTg': 146,
    'SNR_L': 381,
    'SNR_R': 381,
    'PBG_L': 874,
    'PBG_R': 874,
    '3N_L': 35,
    '3N_R': 35,
    '4N_L': 115,
    '4N_R': 115,
    'SNC_L': 374,
    'SNC_R': 374,
    'VLL_L': 612,
    'VLL_R': 612,
    '5N_L': 621,
    '5N_R': 621,
    'LC_L': 147,
    'LC_R': 147,
    'DC_L': 96,
    'DC_R': 96,
    'Sp5C_L': 429,
    'Sp5C_R': 429,
    'Sp5I_L': 437,
    'Sp5I_R': 437,
    'Sp5O_L': 445,
    'Sp5O_R': 445,
    '6N_L': 653,
    '6N_R': 653,
    '7N_L': 661,
    '7N_R': 661,
    '7n_L': 798,
    '7n_R': 798,
    'Amb_L': 135,
    'Amb_R': 135,
    'LRt_L': 235,
    'LRt_R': 235,
    '10N_L': 1000,
    '10N_R': 1001,
    '12N': 1002,
    'Pn_L': 1003,
    'Pn_R': 1004,
    'RMC_L': 1005,
    'RMC_R': 1006,
    'Tz_L': 1007,
    'Tz_R': 1008,
    'VCA_L': 1009,
    'VCA_R': 1010,
    'VCP_L': 1011,
    'VCP_R': 1012,
}
