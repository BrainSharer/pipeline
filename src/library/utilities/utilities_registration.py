"""This code contains helper methods in using Elastix to perform section to section
aligment. Elastix takes in many many different parameter settings. Below are some notes
regarding one particular parameter.

**Notes from the manual regarding MOMENTS vs GEOMETRY:**

 *The* ``CenteredTransformInitializer`` *parameter supports two modes of operation. In the first mode, the centers of
 the images are computed as space coordinates using the image origin, size and spacing. The center of
 the fixed image is assigned as the rotational center of the transform while the vector going from the
 fixed image center to the moving image center is passed as the initial translation of the transform.
 In the second mode, the image centers are not computed geometrically but by using the moments of the
 intensity gray levels.*

 *Keep in mind that the scale of units in rotation and translation is quite different. For example, here
 we know that the first element of the parameters array corresponds to the angle that is measured in radians,
 while the other parameters correspond to the translations that are measured in millimeters*

"""
import os
import sys
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tifffile import imwrite, imread
from skimage import io
from skimage.transform import EuclideanTransform, warp
import numpy as np
import cv2
import math
# from pystackreg import StackReg
from tqdm import tqdm

from library.utilities.utilities_process import read_image, write_image
NUM_ITERATIONS = "2000"


def rigid_transform_to_parmeters(transform,center):
    """convert a 2d transformation matrix (3*3) to the rotation angles, rotation center and translation
    This is used in the manual aligner notebook.

    Args:
        transform (array like): 3*3 array that stores the 2*2 transformation matrix and the 1*2 translation vector for a 
        2D image.  the third row of the array is a place holder of values [0,0,1].

    Returns:
        float: x translation
        float: y translation
        float: rotation angle in arc
        list:  lisf of x and y for rotation center
    """        
    R = transform[:2,:2]
    shift = transform[:2,2]
    tan= R[1,0]/R[0,0]
    rotation = np.arctan(tan)
    xshift,yshift = shift-center +np.dot(R, center)
    return xshift,yshift,rotation,center

def parameters_to_rigid_transform(rotation, xshift, yshift, center):
    """Takes the rotation, xshift, yshift that were created by Elastix
    and stored in the elastix_transformation table. Creates a matrix of the
    rigid transformation.

    :param rotation: a float designating the rotation in radians
    :param xshift: a float for showing how much the moving image shifts in the X direction.
    :param yshift: a float for showing how much the moving image shifts in the Y direction.
    :param center: tuple of floats showing the center of the image.
    :returns: the 3x3 rigid transformation
    """

    rotation, xshift, yshift = np.array([rotation, xshift, yshift]).astype(
        np.float32
    )
    center = np.array(center).astype(np.float32)
    R = np.array(
        [
            [np.cos(rotation), -np.sin(rotation)],
            [np.sin(rotation), np.cos(rotation)],
        ]
    ).astype(np.float32)
    shift = center + (xshift, yshift) - np.dot(R, center)
    T = np.vstack([np.column_stack([R, shift]), [0, 0, 1]])

    return T


def create_rigid_parameters(elastixImageFilter, defaultPixelValue="0.0", debug=False, iteration=0):
    """
    Create and return a dictionary of rigid registration parameters for elastixImageFilter.
    The iterations have been reduced as we are doing two alignment processes.

    Parameters:
    - elastixImageFilter: The elastix image filter object.
    - defaultPixelValue: The default pixel value for the registration.

    Returns:
    - rigid_params: A dictionary of rigid registration parameters.
    """

    rigid_params = elastixImageFilter.GetDefaultParameterMap("rigid")
    rigid_params["AutomaticTransformInitialization"] = ["true"]
    rigid_params["AutomaticTransformInitializationMethod"] = ["GeometricalCenter"]
    rigid_params["FixedInternalImagePixelType"] = ["float"]
    rigid_params["MovingInternalImagePixelType"] = ["float"]
    rigid_params["FixedImageDimension"] = ["2"]
    rigid_params["MovingImageDimension"] = ["2"]
    rigid_params["UseDirectionCosines"] = ["false"]
    rigid_params["HowToCombineTransforms"] = ["Compose"]
    rigid_params["DefaultPixelValue"] = [defaultPixelValue]
    rigid_params["WriteResultImage"] = ["false"]    
    rigid_params["WriteIterationInfo"] = ["true"]
    rigid_params["Resampler"] = ["DefaultResampler"]
    rigid_params["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
    rigid_params["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
    rigid_params["NumberOfResolutions"] = ["5"]
    rigid_params["Registration"] = ["MultiMetricMultiResolutionRegistration"]
    rigid_params["Transform"] = ["EulerTransform"]
    rigid_params["AutomaticScalesEstimation"] = ["true"]
    # the AdvancedMattesMutualInformation metric really helps with the alignment
    rigid_params["Metric"] = ["AdvancedNormalizedCorrelation", "AdvancedMattesMutualInformation"]
    rigid_params["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    rigid_params["UseRandomSampleRegion"] = ["true"]
    rigid_params["SampleRegionSize"] = ["50"]
    if debug:
        rigid_params["MaximumNumberOfIterations"] = ["250"]
    else:
        rigid_params["MaximumNumberOfIterations"] = ["2500"]

    rigid_params["Interpolator"] = ["NearestNeighborInterpolator"]
    rigid_params["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    rigid_params["ImageSampler"] = ["Random"]

    return rigid_params

def create_affine_parameters(elastixImageFilter, defaultPixelValue="0.0", debug=False):
    """
    Create and return a dictionary of rigid registration parameters for elastixImageFilter.
    The iterations have been reduced as we are doing two alignment processes.

    Parameters:
    - elastixImageFilter: The elastix image filter object.
    - defaultPixelValue: The default pixel value for the registration.

    Returns:
    - rigid_params: A dictionary of rigid registration parameters.
    """

    rigid_params = elastixImageFilter.GetDefaultParameterMap("affine")
    rigid_params["AutomaticTransformInitialization"] = ["true"]
    rigid_params["AutomaticTransformInitializationMethod"] = ["GeometricalCenter"]
    rigid_params["FixedInternalImagePixelType"] = ["float"]
    rigid_params["MovingInternalImagePixelType"] = ["float"]
    rigid_params["FixedImageDimension"] = ["2"]
    rigid_params["MovingImageDimension"] = ["2"]
    rigid_params["UseDirectionCosines"] = ["false"]
    rigid_params["HowToCombineTransforms"] = ["Compose"]
    rigid_params["DefaultPixelValue"] = [defaultPixelValue]
    rigid_params["WriteResultImage"] = ["true"]
    rigid_params["(ResultImageFormat"] = ["tif"]    
    rigid_params["WriteIterationInfo"] = ["true"]
    rigid_params["Resampler"] = ["DefaultResampler"]
    rigid_params["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
    rigid_params["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
    rigid_params["NumberOfResolutions"] = ["5"]
    rigid_params["Registration"] = ["MultiMetricMultiResolutionRegistration"]
    rigid_params["Transform"] = ["AffineTransform"]
    rigid_params["AutomaticScalesEstimation"] = ["true"]
    # the AdvancedMattesMutualInformation metric really helps with the alignment
    rigid_params["Metric"] = ["AdvancedNormalizedCorrelation", "AdvancedMattesMutualInformation"]
    rigid_params["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    rigid_params["UseRandomSampleRegion"] = ["true"]
    rigid_params["SampleRegionSize"] = ["50"]
    if debug:
        rigid_params["MaximumNumberOfIterations"] = ["500   "]
    else:
        rigid_params["MaximumNumberOfIterations"] = [NUM_ITERATIONS]

    rigid_params["Interpolator"] = ["NearestNeighborInterpolator"]
    rigid_params["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    rigid_params["ImageSampler"] = ["Random"]

    return rigid_params


def rescale_transformations(transforms: dict, scaling_factor: float) -> dict:
    """
    Rescales the transformation matrices by a given scaling factor.
    Args:
        transforms (dict): A dictionary where keys are file names and values are 
                           2D numpy arrays representing transformation matrices.
        scaling_factor (float): The factor by which to scale the transformation matrices.
    Returns:
        dict: A dictionary with the same keys as the input, where each transformation 
              matrix has been rescaled by the given scaling factor.
    """

    tf_mat_mult_factor = np.array([[1, 1, 32.0], [1, 1, 32.0]])

    transforms_to_anchor = {}
    for img_name, tf in transforms.items():
        transforms_to_anchor[img_name] = \
            convert_2d_transform_forms(np.reshape(tf, (3, 3))[:2] * tf_mat_mult_factor)
    return transforms_to_anchor




    tf_mat_mult_factor = np.array([[1, 1, scaling_factor], [1, 1, scaling_factor]], dtype=np.float32)
    #print(tf_mat_mult_factor)
    print(f'Changing tx and ty in the rigid transformation by a factor of: {scaling_factor}')

    transforms_to_anchor = {}
    for file, transform in transforms.items():
        transformed = np.reshape(transform, (3, 3))[:2] * tf_mat_mult_factor
        #transform[:, -1] *= scaling_factor
        #print(file, transform)
        transforms_to_anchor[file] = transformed
        
    return transforms_to_anchor

def convert_2d_transform_forms(arr):
    """Helper method used by rescale_transformations

    :param arr: an array of data to vertically stack
    :return: a numpy array
    """

    return np.vstack([arr, [0, 0, 1]])


def align_image_to_affine(file_key):
    """This is the method that takes the rigid transformation and uses
    PIL to align the image.
    This method takes about 20 seconds to run as compared to scikit's version 
    which takes 220 seconds to run on a full scale image.

    :param file_key: tuple of file input and output
    :return: nothing
    """
    infile, outfile, T, fillcolor = file_key
    basepath = os.path.basename(os.path.normpath(infile))

    try:
        im0 = imread(infile)
    except Exception as e:
        print(f'Could not use tifffile to open={basepath}')
        print(f'Error={e}')
        sys.exit()


    # If the image is sRGB 16bit, convert to 8bit
    if im0.ndim == 3 and im0.dtype == np.uint16:
        # PIL can't handle sRGB 16bit images
        try:
            im0 = (im0/256).astype(np.uint8)
        except Exception as e:
            print(f'Could not convert file {basepath} to 8bit')
            print(f'Error={e}')
            sys.exit()

    # image is now in numpy array format, we need to get in PIL format to perform the transformation
    try:
        im0 = Image.fromarray(im0)
    except Exception as e:
        print(f'Could not convert file {basepath} to PIL ')
        print(f'Error={e}')
        sys.exit()
    try:
        im1 = im0.transform((im0.size), Image.Transform.AFFINE, T.flatten()[:6], resample=Image.Resampling.BICUBIC, fillcolor=fillcolor)
    except Exception as e:
        print(f'align image to affine: could not transform {infile}')
        print(f'Error={e}')
        sys.exit()

    del im0
    # Put PIL image to numpy
    try:
        im1 = np.asarray(im1)
    except Exception as e:
        print(f'could not convert file type={type(im1)}: {basepath} to numpy')
        print(f'Error={e}')
        sys.exit()

    # The final image: im1 is now a numpy array so we can use
    # tifffile to save the image
    try:
        imwrite(outfile, im1, bigtiff=True, compression='LZW')
    except Exception as e:
        print('could not save {outfile} with tifffile')
        print(f'Error={e}')
        sys.exit()

    del im1
    return


def tif_to_png(file_key):
    """This method creates a PNG from a TIF
    :param file_key: tuple of file input and output
    :return: nothing
    """
    infile, outfile = file_key
    img = read_image(infile)
    img = (img / 256).astype(np.uint8)
    write_image(outfile, img)


def create_rigid_transformation(center, angle, tx, ty):
    """
    Applies a rigid transformation to the input image.

    Parameters:
    - image: Input image as a NumPy array.
    - angle: Rotation angle in degrees (positive values rotate counter-clockwise).
    - tx: Translation along the x-axis.
    - ty: Translation along the y-axis.

    Returns:
    - Transformed image as a NumPy array.
    """
    # Get the dimensions of the image
    #height, width = image.shape[:2]

    # Compute the center of the image
    #center = (width / 2, height / 2)

    # Create the rotation matrix
    #angle = math.radians(angle)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1)

    # Add the translation to the rotation matrix
    rotation_matrix[0, 2] += tx
    rotation_matrix[1, 2] += ty
    return rotation_matrix


def apply_rigid_transform_opencv(image_path, M, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Get image dimensions
    (h, w) = image.shape[:2]
    M = M.reshape(2, 3)

    """
    # Compute the center of the image
    center = (w // 2, h // 2)

    # Create the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply the translation
    M[0, 2] += tx
    M[1, 2] += ty
    """
    # Perform the affine transformation
    transformed_image = cv2.warpAffine(image, M, (w, h))

    # Save the transformed image
    cv2.imwrite(output_path, transformed_image)


def apply_rigid_transform_skimage(file_key):
    # Load the image
    infile, outfile, T, _ = file_key
    #image_path, transform, output_path = file_key
    image = io.imread(infile)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Create the Euclidean transformation
    #transform = EuclideanTransform(rotation=angle_rad, translation=(tx, ty))

    # Apply the transformation
    transformed_image = warp(image, np.linalg.inv(T), output_shape=image.shape)

    # Save the transformed image
    #io.imsave(outfile, transformed_image)
    write_image(outfile, transformed_image)

def compute_rigid_transformations(input_path):
    """
    Computes rigid transformation matrices for each image in the stack relative to a fixed reference image.

    Parameters:
    - image_stack: List or array of 2D numpy arrays representing the images.
    - reference_index: Index of the reference image in the stack (default is 0).

    Returns:
    - transformations: List of 3x3 numpy arrays representing the affine transformation matrices.
    """
    files = sorted(os.listdir(input_path))

    reference_index = len(files) // 2

    image_stack = []
    for file in files:
        image = read_image(os.path.join(input_path, file))
        image_stack.append(image)


    # Initialize the StackReg object for rigid transformation
    sr = StackReg(StackReg.RIGID_BODY)

    # Reference image
    reference_image = image_stack[reference_index]

    # List to store transformation matrices
    transformations = {}

    # Compute transformation matrix for each image
    """
    for i, moving_image in enumerate(tqdm(image_stack)):
        key = f"{str(i).zfill(3)}.tif"
        if i == reference_index:
            # The reference image's transformation is the identity matrix
            #transformations.append(np.eye(3))
            transformations[key] = np.eye(3)
        else:
            # Compute the transformation matrix
            transformation_matrix = sr.register(reference_image, moving_image)
            #transformations.append(transformation_matrix)
            transformations[key] = transformation_matrix

    return transformations
    """
    for i, moving_image in enumerate(tqdm(image_stack)):
        key = f"{str(i).zfill(3)}.tif"
        if i == 0:
            # The reference image's transformation is the identity matrix
            #transformations.append(np.eye(3))
            transformations[key] = np.eye(3)
        else:
            reference_image = image_stack[i-1]
            # Compute the transformation matrix
            transformation_matrix = sr.register(reference_image, moving_image)
            #transformations.append(transformation_matrix)
            transformations[key] = transformation_matrix

    return transformations


def find_matching_points(image1, image2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    # Initialize FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
    # Estimate the Euclidean (rigid) transformation
    model = EuclideanTransform()
    model.estimate(src_pts, dst_pts)

    # Retrieve the transformation matrix
    transformation_matrix = model.params
    # Apply the transformation to the second image
    aligned_image2 = warp(image2, inverse_map=model.inverse, output_shape=image1.shape)

