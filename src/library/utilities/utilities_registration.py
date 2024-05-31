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
import SimpleITK as sitk

from library.utilities.utilities_process import SCALING_FACTOR, read_image, write_image
NUM_ITERATIONS = "1500"


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

    :param rotation: a float designating the rotation
    :param xshift: a float for showing how much the moving image shifts in the X direction.
    :param yshift: a float for showing how much the moving image shifts in the Y direction.
    :param center: tuple of floats showing the center of the image.
    :returns: the 3x3 rigid transformation
    """

    rotation, xshift, yshift = np.array([rotation, xshift, yshift]).astype(
        np.float16
    )
    center = np.array(center).astype(np.float16)
    R = np.array(
        [
            [np.cos(rotation), -np.sin(rotation)],
            [np.sin(rotation), np.cos(rotation)],
        ]
    )
    shift = center + (xshift, yshift) - np.dot(R, center)
    T = np.vstack([np.column_stack([R, shift]), [0, 0, 1]])
    return T


def create_rigid_parameters(elastixImageFilter, defaultPixelValue="0.0", debug=False):
    """
    Create and return a dictionary of rigid registration parameters for elastixImageFilter.

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
    rigid_params["Resampler"] = ["DefaultResampler"]
    rigid_params["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
    rigid_params["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
    rigid_params["NumberOfResolutions"] = ["5"]
    rigid_params["Registration"] = ["MultiMetricMultiResolutionRegistration"]
    rigid_params["Transform"] = ["EulerTransform"]
    rigid_params["AutomaticScalesEstimation"] = ["true"]
    #####rigid_params["Metric"] = ["AdvancedNormalizedCorrelation", "AdvancedMattesMutualInformation"]
    rigid_params["Metric"] = ["AdvancedNormalizedCorrelation"]
    rigid_params["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    if debug:
        rigid_params["MaximumNumberOfIterations"] = ["150"]
        rigid_params["NumberOfSpatialSamples"] = ["2048"]
    else:
        rigid_params["MaximumNumberOfIterations"] = ["2500"]
        rigid_params["NumberOfSpatialSamples"] = ["12288"]

    rigid_params["Interpolator"] = ["NearestNeighborInterpolator"]
    rigid_params["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    rigid_params["ImageSampler"] = ["Random"]

    return rigid_params


def create_downsampled_transforms(transforms: dict, downsample: bool, scaling_factor: float) -> dict:
    """Changes the dictionary of transforms to the correct resolution


    :param animal: prep_id of animal we are working on animal
    :param transforms: dictionary of filename:array of transforms
    :param downsample: boolean: either true for thumbnails, false for full resolution images
    :return: corrected dictionary of filename: array  of transforms
    """

    if downsample:
        transforms_scale_factor = 1
    else:
        transforms_scale_factor = scaling_factor

    tf_mat_mult_factor = np.array([[1, 1, transforms_scale_factor], [1, 1, transforms_scale_factor]])

    transforms_to_anchor = {}
    for img_name, tf in transforms.items():
        transforms_to_anchor[img_name] = \
            convert_2d_transform_forms(np.reshape(tf, (3, 3))[:2] * tf_mat_mult_factor)
    return transforms_to_anchor


def create_scaled_transform(T):
    """Creates a transform (T) to the correct resolution
    """
    transforms_scale_factor = SCALING_FACTOR

    tf_mat_mult_factor = np.array([[1, 1, transforms_scale_factor], [1, 1, transforms_scale_factor]])
    Ts = convert_2d_transform_forms(np.reshape(T, (3, 3))[:2] * tf_mat_mult_factor)
    return Ts


def convert_2d_transform_forms(arr):
    """Helper method used by create_downsampled_transforms

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
    infile, outfile, T = file_key
    try:
        im1 = Image.open(infile)
    except:
        
        try:
            im = imread(infile)
        except Exception as e:
            print(f'Could not use tifffile to open={infile}')
            print(f'Error={e}')
            sys.exit()
        
        try:
            im1 = Image.fromarray(im)
        except Exception as e:
            print(f'Could not convert file type={type(im)} to PIL ')
            print(f'Error={e}')
            sys.exit()
        del im

    try:
        im2 = im1.transform((im1.size), Image.Transform.AFFINE, T.flatten()[:6], resample=Image.Resampling.NEAREST)
    except Exception as e:
        print(f'align image to affine, could not transform {infile} to:')
        print(outfile)
        print(f'Error={e}')
        sys.exit()
    
    del im1

    try:
        im2.save(outfile)
    except:

        try:
            im2 = np.asarray(im2)
        except Exception as e:
            print(f'could not convert file type={type(im2)} to numpy')
            print(f'Error={e}')
            sys.exit()

        try:
            imwrite(outfile, im2)
        except Exception as e:
            print('could not save {outfile} with tifffile')
            print(f'Error={e}')
            sys.exit()

    del im2
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



