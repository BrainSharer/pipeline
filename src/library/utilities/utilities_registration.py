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
import gc
import numpy as np
from PIL import Image, TiffImagePlugin
Image.MAX_IMAGE_PIXELS = None
TiffImagePlugin.DEBUG = True
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


def create_rigid_parameters(elastixImageFilter):
    """Creates the rigid paramaters used by Elastix.
    This sets lots of parameters in this dictionary and it used multiple places.

    :param elastixImageFilter: object set in previous method for Elastix.
    :return: dictionary of parameters
    """

    rigid_params = elastixImageFilter.GetDefaultParameterMap("rigid")
    rigid_params["AutomaticTransformInitializationMethod"] = ["GeometricalCenter"]
    rigid_params["FixedInternalImagePixelType"] = ["float"]
    rigid_params["MovingInternalImagePixelType"] = ["float"]
    rigid_params["FixedImageDimension"] = ["2"]
    rigid_params["MovingImageDimension"] = ["2"]
    rigid_params["UseDirectionCosines"] = ["false"]
    rigid_params["HowToCombineTransforms"] = ["Compose"]
    rigid_params["DefaultPixelValue"] = ["0.0"]
    rigid_params["WriteResultImage"] = ["false"]    
    rigid_params["Resampler"] = ["DefaultResampler"]
    rigid_params["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
    rigid_params["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
    rigid_params["NumberOfResolutions"] = ["8"]
    rigid_params["Registration"] = ["MultiMetricMultiResolutionRegistration"]
    rigid_params["Transform"] = ["EulerTransform"]
    rigid_params["AutomaticScalesEstimation"] = ["true"]
    rigid_params["AutomaticTransformInitialization"] = ["true"]
    rigid_params["Metric"] = ["AdvancedNormalizedCorrelation", "AdvancedMattesMutualInformation"]
    rigid_params["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    rigid_params["MaximumNumberOfIterations"] = [NUM_ITERATIONS]
    rigid_params["Interpolator"] = ["NearestNeighborInterpolator"]
    rigid_params["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    rigid_params["ImageSampler"] = ["Random"]


    return rigid_params



def create_rigid_parametersXXX(elastixImageFilter):
    """Creates the rigid paramaters used by Elastix.
    This sets lots of parameters in this dictionary and it used multiple places.

    :param elastixImageFilter: object set in previous method for Elastix.
    :return: dictionary of parameters
    """

    rigid_params = elastixImageFilter.GetDefaultParameterMap("rigid")
    rigid_params["AutomaticTransformInitializationMethod"] = [
        "GeometricalCenter"
    ]
    rigid_params["ShowExactMetricValue"] = ["false"]
    rigid_params["CheckNumberOfSamples"] = ["true"]
    rigid_params["NumberOfSpatialSamples"] = ["7500"]
    rigid_params["SampleRegionSize"] = ["5.0"]
    rigid_params["SubtractMean"] = ["true"]
    rigid_params["MaximumNumberOfSamplingAttempts"] = ["0"]
    rigid_params["SigmoidInitialTime"] = ["0"]
    rigid_params["MaxBandCovSize"] = ["192"]
    rigid_params["NumberOfBandStructureSamples"] = ["10"]
    rigid_params["UseAdaptiveStepSizes"] = ["true"]
    rigid_params["AutomaticParameterEstimation"] = ["true"]
    rigid_params["MaximumStepLength"] = ["10"]
    rigid_params["NumberOfGradientMeasurements"] = ["0"]
    rigid_params["NumberOfJacobianMeasurements"] = ["1000"]
    rigid_params["NumberOfSamplesForExactGradient"] = ["100000"]
    rigid_params["SigmoidScaleFactor"] = ["0.1"]
    rigid_params["ASGDParameterEstimationMethod"] = ["Original"]
    rigid_params["UseMultiThreadingForMetrics"] = ["true"]
    rigid_params["SP_A"] = ["20"]
    rigid_params["UseConstantStep"] = ["false"]
    ## The internal pixel type, used for internal computations
    ## Leave to float in general.
    ## NB: this is not the type of the input images! The pixel
    ## type of the input images is automatically read from the
    ## images themselves.
    ## This setting can be changed to "short" to save some memory
    ## in case of very large 3D images.
    rigid_params["FixedInternalImagePixelType"] = ["float"]
    rigid_params["MovingInternalImagePixelType"] = ["float"]
    ## note that some other settings may have to specified
    ## for each dimension separately.
    rigid_params["FixedImageDimension"] = ["2"]
    rigid_params["MovingImageDimension"] = ["2"]
    ## Specify whether you want to take into account the so-called
    ## direction cosines of the images. Recommended: true.
    ## In some cases, the direction cosines of the image are corrupt,
    ## due to image format conversions for example. In that case, you
    ## may want to set this option to "false".
    rigid_params["UseDirectionCosines"] = ["true"]
    ## **************** Main Components **************************
    ## The following components should usually be left as they are:
    rigid_params["Registration"] = ["MultiResolutionRegistration"]
    rigid_params["Interpolator"] = ["BSplineInterpolator"]
    rigid_params["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
    rigid_params["Resampler"] = ["DefaultResampler"]
    ## These may be changed to Fixed/MovingSmoothingImagePyramid.
    ## See the manual.
    ##(FixedImagePyramid "FixedRecursiveImagePyramid']
    ##(MovingImagePyramid "MovingRecursiveImagePyramid']
    rigid_params["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
    rigid_params["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
    ## The following components are most important:
    ## The optimizer AdaptiveStochasticGradientDescent (ASGD) works
    ## quite ok in general. The Transform and Metric are important
    ## and need to be chosen careful for each application. See manual.
    rigid_params["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    rigid_params["Transform"] = ["EulerTransform"]
    ##(Metric "AdvancedMattesMutualInformation")
    ## testing 17 dec
    ##rigid_params["Metric"] = ["AdvancedNormalizedCorrelation"]
    rigid_params["Metric"] = ["AdvancedMattesMutualInformation"]
    ## ***************** Transformation **************************
    ## Scales the rotations compared to the translations, to make
    ## sure they are in the same range. In general, it's best to
    ## use automatic scales estimation:
    rigid_params["AutomaticScalesEstimation"] = ["true"]
    ## Automatically guess an initial translation by aligning the
    ## geometric centers of the fixed and moving.
    rigid_params["AutomaticTransformInitialization"] = ["true"]
    ## Whether transforms are combined by composition or by addition.
    ## In generally, Compose is the best option in most cases.
    ## It does not influence the results very much.
    rigid_params["HowToCombineTransforms"] = ["Compose"]
    ## ******************* Similarity measure *********************
    ## Number of grey level bins in each resolution level,
    ## for the mutual information. 16 or 32 usually works fine.
    ## You could also employ a hierarchical strategy:
    ##(NumberOfHistogramBins 16 32 64)
    rigid_params["NumberOfHistogramBins"] = ["32"]
    ## If you use a mask, this option is important.
    ## If the mask serves as region of interest, set it to false.
    ## If the mask indicates which pixels are valid, then set it to true.
    ## If you do not use a mask, the option doesn't matter.
    rigid_params["ErodeMask"] = ["false"]
    ## ******************** Multiresolution **********************
    ## The number of resolutions. 1 Is only enough if the expected
    ## deformations are small. 3 or 4 mostly works fine. For large
    ## images and large deformations, 5 or 6 may even be useful.
    rigid_params["NumberOfResolutions"] = ["8"]
    ##(FinalGridSpacingInVoxels 8.0 8.0)
    ##(GridSpacingSchedule 6.0 6.0 4.0 4.0 2.5 2.5 1.0 1.0)
    ## The downsampling/blurring factors for the image pyramids.
    ## By default, the images are downsampled by a factor of 2
    ## compared to the next resolution.
    ## So, in 2D, with 4 resolutions, the following schedule is used:
    ##(ImagePyramidSchedule 4 4  2 2  1 1 )
    ## And in 3D:
    ##(ImagePyramidSchedule 8 8 8  4 4 4  2 2 2  1 1 1 )
    ## You can specify any schedule, for example:
    ##(ImagePyramidSchedule 4 4  4 3  2 1  1 1 )
    ## Make sure that the number of elements equals the number
    ## of resolutions times the image dimension.
    ## ******************* Optimizer ****************************
    ## Maximum number of iterations in each resolution level:
    ## 200-500 works usually fine for rigid registration.
    ## For more robustness, you may increase this to 1000-2000.
    ## 80 good results, 7 minutes on basalis with 4 jobs
    ## 200 good results except for 1st couple were not aligned, 12 minutes
    ## 500 is best, including first sections, basalis took 21 minutes
    rigid_params["MaximumNumberOfIterations"] = [NUM_ITERATIONS]
    ## The step size of the optimizer, in mm. By default the voxel size is used.
    ## which usually works well. In case of unusual high-resolution images
    ## (eg histology) it is necessary to increase this value a bit, to the size
    ## of the "smallest visible structure" in the image:
    ##(MaximumStepLength 4)
    ## **************** Image sampling **********************
    ## Number of spatial samples used to compute the mutual
    ## information (and its derivative) in each iteration.
    ## With an AdaptiveStochasticGradientDescent optimizer,
    ## in combination with the two options below, around 2000
    ## samples may already suffice.
    ##(NumberOfSpatialSamples 2048)
    ## Refresh these spatial samples in every iteration, and select
    ## them randomly. See the manual for information on other sampling
    ## strategies.
    rigid_params["NewSamplesEveryIteration"] = ["true"]
    rigid_params["ImageSampler"] = ["Random"]
    ## ************* Interpolation and Resampling ****************
    ## Order of B-Spline interpolation used during registration/optimisation.
    ## It may improve accuracy if you set this to 3. Never use 0.
    ## An order of 1 gives linear interpolation. This is in most
    ## applications a good choice.
    rigid_params["BSplineInterpolationOrder"] = ["1"]
    ## Order of B-Spline interpolation used for applying the final
    ## deformation.
    ## 3 gives good accuracy; recommended in most cases.
    ## 1 gives worse accuracy (linear interpolation)
    ## 0 gives worst accuracy, but is appropriate for binary images
    ## (masks, segmentations); equivalent to nearest neighbor interpolation.
    rigid_params["FinalBSplineInterpolationOrder"] = ["3"]
    ##Default pixel value for pixels that come from outside the picture:
    rigid_params["DefaultPixelValue"] = ["0"]
    ## Choose whether to generate the deformed moving image.
    ## You can save some time by setting this to false, if you are
    ## only interested in the final (nonrigidly) deformed moving image
    ## for example.
    rigid_params["WriteResultImage"] = ["false"]
    ## The pixel type and format of the resulting deformed moving image
    rigid_params["ResultImagePixelType"] = ["float"]
    rigid_params["ResultImageFormat"] = ["tif"]
    rigid_params["RequiredRatioOfValidSamples"] = ["0.05"]
    return rigid_params


def align_elastix(fixed, moving):
    """This takes the moving and fixed images runs Elastix on them. Note
    the huge list of parameters Elastix uses here.

    :param fixed: sitk float array for the fixed image (the image behind the moving).
    :param moving: sitk float array for the moving image.
    :return: the Elastix transformation results that get parsed into the rigid transformation
    """
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetMovingImage(moving)
    translationMap = elastixImageFilter.GetDefaultParameterMap("translation")

    rigid_params = create_rigid_parameters(elastixImageFilter)
    elastixImageFilter.SetParameterMap(translationMap)
    elastixImageFilter.AddParameterMap(rigid_params)
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.Execute()
    
    translations = elastixImageFilter.GetTransformParameterMap()[0]["TransformParameters"]
    rigid = elastixImageFilter.GetTransformParameterMap()[1]["TransformParameters"]
    x1,y1 = translations
    R,x2,y2 = rigid
    x = float(x1) + float(x2)
    y = float(y1) + float(y2)
    return float(R), float(x), float(y)


def create_downsampled_transforms(transforms: dict, downsample: bool) -> dict:
    """Changes the dictionary of transforms to the correct resolution


    :param animal: prep_id of animal we are working on animal
    :param transforms: dictionary of filename:array of transforms
    :param downsample: boolean: either true for thumbnails, false for full resolution images
    :return: corrected dictionary of filename: array  of transforms
    """

    if downsample:
        transforms_scale_factor = 1
    else:
        transforms_scale_factor = SCALING_FACTOR

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
    except Exception as e:
        print(f'align image to affine, could not open {infile} error {e}')
        return
    finally:
         gc.collect()

    try:
        im2 = im1.transform((im1.size), Image.Transform.AFFINE, T.flatten()[:6], resample=Image.Resampling.NEAREST)
    except Exception as e:
        print(f'align image to affine, could not transform {infile} to {outfile}')
        return
    finally:
         gc.collect()
    
    del im1

    try:
        im2.save(outfile, compression=None)
    except Exception as e:
        print(f'align image to affine, could not save {outfile}')
        print(f'Error={e}')
        return
    finally:
         gc.collect()

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



