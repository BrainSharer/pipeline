import SimpleITK as sitk
import glob
import os


def read_tiff_stack(folder):
    """
    Read a folder of TIFF images into a 3D SimpleITK image.
    The images must all be the same size and sorted properly.
    """
    file_names = sorted(glob.glob(os.path.join(folder, "*.tif")))
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    volume = reader.Execute()
    return volume


def register_affine(fixed, moving, output_path="affine_registered.nii.gz"):
    """
    Perform affine registration of moving -> fixed using SimpleITK.
    """
    # Cast to float (registration requires float images)
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    # Initialize transform
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.AffineTransform(fixed.GetDimension()),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # Setup registration method
    registration_method = sitk.ImageRegistrationMethod()
    
    # Similarity metric
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2)
    
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution strategy
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Run registration
    final_transform = registration_method.Execute(fixed, moving)

    print("Optimizer stop condition: {0}".format(registration_method.GetOptimizerStopConditionDescription()))
    print("Final metric value: {0}".format(registration_method.GetMetricValue()))
    print("Final transform: \n", final_transform)

    # Resample moving image
    moving_resampled = sitk.Resample(moving, fixed, final_transform,
                                     sitk.sitkLinear, 0.0, moving.GetPixelID())

    # Save results
    sitk.WriteImage(moving_resampled, output_path)

    return moving_resampled, final_transform


if __name__ == "__main__":
    # Paths to TIFF stacks (folders containing sagittal images)
    data_path = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data'
    fixed_folder = os.path.join(data_path, "MD594", "preps/C1/thumbnail_aligned")
    moving_folder = os.path.join(data_path, "MD589", "preps/C1/thumbnail_aligned")

    print("Reading fixed image stack...")
    fixed_img = read_tiff_stack(fixed_folder)

    print("Reading moving image stack...")
    moving_img = read_tiff_stack(moving_folder)

    print("Starting affine registration...")
    output_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/sc_test.tif"
    registered_img, transform = register_affine(fixed_img, moving_img, output_path)

    print("Registration complete. Output saved as affine_registered.nii.gz")
