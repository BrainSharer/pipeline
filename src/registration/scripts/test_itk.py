import SimpleITK as sitk
import numpy as np
import os


def register_3d_images(fixed_path, moving_path, xy_um, z_um):
    # Load fixed and moving images
    fixed_image = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    print(f"Read fixed image: {fixed_path}")
    moving_image = sitk.ReadImage(moving_path, sitk.sitkFloat32)
    print(f"Read moving image: {moving_path}")

    fixed_image_size = fixed_image.GetSize()
    moving_image_size = moving_image.GetSize()


    # Initial alignment of the centers of the two volumes
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, 
        moving_image, 
        sitk.AffineTransform(3), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # Set up the registration method
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50) # [44.27027195 39.20937542 -0.23252082]

    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsGradientDescent(
        learningRate=1, 
        numberOfIterations=300, # changing this has very little
        convergenceMinimumValue=1e-6, 
        convergenceWindowSize=10)

    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(initial_transform, inPlace=False)
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Perform registration
    transform = R.Execute(fixed_image, moving_image)
    reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
    print("Final metric value: ", R.GetMetricValue())
    print("Optimizer's stopping condition: ", R.GetOptimizerStopConditionDescription())
    output_image_path = os.path.join(reg_path, 'ALLEN771602', f'ALLEN771602_Allen_{z_um}x{xy_um}x{xy_um}um_sagittal.tif')
    # Resample moving image onto fixed image grid
    resampled = sitk.Resample(moving_image, fixed_image, transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    sitk.WriteImage(resampled, output_image_path)
    print(f"Resampled moving image written to {output_image_path}")



    output_file_path = os.path.join(reg_path, 'ALLEN771602', f'ALLEN771602_{z_um}x{xy_um}x{xy_um}um_sagittal.tfm')
    # Save the transform
    sitk.WriteTransform(transform, output_file_path)
    print(f"Registration written to {output_file_path}")
    # do inverse
    inverse_transform = transform.GetInverse()
    reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
    output_file_path = os.path.join(reg_path, 'ALLEN771602', f'ALLEN771602_{z_um}x{xy_um}x{xy_um}um_sagittal_inverse.tfm')
    # Save the transform
    sitk.WriteTransform(inverse_transform, output_file_path)
    print(f"Registration written to {output_file_path}")
    return transform, inverse_transform, fixed_image_size, moving_image_size

def transform_points(points_xyz, transform):
    """
    Apply a SimpleITK transform to a list of (x, y, z) points.

    :param points_xyz: Nx3 numpy array of points
    :param transform: SimpleITK.Transform object
    :return: Nx3 numpy array of transformed points
    """
    transformed_points = [transform.TransformPoint(p) for p in points_xyz]
    return np.array(transformed_points)

# Example usage
if __name__ == "__main__":
    # Paths to fixed and moving 3D images
    xy_um = 28.8
    z_um = 32.0
    reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
    fixed_image_path = os.path.join(reg_path, 'Allen', f'Allen_{z_um}x{xy_um}x{xy_um}um_sagittal.tif')
    moving_image_path = os.path.join(reg_path, 'ALLEN771602', f'ALLEN771602_{z_um}x{xy_um}x{xy_um}um_sagittal.tif')

    # Register the images
    transform, inverse_transform, fixed_image_size, moving_image_size = register_3d_images(fixed_image_path, moving_image_path, xy_um, z_um)
    fixed_midpoint = np.array(fixed_image_size) / 2
    moving_midpoint = np.array(moving_image_size) / 2  #
    print(f'Fixed midpoint: {fixed_midpoint}, Moving midpoint: {moving_midpoint}')
    inverse_transformed_point = inverse_transform.TransformPoint(moving_midpoint.tolist())
    print("Inverse Transformed Points using inverse:\n", inverse_transformed_point)
    print(f'Difference between fixed and transformed moving midpoints: {fixed_midpoint},  {np.array(inverse_transformed_point)}')
    print(f'Difference between fixed and transformed moving midpoints: {fixed_midpoint - np.array(inverse_transformed_point)}')

