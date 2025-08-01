import SimpleITK as sitk
import numpy as np
import os

def register_3d_images(fixed_path, moving_path):
    # Load fixed and moving images
    fixed_image = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    print(f"Read fixed image: {fixed_path}")
    moving_image = sitk.ReadImage(moving_path, sitk.sitkFloat32)
    print(f"Read moving image: {moving_path}")

    # Initial alignment of the centers of the two volumes
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, 
        moving_image, 
        sitk.Euler3DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # Set up the registration method
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Perform registration
    final_transform = registration_method.Execute(fixed_image, moving_image)
    print("Registration completed.")
    return final_transform

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
    reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
    fixed_image_path = os.path.join(reg_path, 'Allen', 'Allen_10.0x10.0x10.0um_sagittal.tif')
    moving_image_path = os.path.join(reg_path, 'ALLEN771602', 'ALLEN771602_10.0x10.0x10.0um_sagittal.tif')

    # Register the images
    transform = register_3d_images(fixed_image_path, moving_image_path)

    # Define points in moving image space
    moving_points = np.array(
        [
            [1224.9345891177654, 180.99660519510508, 326.6999963670969],
            [1185.2512136101723, 206.79081790149212, 326.6999963670969],
            [1187.2353963553905, 307.9835791140795, 326.6999963670969],
            [1215.0137685239315, 325.8411306887865, 326.6999963670969],
            [1195.1720342040062, 349.65118393301964, 326.6999963670969],
            [1185.2512136101723, 436.9547590613365, 326.6999963670969],
            [1240.8080510795116, 528.2266531139612, 326.6999963670969],
            [1278.507336974144, 542.1158391982317, 326.6999963670969],
            [1310.253981500864, 506.40078261494637, 326.6999963670969],
            [1320.1748952269554, 454.8122640699148, 326.6999963670969],
            [1292.3965230584145, 413.1447058171034, 326.6999963670969],
            [1240.8080510795116, 393.3029714971781, 326.6999963670969],
            [1242.79223382473, 383.3821043372154, 326.6999963670969],
            [1284.4597920775414, 373.46123717725277, 326.6999963670969],
            [1300.33316090703, 397.2712904214859, 326.6999963670969],
            [1361.8424534797668, 385.3662870824337, 326.6999963670969],
            [1395.5733738839626, 345.6828650087118, 326.6999963670969],
            [1383.668463677168, 311.95192132145166, 326.6999963670969],
            [1395.5733738839626, 278.22102420032024, 326.6999963670969],
            [1353.9058156311512, 218.69586780667305, 326.6999963670969],
            [1316.2066228687763, 226.63255222141743, 326.6999963670969],
            [1302.3173436522484, 200.83831623196602, 326.6999963670969],
        ]
    )

    # Apply transform to points
    fixed_points = transform_points(moving_points, transform)

    print("Transformed Points in Fixed Space:\n", fixed_points)
