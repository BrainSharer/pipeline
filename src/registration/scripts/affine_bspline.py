
import os
import argparse
import SimpleITK as sitk
import zarr
from tqdm import tqdm
import SimpleITK as sitk


def affine_bspline_registration(
    fixed_image,
    moving_image,
    affine_transform,
    grid_physical_spacing=(100.0, 100.0, 100.0),
    number_of_iterations=100,
):
    """
    Refine an existing affine registration with a 3D B-spline transform.

    Parameters
    ----------
    fixed_image : sitk.Image
        Fixed/reference 3D image.

    moving_image : sitk.Image
        Moving 3D image.

    affine_transform : sitk.AffineTransform
        Existing affine transform mapping moving -> fixed.

    grid_physical_spacing : tuple
        Approximate physical spacing between B-spline control points
        in x, y, z. Units must match image spacing (e.g. microns).

        Larger values:
            smoother deformation, fewer parameters.

        Smaller values:
            more local deformation, more parameters.

    number_of_iterations : int
        Number of B-spline optimization iterations.

    Returns
    -------
    final_transform : sitk.CompositeTransform
        Affine + B-spline composite transform.

    bspline_transform : sitk.BSplineTransform
        Optimized B-spline transform.
    """

    # ---------------------------------------------------------
    # 1. Create B-spline transform
    # ---------------------------------------------------------

    image_dimension = fixed_image.GetDimension()

    if image_dimension != 3:
        raise ValueError("This function expects 3D images.")

    fixed_size = fixed_image.GetSize()
    fixed_spacing = fixed_image.GetSpacing()

    # Number of B-spline control points.
    #
    # The mesh size is calculated from the physical size of
    # the fixed image and the requested control-point spacing.
    mesh_size = [
        max(
            1,
            int(
                fixed_size[i]
                * fixed_spacing[i]
                / grid_physical_spacing[i]
            ),
        )
        for i in range(3)
    ]

    print("Fixed size:", fixed_size)
    print("Fixed spacing:", fixed_spacing)
    print("B-spline mesh size:", mesh_size)

    bspline_transform = sitk.BSplineTransformInitializer(
        fixed_image,
        mesh_size,
        order=3,
    )

    # ---------------------------------------------------------
    # 2. Set up registration
    # ---------------------------------------------------------

    registration = sitk.ImageRegistrationMethod()

    # Mattes mutual information is generally a good choice
    # for microscopy images with intensity differences.
    registration.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=50
    )

    # Sampling is important for large 3D volumes.
    registration.SetMetricSamplingStrategy(
        registration.RANDOM
    )

    registration.SetMetricSamplingPercentage(
        0.01
    )

    registration.SetInterpolator(
        sitk.sitkLinear
    )

    # ---------------------------------------------------------
    # 3. Optimizer
    # ---------------------------------------------------------

    registration.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=number_of_iterations,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e7,
    )

    # ---------------------------------------------------------
    # 4. Use the existing affine transform
    # ---------------------------------------------------------

    # Important:
    #
    # We do NOT optimize the affine transform again.
    #
    # The affine is treated as the already-established
    # global transformation, while the B-spline is optimized
    # to account for local deformation.

    composite_transform = sitk.CompositeTransform(3)

    composite_transform.AddTransform(
        affine_transform
    )

    composite_transform.AddTransform(
        bspline_transform
    )

    registration.SetInitialTransform(
        composite_transform,
        inPlace=True,
    )

    # ---------------------------------------------------------
    # 5. Run B-spline optimization
    # ---------------------------------------------------------

    print("Starting B-spline registration...")

    final_transform = registration.Execute(
        fixed_image,
        moving_image,
    )

    print(
        "Final metric value:",
        registration.GetMetricValue()
    )

    print(
        "Optimizer iterations:",
        registration.GetOptimizerIteration()
    )

    # ---------------------------------------------------------
    # 6. Return transforms
    # ---------------------------------------------------------

    return final_transform, bspline_transform



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--moving', help='Enter the animal (moving)', required=True, type=str)
    parser.add_argument('--fixed', help='Enter the animal (fixed)', required=True, type=str)
    parser.add_argument("--task", help="Enter the task you want to perform", required=False, default="status", type=str)
    parser.add_argument("--downsample", help="Enter the downsample", required=False, default=32, type=int)
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)

    args = parser.parse_args()
    moving = args.moving
    fixed = args.fixed
    task = str(args.task).strip().lower()
    downsample = args.downsample
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])


    base_path = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data"
    scratch_path = "/data/pipeline_tmp"
    reg_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration"
    moving_zarr_path = os.path.join(scratch_path, moving, f'source.{downsample}.zarr')
    fixed_zarr_path = os.path.join(scratch_path, fixed, f'source.{downsample}.zarr')
    moving_sitk_path = os.path.join(scratch_path, moving, f'source.{downsample}.nii')
    fixed_sitk_path = os.path.join(scratch_path, fixed, f'source.{downsample}.nii')
    output_volume = os.path.join(scratch_path, moving, f'{moving}_{fixed}_registered.{downsample}.nii')
    xy_resolution = 0.325*downsample
    z_resolution = 20.0
    transform_path = os.path.join(reg_path, f"{moving}_{fixed}.tfm")

    if not os.path.exists(moving_zarr_path):
        print(f'Missing input zarr: {moving_zarr_path}')
        exit(0)
    if not os.path.exists(fixed_zarr_path):
        print(f'Missing fixed zarr: {fixed_zarr_path}')
        exit(0)
    if not os.path.exists(transform_path):
        print(f'Missing input zarr: {transform_path}')
        exit(0)

    if os.path.exists(moving_sitk_path):
        moving_image = sitk.ReadImage(moving_sitk_path,sitk.sitkFloat32)
    else:
        moving_zarr = zarr.open(moving_zarr_path, mode='r')
        moving_image = sitk.GetImageFromArray(moving_zarr)
        moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
        moving_image.SetSpacing((xy_resolution, xy_resolution, z_resolution))
        sitk.WriteImage(moving_image, moving_sitk_path)

    if os.path.exists(fixed_sitk_path):
        fixed_image = sitk.ReadImage(fixed_sitk_path,sitk.sitkFloat32)
    else:
        fixed_zarr = zarr.open(fixed_zarr_path, mode='r')
        fixed_image = sitk.GetImageFromArray(fixed_zarr)
        fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
        fixed_image.SetSpacing((xy_resolution, xy_resolution, z_resolution))
        sitk.WriteImage(fixed_image, fixed_sitk_path)

    print(f'Moving spacing: {moving_image.GetSpacing()} depth: {moving_image.GetDepth()}')
    print(f'Moving spacing: {fixed_image.GetSpacing()} depth: {fixed_image.GetDepth()}')

    # ------------------------------------------------------------
    # Load the affine transform that was generated during
    # registration.
    # ------------------------------------------------------------

    affine = sitk.ReadTransform(transform_path)



    final_transform, bspline = affine_bspline_registration(
        fixed_image=fixed_image,
        moving_image=moving_image,
        affine_transform=affine,
        grid_physical_spacing=(200.0, 200.0, 400.0),
        number_of_iterations=200,
    )

    # Save the transforms
    #sitk.WriteTransform(final_transform, "affine_bspline.tfm",)
    #sitk.WriteTransform(bspline,"bspline.tfm",)

    resampled = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkLinear,
        0,
        sitk.sitkFloat32,
    )

    resultImage = sitk.Cast(sitk.RescaleIntensity(resampled), sitk.sitkUInt16)
    sitk.WriteImage(resultImage, output_volume)
    print(f'Wrote image to {output_volume}')
