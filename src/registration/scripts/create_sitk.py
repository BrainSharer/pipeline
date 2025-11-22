
# simpleitk_affine_registration.py
import os
import numpy as np
import SimpleITK as sitk


def override_spacing_if_requested(img, spacing_tuple):
    # spacing_tuple is in same units as image (microns here)
    img = sitk.Image(img)  # ensure a copy
    img.SetSpacing(spacing_tuple)
    return img

def initialize_affine(fixed, moving):
    # Centered initializer: aligns centers/geometry to help optimizer
    initial = sitk.CenteredTransformInitializer(fixed,
                                               moving,
                                               sitk.AffineTransform(3),
                                               sitk.CenteredTransformInitializerFilter.GEOMETRY)
    return initial

def run_affine_registration(fixed, moving, initial_transform):
    registration = sitk.ImageRegistrationMethod()

    # Metric
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.1)  # sample 10% of voxels for speed

    # Interpolator
    registration.SetInterpolator(sitk.sitkLinear)

    # Optimizer
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,
                                                 minStep=1e-6,
                                                 numberOfIterations=1000,
                                                 gradientMagnitudeTolerance=1e-8)
    registration.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution framework (shrink factors and smoothing)
    registration.SetShrinkFactorsPerLevel(shrinkFactors = [8, 4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Initial transform
    registration.SetInitialTransform(initial_transform, inPlace=False)

    # Execute
    final_transform = registration.Execute(fixed, moving)

    print("Optimizer stop condition:", registration.GetOptimizerStopConditionDescription())
    print("Final metric value:", registration.GetMetricValue())

    return final_transform


def transform_fiducials_indices_to_fixed(indices_array, moving_img, transform, fixed_img):
    """
    indices_array: Nx3 array of voxel indices in moving image (0-based)
    moving_img: SimpleITK image (used to map index -> physical)
    transform: SimpleITK transform mapping moving physical -> fixed physical
    fixed_img: SimpleITK image (used to convert physical -> index)
    Returns: list of dicts with moving_index, moving_phys, fixed_phys, fixed_index
    """
    results = []
    for idx in indices_array:
        # ensure float
        idx = [float(x) for x in idx]
        moving_phys = moving_img.TransformContinuousIndexToPhysicalPoint(idx)
        fixed_phys = transform.TransformPoint(moving_phys)  # physical point in fixed space
        # Convert physical point to nearest fixed image index (integer)
        try:
            fixed_index = fixed_img.TransformPhysicalPointToIndex(fixed_phys)
        except RuntimeError:
            # outside fixed image bounds -> put as None or closest index
            fixed_index = None

        results.append({
            'moving_index': tuple(idx),
            'moving_physical': tuple(moving_phys),
            'fixed_physical': tuple(fixed_phys),
            'fixed_index': fixed_index
        })
    return results

def save_transformed_fiducials(results, out_csv):
    rows = []
    for r in results:
        mi = r['moving_index']
        mp = r['moving_physical']
        fp = r['fixed_physical']
        fi = r['fixed_index']
        rows.append({
            'moving_i': mi[0], 'moving_j': mi[1], 'moving_k': mi[2],
            'moving_phys_x': mp[0], 'moving_phys_y': mp[1], 'moving_phys_z': mp[2],
            'fixed_phys_x': fp[0], 'fixed_phys_y': fp[1], 'fixed_phys_z': fp[2],
            'fixed_i': None if fi is None else fi[0],
            'fixed_j': None if fi is None else fi[1],
            'fixed_k': None if fi is None else fi[2]
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Transformed fiducials written to: {out_csv}")



def save_transform(transform, path):
    sitk.WriteTransform(transform, path)
    print(f"Transform saved to: {path}")

if __name__ == "__main__":
    # ---------------- USER INPUT ----------------
    um = 25.0
    moving_brain = "DK55"
    regpath = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration"
    output_transform_path = os.path.join(regpath, moving_brain, f"{moving_brain}_to_Allen_affine.tfm")
    fixed_image_path = f"{regpath}/Allen/Allen_{um}x{um}x{um}um_sagittal.nii"      # path to Allen reference (10 um isotropic)
    moving_image_path = os.path.join(regpath, moving_brain, "DK55_10.4x10.4x20um_sagittal.nii")
    assert os.path.exists(fixed_image_path), f"Fixed image not found: {fixed_image_path}"
    assert os.path.exists(moving_image_path), f"Moving image not found: {moving_image_path}"    

    print(f"Loading moving image from {moving_image_path}")
    moving = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    print("Moving image size (x,y,z):", moving.GetSize(), "spacing:", moving.GetSpacing())

    print(f"Loading fixed (Allen) image from {fixed_image_path}")
    fixed = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    print("Fixed image size (x,y,z):", fixed.GetSize(), "spacing:", fixed.GetSpacing())
    moving_fiducials = [
        (1062, 1062, 130),
        (1311, 644, 240)
    ]

    moving_spacing_override = (10.4, 10.4, 20.0)  # microns (same units as fixed image assumed)
    moving.SetSpacing(moving_spacing_override)


    # Initialize and run affine registration
    init = initialize_affine(fixed, moving)
    print("Running affine registration (this may take minutes depending on image size)...")
    final_transform = run_affine_registration(fixed, moving, init)
    resampled = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID(),
    )

    resampled_path = os.path.join(regpath, moving_brain, f"{moving_brain}_to_Allen_affine_resampled.nii")
    sitk.WriteImage(resampled, resampled_path)
    print(f"Resampled moving image written to {resampled_path}")


    # Save transform
    save_transform(final_transform, output_transform_path)

    # Load fiducials and transform them
    indices_array = moving_fiducials
    results = transform_fiducials_indices_to_fixed(indices_array, moving, final_transform, fixed)
    for result in results:
        print(result)


