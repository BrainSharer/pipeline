import argparse
import os
import numpy as np
import tifffile as tiff
import SimpleITK as sitk
import glob
from tqdm import tqdm


def preprocess(img):
    # Normalize intensities
    img = sitk.Normalize(img)
    
    # Optional smoothing (helps with damaged tissue)
    img = sitk.DiscreteGaussian(img, variance=1.0)
    
    return img

def detect_image_type(img: np.ndarray):
    """
    Detect grayscale vs RGB and bit depth.
    """
    if img.ndim == 2:
        color = "grayscale"
    elif img.ndim == 3 and img.shape[-1] in [3, 4]:
        color = "rgb"
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    if img.dtype == np.uint8:
        bit_depth = "8-bit"
    elif img.dtype == np.uint16:
        bit_depth = "16-bit"
    else:
        raise ValueError(f"Unsupported dtype: {img.dtype}")

    return color, bit_depth


def register_slice(fixed_sitk, moving_sitk):
    """
    Register moving image to fixed image using SimpleITK.
    """
    registration_method = sitk.ImageRegistrationMethod()

    # Metric
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.2)

    # Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=500,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Transform
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_sitk,
        moving_sitk,
        sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Multi-resolution
    registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_transform = registration_method.Execute(fixed_sitk, moving_sitk)

    # Resample
    resampled = sitk.Resample(
        moving_sitk,
        fixed_sitk,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_sitk.GetPixelID(),
    )

    return resampled


def register_tiff_stack(input_paths, output_dir):
    """
    Register a stack of TIFF images to the middle slice.

    Parameters:
        input_paths (list): List of TIFF file paths
        output_dir (str): Directory to save registered images
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load all images
    #images = [tiff.imread(p) for p in input_paths]

    # Detect type from first image
    #color, bit_depth = detect_image_type(images[0])
    #print(f"Detected: {color}, {bit_depth}")

    # Convert all to SITK
    #sitk_images = [to_sitk(img) for img in images]

    # Choose middle slice as fixed
    mid_idx = len(input_paths) // 2
    fixed_path = input_paths[mid_idx]
    print(f"Using middle slice as fixed image: {fixed_path}")
    exit(0)
    fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)

    registered_images = []

    for i, moving_path_ in enumerate(tqdm(input_paths)):
        #print(f"Registering slice {i} → {mid_idx} from path: {moving_path_}")
        moving = sitk.ReadImage(moving_path_, sitk.sitkFloat32)

        if i == mid_idx:
            registered = moving
        else:
            registered = register_slice(fixed, moving)

        registered_np = sitk.GetArrayFromImage(registered)
        out_path = os.path.join(output_dir, str(i).zfill(3) + ".tif")
        tiff.imwrite(out_path, registered_np.astype(np.uint16))

    print(f"Saved {len(registered_images)} registered images to {output_dir}")



def register_sagittal_slices(
    tiff_dir,
    output_dir,
    reference_index=None):
    """
    Register sagittal TIFF slices into a coherent 3D volume.

    Parameters
    ----------
    tiff_dir : str
        Directory containing TIFF slices.
    output_dir : str
        Directory to save registered images.
    reference_index : int or None
        If None, uses sequential registration. Otherwise registers all slices to this reference.

    Returns
    -------
    volume_sitk : sitk.Image
        Registered 3D volume.
    volume_np : np.ndarray
        Registered volume as NumPy array (z, y, x).
    """

    # Load sorted TIFF files
    files = sorted(glob.glob(os.path.join(tiff_dir, "*.tif")))
    if len(files) == 0:
        raise ValueError("No TIFF files found.")

    print(f"Loaded {len(files)} slices")

    # Read images
    images = [sitk.ReadImage(f, sitk.sitkFloat32) for f in files]

    # Initialize registration method
    def create_registration_method():
        R = sitk.ImageRegistrationMethod()

        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(0.5)

        R.SetInterpolator(sitk.sitkLinear)

        # from elastix_manager
        R.SetOptimizerAsRegularStepGradientDescent(
            learningRate=2,
            minStep=1e-4,
            numberOfIterations=250,
            gradientMagnitudeTolerance=1e-8
        )
        R.SetOptimizerScalesFromPhysicalShift()

        # Interpolator
        R.SetInterpolator(sitk.sitkLinear)

        # Initial transform
        R.SetInitialTransform(initial_transform, inPlace=False)
        R.SetShrinkFactorsPerLevel([4, 2, 1])
        R.SetSmoothingSigmasPerLevel([2, 1, 0])
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()


        return R

    registered_images = []

    # Reference slice
    if reference_index is not None:
        fixed_image = images[reference_index]
        registered_images = [None] * len(images)

        for i, moving_image in enumerate(images):
            print(f"Registering slice {i} to reference {reference_index}")

            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image,
                moving_image,
                sitk.Euler2DTransform(2),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )

            R = create_registration_method()
            R.SetInitialTransform(initial_transform, inPlace=False)

            final_transform = R.Execute(fixed_image, moving_image)

            resampled = sitk.Resample(
                moving_image,
                fixed_image,
                final_transform,
                sitk.sitkLinear,
                0.0,
                moving_image.GetPixelID(),
            )

            registered_images[i] = resampled

    else:
        # Sequential registration
        registered_images.append(images[0])  # first slice unchanged

        for i in tqdm(range(1, len(images))):
            fixed_image = registered_images[i - 1]
            moving_image = images[i]
            #fixed_image = preprocess(fixed_image)
            #moving_image = preprocess(moving_image)
            matcher = sitk.HistogramMatchingImageFilter()
            matcher.SetNumberOfHistogramLevels(256)
            matcher.SetNumberOfMatchPoints(10)
            matcher.ThresholdAtMeanIntensityOn()
            moving_image = matcher.Execute(moving_image, fixed_image)
            #print(f"Registering slice {i} to slice {i-1}")

            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image,
                moving_image,
                sitk.Euler2DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )

            R = create_registration_method()
            R.SetInitialTransform(initial_transform, inPlace=False)

            final_transform = R.Execute(fixed_image, moving_image)

            resampled = sitk.Resample(
                moving_image,
                fixed_image,
                final_transform,
                sitk.sitkLinear,
                0.0,
                moving_image.GetPixelID(),
            )

            registered_images.append(resampled)
            registered_np = sitk.GetArrayFromImage(resampled)
            out_path = os.path.join(output_dir, str(i-1).zfill(3) + ".tif")
            #print(f"Saving registered slice {i-1} to {out_path}")
            #print(f"Registered slice {i-1} shape: {registered_np.shape}, dtype: {registered_np.dtype}")
            tiff.imwrite(out_path, registered_np.astype(np.uint16))


    # Stack into 3D volume
    volume = sitk.JoinSeries(registered_images)

    # Convert to numpy (z, y, x)
    volume_np = sitk.GetArrayFromImage(volume)

    return volume, volume_np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)

    args = parser.parse_args()
    animal = args.animal
    prep_path = f"/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps"
    image_path = os.path.join(prep_path, "C1")
    input_directory = os.path.join(image_path, "thumbnail_cleaned")
    if not os.path.exists(input_directory):
        raise ValueError(f"Input directory does not exist: {input_directory}")
    output_directory = os.path.join(image_path, "registered_to_mid")
    os.makedirs(output_directory, exist_ok=True)
    input_files = sorted(glob.glob(os.path.join(input_directory, "*.tif")))
    volume_sitk, volume_np = register_sagittal_slices(input_directory, output_directory,reference_index=None)

    print(volume_np.shape)
