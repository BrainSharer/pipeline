from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import os
import numpy as np
import SimpleITK as sitk
from tqdm.contrib import tzip

def read_tif_stack(directory: str) -> List[sitk.Image]:
    """
    Read TIFF slices in sorted filename order.

    Returns
    -------
    list[sitk.Image]
        One 2D SimpleITK image per sagittal section.
    """
    directory = Path(directory)

    files = sorted(list(directory.glob("*.tif")))

    if not files:
        raise FileNotFoundError(f"No TIFF files found in {directory}")

    images = []

    for filename in files:
        image = sitk.ReadImage(str(filename))
        # Convert RGB/vector images to grayscale.
        if image.GetNumberOfComponentsPerPixel() > 1:
            image = sitk.VectorIndexSelectionCast(image, 0)

        image = sitk.Cast(image, sitk.sitkFloat32)
        images.append(image)

    return images


def register_adjacent(fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
    """
    Register moving to fixed using a rigid 2D transformation.
    """

    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    initial_transform = sitk.Euler2DTransform()
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        initial_transform,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()
    # Mutual information is generally robust for microscopy intensity
    registration.SetMetricAsMattesMutualInformation()
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.1)
    registration.SetInterpolator(sitk.sitkLinear)
    # Optimizer settings.
    registration.SetOptimizerAsGradientDescent(
        learningRate=1,
        numberOfIterations=250,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration.Execute(fixed,moving)

    print(f"Stopping condition, {registration.GetOptimizerStopConditionDescription()}", end=" ")
    print(f"metric={registration.GetMetricValue():.6f}, iterations={registration.GetOptimizerIteration()}")

    return final_transform


def compose_transforms(transforms: List[sitk.Transform]) -> sitk.CompositeTransform:
    """
    Compose a sequence of transforms.

    The transforms are applied in sequence:

        T0 -> T1 -> T2 -> ...

    """
    composite = sitk.CompositeTransform(2)

    for transform in transforms:
        composite.AddTransform(transform)

    return composite


def register_serial_stack(
    images: List[sitk.Image],
    reference_index: int = 0,
) -> Tuple[
    List[sitk.Transform],
    List[sitk.CompositeTransform],
]:
    """
    Register every serial section to its adjacent section and construct
    accumulated transforms to the reference section.

    Parameters
    ----------
    images
        Ordered sagittal sections.

    reference_index
        Index of the reference section.

    Returns
    -------
    adjacent_transforms
        Pairwise transforms.

    accumulated_transforms
        Transform for every section mapping it to the reference
        coordinate system.
    """

    n = len(images)

    if n == 0:
        raise ValueError("No images supplied.")

    if not (0 <= reference_index < n):
        raise ValueError("Invalid reference index.")

    adjacent_transforms = [None] * n
    accumulated_transforms = [None] * n
    # Reference section has identity transform.
    identity = sitk.Euler2DTransform()
    identity.SetIdentity()
    accumulated_transforms[reference_index] = (sitk.CompositeTransform(identity))

    # ------------------------------------------------------------
    # Register sections BEFORE the reference.
    #
    # Example:
    #
    #   3 -> 2 -> 1 -> 0
    #
    # ------------------------------------------------------------

    accumulated = sitk.CompositeTransform(2)

    for i in range(reference_index - 1, -1, -1):
        fixed = images[i + 1]
        moving = images[i]
        print(f"Registering slice {i} -> slice {i + 1}", end=" ")
        transform = register_adjacent(fixed=fixed, moving=moving,)
        adjacent_transforms[i] = transform
        # transform maps i -> i+1.
        accumulated.AddTransform(transform)
        accumulated_transforms[i] = sitk.CompositeTransform(accumulated)

    # ------------------------------------------------------------
    # Register sections AFTER the reference.
    #
    # Example:
    #
    #   0 -> 1 -> 2 -> 3
    #
    # ------------------------------------------------------------

    accumulated = sitk.CompositeTransform(2)

    for i in range(reference_index + 1, n):
        fixed = images[i - 1]
        moving = images[i]
        print(f"Registering slice {i} -> slice {i - 1}", end=" ")
        transform = register_adjacent(fixed=fixed, moving=moving,)
        adjacent_transforms[i] = transform
        accumulated.AddTransform(transform)
        accumulated_transforms[i] = sitk.CompositeTransform(accumulated)

    return adjacent_transforms, accumulated_transforms


def resample_stack(images: List[sitk.Image], transforms: List[sitk.Transform],reference_image: sitk.Image) -> List[sitk.Image]:
    """
    Resample all sections into the reference coordinate system.
    """

    registered = []

    print(f"Resampling slices")
    for (image, transform) in tzip(images, transforms):
        result = sitk.Resample(
            image,
            reference_image,
            transform,
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )
        registered.append(result)

    return registered


def save_stack(images: List[sitk.Image],output_directory: str):
    """
    Save registered sections as TIFF files.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    for i, image in enumerate(images):
        filename = output_directory / f"{i:03d}.tif"
        image = sitk.Cast(image, sitk.sitkUInt16)
        sitk.WriteImage(image, str(filename))


def register_tif_directory(input_directory: str, output_directory: str, reference_index: int | None = None):
    """
    Complete serial-section registration pipeline.
    """

    images = read_tif_stack(input_directory)

    if reference_index is None:
        reference_index = len(images) // 2

    print(f"Loaded {len(images)} sections")
    print(f"Reference section: {reference_index}")

    adjacent, accumulated = register_serial_stack(images, reference_index=reference_index)
    reference = images[reference_index]
    registered = resample_stack(images, accumulated,reference)
    save_stack(registered, output_directory,)

    return adjacent, accumulated, registered

if __name__ == "__main__":

    input_directory = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK37/preps/C1/thumbnail_cleaned"
    output_directory = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK37/preps/C1/thumbnail_aligned"
    files = sorted(os.listdir(input_directory))
    reference_index = len(files) // 2

    adjacent, accumulated, registered = register_tif_directory(
        input_directory=input_directory,
        output_directory=output_directory,
        reference_index=reference_index,
    )