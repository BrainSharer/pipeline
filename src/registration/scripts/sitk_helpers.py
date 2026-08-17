from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import json
import math

import SimpleITK as sitk
import numpy as np

def dice_coefficient(fixed_mask, moving_mask):
    """
    Compute Dice coefficient between two SimpleITK binary images.

    Parameters
    ----------
    fixed_mask : sitk.Image
        Fixed binary mask.
    moving_mask : sitk.Image
        Registered moving binary mask.

    Returns
    -------
    float
        Dice coefficient in [0, 1].
    """

    # Make sure the images have the same geometry
    if fixed_mask.GetSize() != moving_mask.GetSize():
        raise ValueError("Fixed and moving masks have different sizes.")

    if fixed_mask.GetSpacing() != moving_mask.GetSpacing():
        raise ValueError("Fixed and moving masks have different spacing.")

    if fixed_mask.GetOrigin() != moving_mask.GetOrigin():
        raise ValueError("Fixed and moving masks have different origins.")

    # Convert to NumPy
    fixed = sitk.GetArrayViewFromImage(fixed_mask) > 0
    moving = sitk.GetArrayViewFromImage(moving_mask) > 0

    intersection = np.logical_and(fixed, moving).sum()

    fixed_volume = fixed.sum()
    moving_volume = moving.sum()

    if fixed_volume + moving_volume == 0:
        return 1.0

    dice = 2.0 * intersection / (fixed_volume + moving_volume)

    return float(dice)

def create_tissue_mask(image, threshold=10):
    """
    Create a mask to exclude empty regions.
    Assumes background is near zero.
    """
    mask = sitk.BinaryThreshold(
        image,
        lowerThreshold=threshold,
        upperThreshold=1e9,
        insideValue=255,
        outsideValue=0
    )
    return sitk.Cast(mask, sitk.sitkUInt8)
    eroder = sitk.GrayscaleErodeImageFilter()
    eroder.SetKernelType(sitk.sitkBall)
    eroder.SetKernelRadius(10)
    eroded_img = eroder.Execute(mask)




# -------------------------------------------------------------------------
# Results
# -------------------------------------------------------------------------

@dataclass
class RegistrationMetrics:
    """
    Quantitative metrics describing the overlap and spatial agreement
    between two binary 3D masks.

    All distances are reported in the physical coordinate system of the
    fixed image, normally microns for microscopy data.
    """

    # Overlap metrics
    dice: float
    jaccard: float

    # Volumes
    fixed_volume: float
    moving_volume: float
    intersection_volume: float
    union_volume: float

    # Distance metrics
    hausdorff_distance: float
    hausdorff_distance_95: float

    # Centroid information
    centroid_distance: float
    fixed_centroid: Optional[tuple[float, float, float]]
    moving_centroid: Optional[tuple[float, float, float]]

    # Foreground voxel counts
    fixed_voxels: int
    moving_voxels: int
    intersection_voxels: int
    union_voxels: int

    # Metadata
    spacing: tuple[float, float, float]
    number_of_dimensions: int

    def to_dict(self) -> dict:
        """Return metrics as a JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Return metrics as JSON."""
        return json.dumps(self.to_dict(), indent=indent)


# -------------------------------------------------------------------------
# Geometry
# -------------------------------------------------------------------------

def _same_geometry(
    image1: sitk.Image,
    image2: sitk.Image,
    tolerance: float = 1e-6,
) -> bool:
    """Check whether two SimpleITK images have the same physical geometry."""

    if image1.GetDimension() != image2.GetDimension():
        return False

    if image1.GetSize() != image2.GetSize():
        return False

    for a, b in zip(image1.GetSpacing(), image2.GetSpacing()):
        if not math.isclose(a, b, abs_tol=tolerance, rel_tol=0.0):
            return False

    for a, b in zip(image1.GetOrigin(), image2.GetOrigin()):
        if not math.isclose(a, b, abs_tol=tolerance, rel_tol=0.0):
            return False

    direction1 = image1.GetDirection()
    direction2 = image2.GetDirection()

    for a, b in zip(direction1, direction2):
        if not math.isclose(a, b, abs_tol=tolerance, rel_tol=0.0):
            return False

    return True


def _resample_mask_to_reference(
    moving_mask: sitk.Image,
    reference: sitk.Image,
) -> sitk.Image:
    """
    Resample a binary mask onto the reference image grid.

    Nearest-neighbor interpolation is mandatory for binary masks.
    """

    return sitk.Resample(
        moving_mask,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


# -------------------------------------------------------------------------
# Mask preparation
# -------------------------------------------------------------------------

def _make_binary_mask(
    image: sitk.Image,
    foreground_value: Optional[int] = None,
) -> sitk.Image:
    """
    Convert an image into a UInt8 binary mask.

    If foreground_value is None, every non-zero voxel is foreground.

    If foreground_value is specified, only voxels equal to that value
    are foreground.
    """

    if foreground_value is None:
        mask = image > 0
    else:
        mask = image == foreground_value

    return sitk.Cast(mask, sitk.sitkUInt8)


# -------------------------------------------------------------------------
# Volume calculations
# -------------------------------------------------------------------------

def _voxel_volume(image: sitk.Image) -> float:
    """Return physical volume of one voxel."""

    spacing = image.GetSpacing()

    volume = 1.0

    for s in spacing:
        volume *= s

    return volume


def _count_voxels(mask: sitk.Image) -> int:
    """Count non-zero voxels."""

    statistics = sitk.StatisticsImageFilter()
    statistics.Execute(mask)

    # Sum is safe for UInt8 masks.
    return int(statistics.GetSum())


# -------------------------------------------------------------------------
# Centroid
# -------------------------------------------------------------------------

def _centroid(
    mask: sitk.Image,
) -> Optional[tuple[float, float, float]]:
    """
    Calculate the physical-space centroid of a binary mask.

    Returns None for an empty mask.
    """

    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(mask)

    labels = statistics.GetLabels()

    if not labels:
        return None

    # Since this is a binary mask, there should normally be label 1.
    label = 1 if 1 in labels else labels[0]

    centroid = statistics.GetCentroid(label)

    return tuple(float(x) for x in centroid)


def _euclidean_distance(
    p1: Optional[tuple[float, float, float]],
    p2: Optional[tuple[float, float, float]],
) -> float:
    """Euclidean distance between two physical-space points."""

    if p1 is None or p2 is None:
        return float("nan")

    return math.sqrt(
        sum(
            (a - b) ** 2
            for a, b in zip(p1, p2)
        )
    )


# -------------------------------------------------------------------------
# Main metric function
# -------------------------------------------------------------------------

def compute_registration_metrics(
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    *,
    resample_moving: bool = True,
    geometry_tolerance: float = 1e-6,
) -> RegistrationMetrics:
    """
    Compute quantitative registration metrics between two binary 3D masks.

    Parameters
    ----------
    fixed_mask:
        Binary SimpleITK image defining the reference/fixed tissue.

    moving_mask:
        Binary SimpleITK image containing the registered moving tissue.

    resample_moving:
        If True, resample the moving mask onto the fixed image grid when
        the two images have different geometry.

        Nearest-neighbor interpolation is used.

    geometry_tolerance:
        Tolerance used when comparing image geometry.

    Returns
    -------
    RegistrationMetrics
        Dataclass containing Dice, Jaccard, volumes, Hausdorff distances,
        centroid displacement, and voxel counts.

    Notes
    -----
    All physical distances and volumes are based on the fixed image's
    spacing.

    The input images should represent binary masks. Non-zero voxels are
    treated as foreground.
    """

    # ------------------------------------------------------------------
    # Validate dimensions
    # ------------------------------------------------------------------

    if fixed_mask.GetDimension() != moving_mask.GetDimension():
        raise ValueError(
            "Fixed and moving masks must have the same dimensionality."
        )

    if fixed_mask.GetDimension() != 3:
        raise ValueError(
            "compute_registration_metrics() expects 3D images."
        )

    # ------------------------------------------------------------------
    # Convert to binary UInt8
    # ------------------------------------------------------------------

    fixed = _make_binary_mask(fixed_mask)
    moving = _make_binary_mask(moving_mask)

    # ------------------------------------------------------------------
    # Ensure same physical grid
    # ------------------------------------------------------------------

    if not _same_geometry(
        fixed,
        moving,
        tolerance=geometry_tolerance,
    ):

        if not resample_moving:
            raise ValueError(
                "Fixed and moving masks have different geometry. "
                "Set resample_moving=True or resample the moving mask "
                "before calling this function."
            )

        moving = _resample_mask_to_reference(
            moving,
            fixed,
        )

    # ------------------------------------------------------------------
    # Voxel volume
    # ------------------------------------------------------------------

    voxel_volume = _voxel_volume(fixed)

    # ------------------------------------------------------------------
    # Foreground counts
    # ------------------------------------------------------------------

    fixed_voxels = _count_voxels(fixed)
    moving_voxels = _count_voxels(moving)

    # ------------------------------------------------------------------
    # Intersection
    # ------------------------------------------------------------------

    intersection = sitk.And(
        fixed,
        moving,
    )

    intersection_voxels = _count_voxels(intersection)

    # ------------------------------------------------------------------
    # Union
    # ------------------------------------------------------------------

    union = sitk.Or(
        fixed,
        moving,
    )

    union_voxels = _count_voxels(union)

    # ------------------------------------------------------------------
    # Dice
    # ------------------------------------------------------------------

    denominator = fixed_voxels + moving_voxels

    if denominator == 0:
        dice = 1.0
    else:
        dice = (
            2.0 * intersection_voxels
            / denominator
        )

    # ------------------------------------------------------------------
    # Jaccard
    # ------------------------------------------------------------------

    if union_voxels == 0:
        jaccard = 1.0
    else:
        jaccard = (
            intersection_voxels
            / union_voxels
        )

    # ------------------------------------------------------------------
    # Physical volumes
    # ------------------------------------------------------------------

    fixed_volume = fixed_voxels * voxel_volume
    moving_volume = moving_voxels * voxel_volume
    intersection_volume = intersection_voxels * voxel_volume
    union_volume = union_voxels * voxel_volume

    # ------------------------------------------------------------------
    # Centroids
    # ------------------------------------------------------------------

    fixed_centroid = _centroid(fixed)
    moving_centroid = _centroid(moving)

    centroid_distance = _euclidean_distance(
        fixed_centroid,
        moving_centroid,
    )

    # ------------------------------------------------------------------
    # Hausdorff distance
    # ------------------------------------------------------------------

    if fixed_voxels == 0 or moving_voxels == 0:

        hausdorff_distance = float("nan")
        hausdorff_distance_95 = float("nan")

    else:

        distance_filter = sitk.HausdorffDistanceImageFilter()

        distance_filter.Execute(
            fixed,
            moving,
        )

        hausdorff_distance = float(
            distance_filter.GetHausdorffDistance()
        )

        # SimpleITK's HausdorffDistanceImageFilter does not provide
        # HD95, so calculate the distance transform based percentile.

        fixed_distance = sitk.SignedMaurerDistanceMap(
            fixed,
            squaredDistance=False,
            useImageSpacing=True,
        )

        moving_distance = sitk.SignedMaurerDistanceMap(
            moving,
            squaredDistance=False,
            useImageSpacing=True,
        )

        # Distance from fixed boundary to moving mask.
        fixed_boundary = sitk.LabelContour(
            fixed,
            fullyConnected=False,
        )

        moving_boundary = sitk.LabelContour(
            moving,
            fullyConnected=False,
        )

        fixed_to_moving = sitk.Abs(
            sitk.Mask(
                moving_distance,
                fixed_boundary,
            )
        )

        moving_to_fixed = sitk.Abs(
            sitk.Mask(
                fixed_distance,
                moving_boundary,
            )
        )

        # Convert boundary distances to NumPy arrays.
        import numpy as np

        d1 = sitk.GetArrayViewFromImage(
            fixed_to_moving
        )

        d2 = sitk.GetArrayViewFromImage(
            moving_to_fixed
        )

        d1 = d1[d1 > 0]
        d2 = d2[d2 > 0]

        if len(d1) == 0 and len(d2) == 0:
            hausdorff_distance_95 = 0.0

        else:
            distances = np.concatenate(
                [d1, d2]
            )

            hausdorff_distance_95 = float(
                np.percentile(
                    distances,
                    95,
                )
            )

    # ------------------------------------------------------------------
    # Return results
    # ------------------------------------------------------------------

    return RegistrationMetrics(
        dice=float(dice),
        jaccard=float(jaccard),

        fixed_volume=float(fixed_volume),
        moving_volume=float(moving_volume),
        intersection_volume=float(intersection_volume),
        union_volume=float(union_volume),

        hausdorff_distance=float(
            hausdorff_distance
        ),
        hausdorff_distance_95=float(
            hausdorff_distance_95
        ),

        centroid_distance=float(
            centroid_distance
        ),

        fixed_centroid=fixed_centroid,
        moving_centroid=moving_centroid,

        fixed_voxels=int(fixed_voxels),
        moving_voxels=int(moving_voxels),
        intersection_voxels=int(
            intersection_voxels
        ),
        union_voxels=int(union_voxels),

        spacing=tuple(
            float(x)
            for x in fixed.GetSpacing()
        ),

        number_of_dimensions=fixed.GetDimension(),
    )