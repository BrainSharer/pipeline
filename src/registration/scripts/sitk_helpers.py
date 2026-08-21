##### Padding helper
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Sequence
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

    radius = [18,18,18]

    mask = sitk.BinaryMorphologicalClosing(mask, kernelRadius=radius)
    mask = sitk.BinaryMorphologicalOpening(mask, radius)
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    return mask

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



@dataclass(frozen=True)
class AffinePadding:
    """
    Conservative source-region expansion caused by an affine transform.

    Values are in VOXELS in (z, y, x) order.
    """
    lower: Tuple[int, int, int]
    upper: Tuple[int, int, int]

    @property
    def z(self) -> Tuple[int, int]:
        return self.lower[0], self.upper[0]

    @property
    def y(self) -> Tuple[int, int]:
        return self.lower[1], self.upper[1]

    @property
    def x(self) -> Tuple[int, int]:
        return self.lower[2], self.upper[2]


@dataclass(frozen=True)
class SourceRegion:
    """
    Source region required to generate one output chunk.

    All coordinates are half-open:
        [start, stop)

    Coordinates are (z, y, x).
    """
    start: Tuple[int, int, int]
    stop: Tuple[int, int, int]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(
            stop - start
            for start, stop in zip(self.start, self.stop)
        )


def compute_affine_padding(
    matrix: np.ndarray,
    translation: np.ndarray,
    chunk_shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float] | None = None,
) -> AffinePadding:
    """
    Compute a conservative maximum padding required by an affine transform.

    Parameters
    ----------
    matrix:
        3x3 affine matrix.

        Coordinates must use the same axis ordering as `chunk_shape`.
        For a Zarr array with shape (z, y, x), this means:
            [z, y, x]

    translation:
        Translation vector with the same coordinate ordering.

    chunk_shape:
        Output chunk dimensions in voxels:
            (z, y, x)

    spacing:
        Optional physical voxel spacing:
            (z, y, x)

        If supplied, the affine matrix and translation are assumed to
        operate in physical units. The resulting padding is converted
        back to voxels.

    Returns
    -------
    AffinePadding
        Conservative lower/upper padding in voxels.

    Notes
    -----
    The calculation transforms the eight corners of a chunk and determines
    the largest displacement from the corresponding untransformed chunk.

    The returned padding is deliberately conservative so that every chunk
    can use the same padding without recomputing affine geometry.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)

    if matrix.shape != (3, 3):
        raise ValueError("matrix must have shape (3, 3)")

    if translation.shape != (3,):
        raise ValueError("translation must have shape (3,)")

    if len(chunk_shape) != 3:
        raise ValueError("chunk_shape must contain (z, y, x)")

    if any(v <= 0 for v in chunk_shape):
        raise ValueError("chunk dimensions must be positive")

    if spacing is not None:
        spacing = np.asarray(spacing, dtype=np.float64)

        if spacing.shape != (3,):
            raise ValueError("spacing must have shape (3,)")

        if np.any(spacing <= 0):
            raise ValueError("spacing must be positive")

    # ------------------------------------------------------------
    # Use the inverse transform because output -> input sampling
    # is what determines the required source region.
    # ------------------------------------------------------------
    try:
        inverse_matrix = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Affine matrix is singular") from exc

    inverse_translation = -inverse_matrix @ translation

    # Eight corners of one chunk.
    #
    # Coordinates are expressed as voxel coordinates.
    # Use [0, size] rather than [0, size-1] because we want a
    # conservative bounding box.
    z, y, x = chunk_shape

    corners = np.array(
        [
            [cz, cy, cx]
            for cz in (0.0, float(z))
            for cy in (0.0, float(y))
            for cx in (0.0, float(x))
        ],
        dtype=np.float64,
    )

    # ------------------------------------------------------------
    # Convert voxel coordinates to physical coordinates if needed.
    # ------------------------------------------------------------
    if spacing is not None:
        physical_corners = corners * spacing

        transformed = (
            physical_corners @ inverse_matrix.T
            + inverse_translation
        )

        # Back to voxel coordinates.
        transformed /= spacing
    else:
        transformed = (
            corners @ inverse_matrix.T
            + inverse_translation
        )

    # ------------------------------------------------------------
    # Determine transformed bounding box.
    # ------------------------------------------------------------
    transformed_min = transformed.min(axis=0)
    transformed_max = transformed.max(axis=0)

    original_min = np.zeros(3, dtype=np.float64)
    original_max = np.asarray(chunk_shape, dtype=np.float64)

    # How far the transformed region extends below/above the original
    # chunk.
    lower = original_min - transformed_min
    upper = transformed_max - original_max

    # Translation can move the entire chunk. The source bounding box
    # therefore needs to include that displacement too.
    lower = np.maximum(lower, 0.0)
    upper = np.maximum(upper, 0.0)

    return AffinePadding(
        lower=tuple(np.ceil(lower).astype(int)),
        upper=tuple(np.ceil(upper).astype(int)),
    )


def affine_source_region(
    output_start: Tuple[int, int, int],
    output_stop: Tuple[int, int, int],
    padding: AffinePadding,
    source_shape: Tuple[int, int, int],
) -> SourceRegion:
    """
    Construct an efficient source read region for one output chunk.

    Parameters
    ----------
    output_start:
        Output chunk start as (z, y, x).

    output_stop:
        Output chunk stop as (z, y, x), exclusive.

    padding:
        Precomputed affine padding.

    source_shape:
        Full source volume shape as (z, y, x).

    Returns
    -------
    SourceRegion
        Clipped source region required for this output chunk.
    """
    output_start = np.asarray(output_start, dtype=np.int64)
    output_stop = np.asarray(output_stop, dtype=np.int64)
    source_shape = np.asarray(source_shape, dtype=np.int64)

    if np.any(output_start < 0):
        raise ValueError("output_start cannot be negative")

    if np.any(output_stop <= output_start):
        raise ValueError("output_stop must be greater than output_start")

    if np.any(output_stop > source_shape):
        raise ValueError(
            f"Output region {output_start}-{output_stop} "
            f"exceeds source shape {source_shape}"
        )

    lower = np.asarray(padding.lower, dtype=np.int64)
    upper = np.asarray(padding.upper, dtype=np.int64)

    source_start = np.maximum(
        output_start - lower,
        0,
    )

    source_stop = np.minimum(
        output_stop + upper,
        source_shape,
    )

    return SourceRegion(
        start=tuple(source_start),
        stop=tuple(source_stop),
    )


def compute_chunk_source_region(
    chunk_index: Tuple[int, int, int],
    chunk_shape: Tuple[int, int, int],
    source_shape: Tuple[int, int, int],
    padding: AffinePadding,
) -> SourceRegion:
    """
    Convenience function for Zarr chunk indices.

    Parameters
    ----------
    chunk_index:
        Chunk index (cz, cy, cx).

    chunk_shape:
        Chunk size (z, y, x).

    source_shape:
        Full source volume shape (z, y, x).

    padding:
        Precomputed affine padding.

    Returns
    -------
    SourceRegion
    """
    chunk_index = np.asarray(chunk_index, dtype=np.int64)
    chunk_shape = np.asarray(chunk_shape, dtype=np.int64)
    source_shape = np.asarray(source_shape, dtype=np.int64)

    output_start = chunk_index * chunk_shape
    output_stop = np.minimum(
        output_start + chunk_shape,
        source_shape,
    )

    return affine_source_region(
        tuple(output_start),
        tuple(output_stop),
        padding,
        tuple(source_shape),
    )

##### Helpers for RGB data

# ------------------------------------------------------------------
# IMAGE / CHANNEL HELPERS
# ------------------------------------------------------------------

def detect_channels(array: np.ndarray) -> int:
    """
    Determine whether a TIFF is grayscale or RGB.

    Supported input:

        (Y, X)       -> 1 channel
        (Y, X, 1)    -> 1 channel
        (Y, X, 3)    -> 3 channels
        (Y, X, 4)    -> 4 channels

    Four-channel input is preserved, although RGB registration
    uses only the first three channels.
    """

    array = np.asarray(array)

    if array.ndim == 2:
        return 1

    if array.ndim == 3:
        if array.shape[-1] in (1, 3, 4):
            return int(array.shape[-1])

    raise ValueError(
        f"Unsupported TIFF dimensions: {array.shape}. "
        "Expected (Y,X), (Y,X,1), (Y,X,3), or (Y,X,4)."
    )

def normalize_tiff_array(
    array: np.ndarray,
) -> np.ndarray:
    """
    Normalize a TIFF plane to either:

        (Y,X)

    or:

        (Y,X,C)

    without changing its dtype.
    """

    array = np.asarray(array)

    if array.ndim == 2:
        return array

    if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
        if array.shape[-1] == 1:
            return array[..., 0]

        return array

    raise ValueError(
        f"Unsupported TIFF shape {array.shape}. "
        "Expected (Y,X) or (Y,X,C)."
    )

def rgb_to_luminance(
    array: np.ndarray,
) -> np.ndarray:
    """
    Convert an RGB array to a scalar floating-point image.

    Input:
        (Y,X,3)

    Output:
        (Y,X), float32
    """

    array = np.asarray(array)

    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(
            f"Expected RGB image with shape (Y,X,3+), "
            f"got {array.shape}"
        )

    rgb = array[..., :3].astype(np.float32)

    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    ).astype(np.float32)


def rgb_channel(
    array: np.ndarray,
    channel: str,
) -> np.ndarray:
    """
    Extract a scalar registration image from RGB data.
    """

    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(
            f"Expected RGB array, got {array.shape}"
        )

    if channel == "luminance":
        return rgb_to_luminance(array)

    index = {
        "red": 0,
        "green": 1,
        "blue": 2,
    }[channel]

    return array[..., index].astype(np.float32)

def make_registration_image(
    array: np.ndarray,
    registration_channel: str = "luminance",
) -> np.ndarray:
    """
    Convert either grayscale or RGB data into a scalar image
    suitable for SimpleITK registration.
    """

    array = normalize_tiff_array(array)

    if array.ndim == 2:
        return array.astype(np.float32)

    return rgb_channel(
        array,
        registration_channel,
    )


# ------------------------------------------------------------------
# RGB / GRAYSCALE BLOCK RESAMPLING
# ------------------------------------------------------------------

def _resample_scalar_block(
    src_zyx: np.ndarray,
    src_origin_xyz: Sequence[float],
    reference: sitk.Image,
    transform: sitk.Transform,
    background_color: int
) -> np.ndarray:
    """
    Resample a scalar ZYX block.
    """

    moving = sitk.GetImageFromArray(
        src_zyx
    )

    moving = sitk.Cast(
        moving,
        sitk.sitkFloat32,
    )

    moving.SetSpacing(
        reference.GetSpacing()
    )

    moving.SetOrigin(
        src_origin_xyz
    )

    moving.SetDirection(
        np.eye(3).flatten()
    )

    result = sitk.Resample(
        moving,
        reference,
        transform,
        sitk.sitkLinear,
        background_color,
        sitk.sitkFloat32,
    )

    return sitk.GetArrayFromImage(
        result
    )

def _resample_rgb_block(
    src_zyxc: np.ndarray,
    src_origin_xyz: Sequence[float],
    reference: sitk.Image,
    transform: sitk.Transform,
    output_dtype,
    background_color
) -> np.ndarray:
    """
    Resample an RGB/RGBA block channel-by-channel.

    This avoids attempting to use SimpleITK's vector image
    interpolation for the large block path and gives explicit
    control over dtype preservation.
    """

    channels = src_zyxc.shape[-1]

    output = np.empty(
        (
            reference.GetSize()[2],
            reference.GetSize()[1],
            reference.GetSize()[0],
            channels,
        ),
        dtype=np.float32,
    )

    for channel in range(channels):

        result = _resample_scalar_block(
            src_zyx=src_zyxc[..., channel],
            src_origin_xyz=src_origin_xyz,
            reference=reference,
            transform=transform,
            background_color=background_color
        )

        output[..., channel] = result

    # Round before integer conversion to reduce interpolation
    # bias for integer-valued microscopy data.
    if np.issubdtype(
        output_dtype,
        np.integer,
    ):
        info = np.iinfo(output_dtype)

        output = np.rint(
            output
        )

        output = np.clip(
            output,
            info.min,
            info.max,
        )

    return output.astype(
        output_dtype,
        copy=False,
    )

# ------------------------------------------------------------------
# CHANNEL-AWARE ZARR HELPERS
# ------------------------------------------------------------------

def _channel_count(zarr_array) -> int:
    """
    Return number of channels in a Zarr volume.
    """

    if len(zarr_array.shape) == 3:
        return 1

    if len(zarr_array.shape) == 4:
        channels = zarr_array.shape[-1]

        if channels not in (1, 3, 4):
            raise ValueError(
                f"Unsupported channel count: {channels}"
            )

        return int(channels)

    raise ValueError(
        f"Expected 3-D or 4-D Zarr array, "
        f"got shape={zarr_array.shape}"
    )

def _spatial_shape(zarr_array):
    """
    Return spatial shape as (Z,Y,X).
    """

    return tuple(
        int(x)
        for x in zarr_array.shape[:3]
    )

def _spatial_chunks(zarr_array):
    """
    Return spatial chunks as (Z,Y,X).
    """

    return tuple(
        int(x)
        for x in zarr_array.chunks[:3]
    )

def _make_spatial_slice(
    start: Sequence[int],
    stop: Sequence[int],
):
    return (
        slice(start[0], stop[0]),
        slice(start[1], stop[1]),
        slice(start[2], stop[2]),
    )
