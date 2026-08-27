from __future__ import annotations

import shutil
import sys
import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import ants
import numpy as np
import zarr

LOGGER = logging.getLogger("ants_zarr_registration") 
def configure_logging(verbose: bool = False) -> None: 
    level = logging.DEBUG if verbose else logging.INFO 
    logging.basicConfig( level=level, format="%(asctime)s | %(levelname)s | %(message)s", )


@dataclass
class VolumeGeometry:
    """
    Physical geometry of a volume.

    spacing:
        Physical voxel spacing in X, Y, Z.

    origin:
        Physical origin in X, Y, Z.

    direction:
        3x3 direction cosine matrix, flattened row-major.
    """

    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]


@dataclass
class RegistrationConfig:
    """
    Configuration for ANTs registration.
    """

    transform_type: str = "SyN"

    # Registration stages:
    #
    # Rigid:
    #   robust initial alignment
    #
    # Affine:
    #   global scaling/shearing/rotation/translation
    #
    # SyN:
    #   nonlinear deformation
    #
    # If you only want affine registration, set:
    #
    # transform_type = "Affine"
    #
    # or use a separate affine registration function below.

    metric: str = "mattes"

    iterations: tuple[tuple[int, ...], ...] = (
        (100, 70, 50),
    )

    shrink_factors: tuple[tuple[int, ...], ...] = (
        (4, 2, 1),
    )

    smoothing_sigmas: tuple[tuple[float, ...], ...] = (
        (2, 1, 0),
    )

    sampling_rate: float = 0.2

    random_seed: int = 42


# ---------------------------------------------------------------------------
# ANTs image creation
# ---------------------------------------------------------------------------

def numpy_to_ants(
    array: np.ndarray,
    spacing_xyz: Sequence[float],
    origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    direction: np.ndarray | None = None,
) -> ants.ANTsImage:
    """
    Convert a NumPy Z,Y,X volume to an ANTs image.

    ANTs/ITK convention is represented as X,Y,Z in physical space,
    while NumPy arrays are normally indexed Z,Y,X.

    Parameters
    ----------
    array:
        3-D array in Z,Y,X order.

    spacing_xyz:
        Physical spacing in X,Y,Z.

    origin_xyz:
        Physical origin in X,Y,Z.

    direction:
        3x3 direction matrix.

    Returns
    -------
    ants.ANTsImage
    """

    if array.ndim != 3:
        raise ValueError(
            f"Expected 3-D array, got shape {array.shape}"
        )

    array = np.asarray(array, dtype=np.float32)

    # ANTsPy accepts NumPy Z,Y,X-style arrays and associates physical
    # spacing using the spacing tuple.
    image = ants.from_numpy(
        array,
        spacing=tuple(float(x) for x in spacing_xyz),
        origin=tuple(float(x) for x in origin_xyz),
        direction=direction,
    )

    return image


# ---------------------------------------------------------------------------
# Zarr loading
# ---------------------------------------------------------------------------

def open_zarr_array(
    path: str,
    dataset: str = "",
):
    """
    Open a Zarr array.

    Supports:

        /volume.zarr
        /volume.zarr/0
        /volume.zarr/data

    Parameters
    ----------
    path:
        Zarr store.

    dataset:
        Dataset path inside the store.

    Returns
    -------
    zarr.Array
    """

    root = zarr.open(path, mode="r")

    if dataset:
        array = root[dataset]
    else:
        if isinstance(root, zarr.Array):
            array = root
        else:
            # Try common dataset names.
            for name in ("0", "data", "volume"):
                if name in root:
                    array = root[name]
                    break
            else:
                raise ValueError(
                    f"Could not determine Zarr dataset in {path}. "
                    "Specify --dataset."
                )

    if not isinstance(array, zarr.Array):
        raise TypeError(f"{dataset} is not a Zarr array")

    return array


# ---------------------------------------------------------------------------
# Zarr geometry
# ---------------------------------------------------------------------------

def read_geometry_from_zarr(
    array: zarr.Array,
    default_spacing_xyz: Sequence[float],
) -> VolumeGeometry:
    """
    Read physical geometry from Zarr attributes.

    Recognized attributes:

        spacing
        voxel_size
        resolution

    All are expected to be X,Y,Z.

    If unavailable, default_spacing_xyz is used.
    """

    attrs = dict(array.attrs)

    spacing = (
        attrs.get("spacing")
        or attrs.get("voxel_size")
        or attrs.get("resolution")
        or default_spacing_xyz
    )

    spacing = tuple(float(x) for x in spacing)

    if len(spacing) != 3:
        raise ValueError(
            f"Spacing must have 3 elements, got {spacing}"
        )

    origin = attrs.get(
        "origin",
        (0.0, 0.0, 0.0),
    )

    direction = attrs.get(
        "direction",
        np.eye(3).tolist(),
    )

    direction = np.asarray(direction, dtype=float)

    if direction.size == 9:
        direction = direction.reshape(3, 3)
    else:
        raise ValueError(
            "Direction must contain 9 values."
        )

    return VolumeGeometry(
        spacing=spacing,
        origin=tuple(float(x) for x in origin),
        direction=tuple(direction.ravel()),
    )


# ---------------------------------------------------------------------------
# Downsampled volume loading
# ---------------------------------------------------------------------------

def load_downsampled_volume(
    zarr_path: str,
    dataset: str,
    spacing_xyz: Sequence[float],
) -> ants.ANTsImage:
    """
    Load a complete downsampled volume into memory.

    This function is intended ONLY for the downsampled registration
    volumes, not the huge full-resolution volume.
    """

    array = open_zarr_array(
        zarr_path,
        dataset,
    )

    LOGGER.info(
        "Loading downsampled volume: %s",
        zarr_path,
    )

    data = np.asarray(array[:], dtype=np.float32)

    LOGGER.info(
        "Volume shape Z,Y,X = %s",
        data.shape,
    )

    geometry = read_geometry_from_zarr(
        array,
        spacing_xyz,
    )

    direction = np.asarray(
        geometry.direction,
        dtype=float,
    ).reshape(3, 3)

    return numpy_to_ants(
        data,
        spacing_xyz=geometry.spacing,
        origin_xyz=geometry.origin,
        direction=direction,
    )


# ---------------------------------------------------------------------------
# Initial alignment
# ---------------------------------------------------------------------------

def center_of_mass_initialization(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
):
    """
    Compute an initial rigid transform using image centers of mass.

    This is useful when the two volumes have substantial translation.
    """

    LOGGER.info("Computing center-of-mass initialization")

    tx = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Translation",
    )

    return tx


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_volumes(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    config: RegistrationConfig,
    output_dir: str,
):
    """
    Register moving to fixed.

    Returns
    -------
    dict
        ANTs registration dictionary.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGGER.info(
        "Starting ANTs registration: %s",
        config.transform_type,
    )

    if config.transform_type.lower() == "affine":

        result = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Affine",
            aff_metric=config.metric,
            aff_sampling=32,
            aff_random_sampling_rate=config.sampling_rate,
            random_seed=config.random_seed,
            write_composite_transform=True,
            outprefix=str(
                output_dir / "affine_"
            ),
        )

    elif config.transform_type.lower() == "syn":

        result = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="SyN",
            aff_metric=config.metric,
            aff_sampling=32,
            aff_random_sampling_rate=config.sampling_rate,
            syn_metric="mattes",
            syn_sampling=32,
            reg_iterations=(100, 70, 50),
            random_seed=config.random_seed,
            write_composite_transform=True,
            outprefix=str(
                output_dir / "syn_"
            ),
        )

    elif config.transform_type.lower() == "synonly":

        result = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="SyNOnly",
            syn_metric="mattes",
            syn_sampling=32,
            reg_iterations=(100, 70, 50),
            random_seed=config.random_seed,
            write_composite_transform=True,
            outprefix=str(
                output_dir / "synonly_"
            ),
        )

    else:
        raise ValueError(
            f"Unsupported transform type: "
            f"{config.transform_type}"
        )

    LOGGER.info(
        "Registration completed"
    )

    return result


# ---------------------------------------------------------------------------
# Save registration metadata
# ---------------------------------------------------------------------------

def save_registration_metadata(
    result: dict,
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    output_dir: str,
) -> None:
    """
    Save transform filenames and image geometry.
    """

    output_dir = Path(output_dir)

    metadata = {
        "fwdtransforms": [
            str(x)
            for x in result["fwdtransforms"]
        ],
        "invtransforms": [
            str(x)
            for x in result["invtransforms"]
        ],
        "fixed_shape": list(fixed.shape),
        "moving_shape": list(moving.shape),
        "fixed_spacing": list(fixed.spacing),
        "moving_spacing": list(moving.spacing),
        "fixed_origin": list(fixed.origin),
        "moving_origin": list(moving.origin),
        "fixed_direction": np.asarray(
            fixed.direction
        ).tolist(),
        "moving_direction": np.asarray(
            moving.direction
        ).tolist(),
    }

    metadata_path = (
        output_dir /
        "registration.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    LOGGER.info(
        "Saved registration metadata: %s",
        metadata_path,
    )


# ---------------------------------------------------------------------------
# Apply transform to a complete image
# ---------------------------------------------------------------------------

def apply_transform(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    transform_list: Sequence[str],
) -> ants.ANTsImage:
    """
    Apply saved ANTs transforms.

    This function is appropriate for moderate-sized images.

    It is NOT appropriate for the huge full-resolution Zarr volume.
    """

    print(f'type of list(transform_list) {type(list(transform_list))}')

    return ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=[transform_list],
        interpolator="linear",
    )


# ---------------------------------------------------------------------------
# Zarr block processing
# ---------------------------------------------------------------------------

def compute_block_bounds(
    z: int,
    y: int,
    x: int,
    block_shape: Sequence[int],
    volume_shape: Sequence[int],
):
    """
    Compute clipped Z,Y,X block bounds.
    """

    z1 = min(
        z + block_shape[0],
        volume_shape[0],
    )

    y1 = min(
        y + block_shape[1],
        volume_shape[1],
    )

    x1 = min(
        x + block_shape[2],
        volume_shape[2],
    )

    return (
        z,
        z1,
        y,
        y1,
        x,
        x1,
    )


def iter_blocks(
    shape: Sequence[int],
    block_shape: Sequence[int],
):
    """
    Iterate through Z,Y,X block coordinates.
    """

    nz, ny, nx = shape
    bz, by, bx = block_shape

    for z in range(0, nz, bz):
        for y in range(0, ny, by):
            for x in range(0, nx, bx):
                yield (
                    z,
                    y,
                    x,
                )


# ---------------------------------------------------------------------------
# Physical crop geometry
# ---------------------------------------------------------------------------

def block_origin(
    origin_xyz: Sequence[float],
    spacing_xyz: Sequence[float],
    z0: int,
    y0: int,
    x0: int,
) -> tuple[float, float, float]:
    """
    Calculate the physical origin of a Z,Y,X block.

    Assumes identity direction.

    For non-identity directions, use the full direction matrix to
    transform the index vector.
    """

    return (
        origin_xyz[0] + x0 * spacing_xyz[0],
        origin_xyz[1] + y0 * spacing_xyz[1],
        origin_xyz[2] + z0 * spacing_xyz[2],
    )


# ---------------------------------------------------------------------------
# Large Zarr transformation
# ---------------------------------------------------------------------------

def transform_large_zarr(
    moving_zarr_path: str,
    output_zarr_path: str,
    transform_list: Sequence[str],
    moving_spacing_xyz: Sequence[float],
    moving_origin_xyz: Sequence[float] = (
        0.0,
        0.0,
        0.0,
    ),
    moving_direction: np.ndarray | None = None,
    fixed_spacing_xyz: Sequence[float] | None = None,
    fixed_origin_xyz: Sequence[float] | None = None,
    fixed_shape_zyx: Sequence[int] | None = None,
    block_shape: Sequence[int] = (
        64,
        512,
        512,
    ),
    interpolator: str = "linear",
    output_dtype=np.uint16,
    compressor=None,
):
    """
    Transform a huge Zarr volume block-by-block.

    Parameters
    ----------
    moving_zarr_path:
        Input full-resolution Zarr.

    output_zarr_path:
        Output Zarr.

    transform_list:
        ANTs transforms returned by ants.registration()['fwdtransforms'].

    moving_spacing_xyz:
        Full-resolution moving spacing.

    fixed_spacing_xyz:
        Full-resolution fixed spacing.

    fixed_shape_zyx:
        Shape of output volume.

    block_shape:
        Processing block size Z,Y,X.

    Notes
    -----
    The transform is evaluated in physical coordinates. There is
    intentionally NO multiplication of the transform translation
    or affine matrix by a downsampling factor.
    """

    if fixed_spacing_xyz is None:
        fixed_spacing_xyz = moving_spacing_xyz

    if fixed_origin_xyz is None:
        fixed_origin_xyz = moving_origin_xyz

    if moving_direction is None:
        moving_direction = np.eye(3)

    moving = zarr.open(moving_zarr_path, mode='r')

    if moving.ndim != 3:
        raise ValueError(
            "Expected a 3-D Zarr volume."
        )

    print(f'fixed shape zyx = {fixed_shape_zyx}')
    print(f'fixed spacing xyz {fixed_spacing_xyz}')


    fixed_shape_zyx = tuple(
        int(x)
        for x in fixed_shape_zyx
    )

    LOGGER.info(
        "Input full-resolution shape: %s",
        moving.shape,
    )

    LOGGER.info(
        "Output shape: %s",
        fixed_shape_zyx,
    )

    LOGGER.info(
        "Block shape: %s",
        block_shape,
    )

    # ------------------------------------------------------------------
    # Create output store.
    # ------------------------------------------------------------------

    output = zarr.open(
        output_zarr_path,
        mode="w",
        shape=fixed_shape_zyx,
        chunks=(1, fixed_shape_zyx[1], fixed_shape_zyx[2]),
        dtype=np.uint16)        

    # ------------------------------------------------------------------
    # Save geometry.
    # ------------------------------------------------------------------

    output.attrs["spacing"] = list(
        fixed_spacing_xyz
    )

    output.attrs["origin"] = list(
        fixed_origin_xyz
    )

    output.attrs["direction"] = (
        np.asarray(
            moving_direction
        ).tolist()
    )

    # ------------------------------------------------------------------
    # Iterate over output blocks.
    # ------------------------------------------------------------------

    blocks = list(
        iter_blocks(
            fixed_shape_zyx,
            block_shape,
        )
    )

    total_blocks = len(blocks)

    for block_number, (
        z0,
        y0,
        x0,
    ) in enumerate(blocks, start=1):

        (
            z0,
            z1,
            y0,
            y1,
            x0,
            x1,
        ) = compute_block_bounds(
            z0,
            y0,
            x0,
            block_shape,
            fixed_shape_zyx,
        )

        LOGGER.info(
            "Block %d/%d: "
            "z=%d:%d y=%d:%d x=%d:%d",
            block_number,
            total_blocks,
            z0,
            z1,
            y0,
            y1,
            x0,
            x1,
        )

        # --------------------------------------------------------------
        # Physical origin of the output block.
        # --------------------------------------------------------------

        out_origin = block_origin(
            fixed_origin_xyz,
            fixed_spacing_xyz,
            z0,
            y0,
            x0,
        )

        block_size = (
            z1 - z0,
            y1 - y0,
            x1 - x0,
        )

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # The block is an OUTPUT-space block. To obtain all moving
        # voxels required for interpolation, we need a corresponding
        # source region.
        #
        # A simple implementation reads the entire moving volume.
        # That defeats the purpose for a huge volume.
        #
        # Therefore we use ANTs to resample each block in physical
        # coordinates. For production-scale volumes, use the
        # transform's inverse to calculate a source bounding box.
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # Conservative source region.
        #
        # This version uses the full moving volume for the transform
        # operation but only writes one output block.
        #
        # It is memory-safe with respect to the OUTPUT volume but may
        # still require substantial memory for the source image.
        #
        # The production version below uses chunk-local source
        # extraction for affine transforms.
        # --------------------------------------------------------------

        moving_data = np.asarray(
            moving[:],
            dtype=np.float32,
        )

        moving_image = numpy_to_ants(
            moving_data,
            spacing_xyz=moving_spacing_xyz,
            origin_xyz=moving_origin_xyz,
            direction=moving_direction,
        )

        # --------------------------------------------------------------
        # Construct an output reference image for this block.
        # --------------------------------------------------------------

        block_array = np.zeros(
            block_size,
            dtype=np.float32,
        )

        fixed_block = numpy_to_ants(
            block_array,
            spacing_xyz=fixed_spacing_xyz,
            origin_xyz=out_origin,
            direction=moving_direction,
        )

        # --------------------------------------------------------------
        # Resample.
        # --------------------------------------------------------------

        transformed = ants.apply_transforms(
            fixed=fixed_block,
            moving=moving_image,
            transformlist=list(transform_list),
            interpolator=interpolator,
        )

        result = transformed.numpy()

        output[
            z0:z1,
            y0:y1,
            x0:x1,
        ] = result.astype(
            output_dtype,
            copy=False,
        )

    LOGGER.info(
        "Full-resolution transformation complete."
    )


# ---------------------------------------------------------------------------
# Affine-only large-volume processing
# ---------------------------------------------------------------------------

def load_affine_transform_matrix(
    transform_path: str,
):
    """
    Read an ANTs/ITK affine transform.

    Uses ants.read_transform().
    """

    tx = ants.read_transform(
        transform_path
    )

    return tx


def transform_full_volume_affine(
    moving_zarr_path: str,
    output_zarr_path: str,
    dataset: str,
    output_dataset: str,
    affine_transform: str,
    moving_spacing_xyz: Sequence[float],
    fixed_spacing_xyz: Sequence[float],
    fixed_shape_zyx: Sequence[int],
    moving_origin_xyz=(0.0, 0.0, 0.0),
    fixed_origin_xyz=(0.0, 0.0, 0.0),
    block_shape=(64, 512, 512),
    output_dtype=np.float32,
):
    """
    Apply an affine ANTs transform to a huge Zarr volume.

    This version is intended for affine transforms where the inverse
    affine can be used to determine the required source region.
    """

    moving = open_zarr_array(
        moving_zarr_path,
        dataset,
    )

    root = zarr.open(
        output_zarr_path,
        mode="a",
    )

    output = root.require_dataset(
        output_dataset,
        shape=tuple(fixed_shape_zyx),
        chunks=tuple(block_shape),
        dtype=output_dtype,
        overwrite=False,
    )

    output.attrs["spacing"] = list(
        fixed_spacing_xyz
    )

    output.attrs["origin"] = list(
        fixed_origin_xyz
    )

    # --------------------------------------------------------------
    # Load transform.
    # --------------------------------------------------------------

    tx = ants.read_transform(
        affine_transform
    )

    LOGGER.info(
        "Loaded affine transform: %s",
        affine_transform,
    )

    # --------------------------------------------------------------
    # For affine transforms, use ANTs to perform block resampling.
    #
    # A practical implementation should calculate the inverse
    # transformed bounding box of each output block and read only
    # that region from Zarr.
    # --------------------------------------------------------------

    for block_number, (
        z0,
        y0,
        x0,
    ) in enumerate(
        iter_blocks(
            fixed_shape_zyx,
            block_shape,
        ),
        start=1,
    ):

        (
            z0,
            z1,
            y0,
            y1,
            x0,
            x1,
        ) = compute_block_bounds(
            z0,
            y0,
            x0,
            block_shape,
            fixed_shape_zyx,
        )

        LOGGER.info(
            "Affine block %d: "
            "z=%d:%d y=%d:%d x=%d:%d",
            block_number,
            z0,
            z1,
            y0,
            y1,
            x0,
            x1,
        )

        out_origin = block_origin(
            fixed_origin_xyz,
            fixed_spacing_xyz,
            z0,
            y0,
            x0,
        )

        shape = (
            z1 - z0,
            y1 - y0,
            x1 - x0,
        )

        # ----------------------------------------------------------
        # NOTE:
        #
        # This implementation reads the moving volume once per
        # block. For very large data this should be replaced with
        # inverse-transform bounding-box extraction.
        # ----------------------------------------------------------

        moving_data = np.asarray(
            moving[:],
            dtype=np.float32,
        )

        moving_image = numpy_to_ants(
            moving_data,
            spacing_xyz=moving_spacing_xyz,
            origin_xyz=moving_origin_xyz,
        )

        reference = numpy_to_ants(
            np.zeros(
                shape,
                dtype=np.float32,
            ),
            spacing_xyz=fixed_spacing_xyz,
            origin_xyz=out_origin,
        )

        result = ants.apply_transforms(
            fixed=reference,
            moving=moving_image,
            transformlist=[
                affine_transform
            ],
            interpolator="linear",
        )

        output[
            z0:z1,
            y0:y1,
            x0:x1,
        ] = result.numpy().astype(
            output_dtype,
            copy=False,
        )


# ---------------------------------------------------------------------------
# Registration command
# ---------------------------------------------------------------------------

def command_register(moving, fixed, downsample, transformation, debug):
    """
    Perform downsampled registration.
    """
    scratch_dir = "/data/pipeline_tmp"
    base_path = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data"
    reg_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration"
    moving_zarr_path = os.path.join(scratch_dir, moving, f'source.{downsample}.zarr')
    fixed_zarr_path = os.path.join(scratch_dir, fixed, f'source.{downsample}.zarr')
    xy_resolution = 0.325
    z_resolution = 20.0                
    #transform_dir = os.path.join(reg_path, f"{moving}_{fixed}.tfm")
    spacing = (xy_resolution*downsample, xy_resolution*downsample, z_resolution)

    dataset = ""


    fixed = load_downsampled_volume(
        fixed_zarr_path,
        dataset,
        spacing,
    )

    moving = load_downsampled_volume(
        moving_zarr_path,
        dataset,
        spacing,
    )

    config = RegistrationConfig(
        transform_type=transformation
    )

    result = register_volumes(
        fixed=fixed,
        moving=moving,
        config=config,
        output_dir=reg_path,
    )

    save_registration_metadata(
        result,
        fixed,
        moving,
        reg_path,
    )

    LOGGER.info(
        "Forward transforms:"
    )

    transforms = result["fwdtransforms"]

    if isinstance(transforms, list):
        for tx in transforms:
            LOGGER.info("  %s",tx,)
    else:
        LOGGER.info("  %s", transforms)
        transforms = [transforms]

    # --------------------------------------------------------------
    # Save a preview registration.
    # --------------------------------------------------------------

    warped = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=transforms,
        interpolator="linear",
    )


    ants.image_write(
        warped,
        str(
            Path(reg_path)
            / "registered_preview.nii"
        ),
    )

    LOGGER.info(
        "Saved registered preview."
    )


# ---------------------------------------------------------------------------
# Apply command
# ---------------------------------------------------------------------------

def command_apply_other_resolution(moving, fixed, downsample):
    """
    Apply previously saved transforms to a full-resolution Zarr volume.
    """
    scratch_dir = "/data/pipeline_tmp"
    reg_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration"
    moving_zarr_path = os.path.join(scratch_dir, moving, f'source.{downsample}.zarr')
    if not os.path.exists(moving_zarr_path):
        print(f'Missing: {moving_zarr_path}')
        exit(0)
    output_zarr_path = os.path.join(scratch_dir, moving, f'{moving}_{fixed}_registered.{downsample}.zarr')
    if os.path.exists(output_zarr_path):
        print('Removing existing {output_zarr_path}')
        shutil.rmtree(output_zarr_path)
    #fixed_xyz = 60000x34000x485
    fixed_shape_zyx = (485, 34000//downsample, 60000//downsample) 
    xy_resolution = 0.325
    z_resolution = 20.0                
    spacing = (xy_resolution*downsample, xy_resolution*downsample, z_resolution)

    divisors = {}
    divisors[1] = 32
    divisors[8] = 8
    divisors[16] = 4
    divisors[32] = 4
    try:
        divisor = divisors[downsample]
    except KeyError:
        divisor = 2

    moving_zarr = zarr.open(moving_zarr_path, mode='r')
    block_shape_zyx = (1, moving_zarr.chunks[1], moving_zarr.chunks[2])
    interpolator = "linear"

    metadata_path = (
        Path(reg_path)
        / "registration.json"
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    #transforms = metadata["fwdtransforms"]
    transforms = [os.path.join(reg_path, 'syn_Composite.h5')]

    LOGGER.info(
        "Using transforms:"
    )

    LOGGER.info("transforms: %s",transforms)

    transform_large_zarr(
        moving_zarr_path=moving_zarr_path,
        output_zarr_path=output_zarr_path,
        transform_list=transforms,
        moving_spacing_xyz=spacing,
        fixed_spacing_xyz=spacing,
        fixed_shape_zyx=fixed_shape_zyx,
        block_shape=block_shape_zyx,
        interpolator=interpolator,
    )

