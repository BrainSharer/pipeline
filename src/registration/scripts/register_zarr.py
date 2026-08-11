"""
Apply a SimpleITK affine transform to a very large Zarr volume
without loading the complete volume into RAM.

Input Zarr:
    shape = (z, y, x)
    spacing = (x, y, z) in physical units

Example:
    shape   = (485, 35500, 65500)
    spacing = (0.325, 0.325, 20.0)

The SimpleITK transform is assumed to operate in physical coordinates.

The output is another Zarr array with the requested output geometry.

Requirements:
    pip install numpy zarr SimpleITK tqdm
"""

from __future__ import annotations

import os
import argparse
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK as sitk
import zarr
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def index_to_physical(
    index_xyz: np.ndarray,
    spacing_xyz: np.ndarray,
    origin_xyz: np.ndarray,
) -> np.ndarray:
    """Convert XYZ voxel indices to physical coordinates."""
    return origin_xyz + index_xyz * spacing_xyz


def physical_to_index(
    physical_xyz: np.ndarray,
    spacing_xyz: np.ndarray,
    origin_xyz: np.ndarray,
) -> np.ndarray:
    """Convert physical coordinates to continuous XYZ voxel indices."""
    return (physical_xyz - origin_xyz) / spacing_xyz


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def transform_corners(
    transform: sitk.Transform,
    size_xyz: Tuple[int, int, int],
    spacing_xyz: Tuple[float, float, float],
    origin_xyz: Tuple[float, float, float],
) -> np.ndarray:
    """
    Transform all 8 corners of a volume.

    Returns
    -------
    ndarray
        Shape (8, 3), physical XYZ coordinates.
    """

    sx, sy, sz = size_xyz

    # Use voxel-center coordinates for the corners.
    x = [0.0, float(sx - 1)]
    y = [0.0, float(sy - 1)]
    z = [0.0, float(sz - 1)]

    corners = np.array(
        [
            [xx, yy, zz]
            for zz in z
            for yy in y
            for xx in x
        ],
        dtype=np.float64,
    )

    spacing = np.asarray(spacing_xyz, dtype=np.float64)
    origin = np.asarray(origin_xyz, dtype=np.float64)

    physical = origin + corners * spacing

    transformed = np.array(
        [transform.TransformPoint(tuple(p)) for p in physical],
        dtype=np.float64,
    )

    return transformed


# ---------------------------------------------------------------------------
# Output geometry
# ---------------------------------------------------------------------------

def calculate_output_geometry(
    input_size_xyz: Tuple[int, int, int],
    input_spacing_xyz: Tuple[float, float, float],
    input_origin_xyz: Tuple[float, float, float],
    transform: sitk.Transform,
    output_spacing_xyz: Tuple[float, float, float] | None = None,
    padding_voxels: int = 0,
):
    """
    Calculate a bounding-box output geometry containing the transformed
    input volume.

    This is useful when the affine includes translation, rotation, scaling,
    or shear.

    Returns
    -------
    output_size_xyz
    output_spacing_xyz
    output_origin_xyz
    """

    if output_spacing_xyz is None:
        output_spacing_xyz = input_spacing_xyz

    corners = transform_corners(
        transform=transform,
        size_xyz=input_size_xyz,
        spacing_xyz=input_spacing_xyz,
        origin_xyz=input_origin_xyz,
    )

    min_phys = corners.min(axis=0)
    max_phys = corners.max(axis=0)

    spacing = np.asarray(output_spacing_xyz, dtype=np.float64)

    # Add optional padding.
    min_phys -= padding_voxels * spacing
    max_phys += padding_voxels * spacing

    output_size = np.ceil(
        (max_phys - min_phys) / spacing
    ).astype(np.int64) + 1

    return (
        tuple(int(v) for v in output_size),
        tuple(float(v) for v in output_spacing_xyz),
        tuple(float(v) for v in min_phys),
    )


# ---------------------------------------------------------------------------
# Determine source ROI required for an output block
# ---------------------------------------------------------------------------

def source_roi_for_output_block(
    transform: sitk.Transform,
    output_start_xyz: np.ndarray,
    output_size_xyz: np.ndarray,
    output_spacing_xyz: np.ndarray,
    output_origin_xyz: np.ndarray,
    input_spacing_xyz: np.ndarray,
    input_origin_xyz: np.ndarray,
    input_size_xyz: np.ndarray,
    interpolation_radius: int = 2,
):
    """
    Determine the minimum input XYZ voxel ROI required to resample one
    output block.

    SimpleITK ResampleImageFilter expects an output-to-input transform.
    Therefore the inverse transform is used to determine which source
    voxels are required.

    Returns
    -------
    source_start_xyz
    source_size_xyz
    """

    inverse_transform = transform.GetInverse()

    # Output block corner voxel coordinates.
    x0, y0, z0 = output_start_xyz
    nx, ny, nz = output_size_xyz

    x1 = x0 + nx - 1
    y1 = y0 + ny - 1
    z1 = z0 + nz - 1

    corners_index = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x0, y1, z0],
            [x1, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x0, y1, z1],
            [x1, y1, z1],
        ],
        dtype=np.float64,
    )

    # Convert output indices -> physical coordinates.
    output_physical = (
        output_origin_xyz
        + corners_index * output_spacing_xyz
    )

    # Map output physical coordinates -> input physical coordinates.
    input_physical = np.array(
        [
            inverse_transform.TransformPoint(tuple(p))
            for p in output_physical
        ],
        dtype=np.float64,
    )

    # Convert input physical coordinates -> continuous input indices.
    input_indices = (
        input_physical - input_origin_xyz
    ) / input_spacing_xyz

    min_index = np.floor(input_indices.min(axis=0)).astype(np.int64)
    max_index = np.ceil(input_indices.max(axis=0)).astype(np.int64)

    # Add interpolation halo.
    min_index -= interpolation_radius
    max_index += interpolation_radius

    # Clip to actual source volume.
    min_index = np.maximum(min_index, 0)
    max_index = np.minimum(max_index, input_size_xyz - 1)

    source_size = max_index - min_index + 1

    return min_index, source_size


# ---------------------------------------------------------------------------
# Zarr -> SimpleITK
# ---------------------------------------------------------------------------

def zarr_block_to_sitk(
    array,
    start_xyz: np.ndarray,
    size_xyz: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    origin_xyz: Tuple[float, float, float],
) -> sitk.Image:
    """
    Read a Zarr ROI and convert it to SimpleITK.

    Zarr layout:
        [z, y, x]

    SimpleITK layout:
        [x, y, z] physical indexing
    """

    x, y, z = start_xyz
    nx, ny, nz = size_xyz

    # Zarr is indexed Z,Y,X.
    block = np.asarray(
        array[
            z:z + nz,
            y:y + ny,
            x:x + nx,
        ]
    )

    # SimpleITK.GetImageFromArray expects NumPy Z,Y,X
    # for a 3-D image.
    image = sitk.GetImageFromArray(block)

    image.SetSpacing(tuple(spacing_xyz))

    block_origin = (
        origin_xyz[0] + x * spacing_xyz[0],
        origin_xyz[1] + y * spacing_xyz[1],
        origin_xyz[2] + z * spacing_xyz[2],
    )

    image.SetOrigin(block_origin)

    image.SetDirection(
        (
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        )
    )

    return image


# ---------------------------------------------------------------------------
# Resample one block
# ---------------------------------------------------------------------------

def resample_block(
    source_image: sitk.Image,
    transform: sitk.Transform,
    output_size_xyz: Tuple[int, int, int],
    output_spacing_xyz: Tuple[float, float, float],
    output_origin_xyz: Tuple[float, float, float],
    interpolator=sitk.sitkLinear,
    default_value=0,
) -> np.ndarray:
    """
    Resample one output block with SimpleITK.
    """

    resampler = sitk.ResampleImageFilter()

    resampler.SetSize(
        [int(v) for v in output_size_xyz]
    )

    resampler.SetOutputSpacing(
        tuple(float(v) for v in output_spacing_xyz)
    )

    resampler.SetOutputOrigin(
        tuple(float(v) for v in output_origin_xyz)
    )

    resampler.SetOutputDirection(
        (
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        )
    )

    # IMPORTANT:
    #
    # ResampleImageFilter expects OUTPUT -> INPUT.
    #
    # If your registration transform represents:
    #
    #       moving -> fixed
    #
    # and your output grid is fixed, you normally need to supply the
    # corresponding output->moving transform.
    #
    # For the generic chunk implementation below, we therefore invert
    # the transform here.
    resampler.SetTransform(
        transform.GetInverse()
    )

    resampler.SetInterpolator(interpolator)

    resampler.SetDefaultPixelValue(
        float(default_value)
    )

    output = resampler.Execute(source_image)

    return sitk.GetArrayFromImage(output)


# ---------------------------------------------------------------------------
# Main Zarr processing function
# ---------------------------------------------------------------------------

def apply_sitk_transform_to_zarr(
    input_zarr: str,
    output_zarr: str,
    transform: sitk.Transform,
    input_spacing_xyz: Tuple[float, float, float],
    output_spacing_xyz: Tuple[float, float, float] | None = None,
    input_origin_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    output_origin_xyz: Tuple[float, float, float] | None = None,
    output_size_xyz: Tuple[int, int, int] | None = None,
    output_chunk_xyz: Tuple[int, int, int] = (512, 512, 32),
    interpolator=sitk.sitkLinear,
    default_value=0,
    padding_voxels: int = 0,
    overwrite: bool = False,
    debug: bool = False
):
    """
    Apply a SimpleITK affine transform to a huge Zarr volume.

    Parameters
    ----------
    input_zarr:
        Path to input Zarr array.

    output_zarr:
        Path to output Zarr array.

    transform:
        SimpleITK transform.

    input_spacing_xyz:
        Physical spacing of input volume.

    output_spacing_xyz:
        Spacing of output volume. Defaults to input spacing.

    input_origin_xyz:
        Physical origin of input volume.

    output_origin_xyz:
        Output physical origin. If None, automatically calculated from
        transformed input bounding box.

    output_size_xyz:
        Output XYZ dimensions. If None, automatically calculated from
        transformed bounding box.

    output_chunk_xyz:
        Output block size in XYZ.

    interpolator:
        SimpleITK interpolator.

    default_value:
        Value used outside source image.

    padding_voxels:
        Padding around transformed bounding box.

    overwrite:
        Remove/recreate output Zarr if it already exists.
    """

    # ---------------------------------------------------------------
    # Open Zarr
    # ---------------------------------------------------------------

    source_root = zarr.open(input_zarr, mode="r")
    print(source_root.info)

    if hasattr(source_root, "shape"):
        source = source_root
    else:
        # Assume dataset named "0" if root is a Zarr group.
        if "0" in source_root:
            source = source_root["0"]
        else:
            raise ValueError(
                "Input Zarr is a group. Specify the array or use "
                "a group containing dataset '0'."
            )

    if source.ndim != 3:
        raise ValueError(
            f"Expected a 3-D Zarr array, got {source.ndim} dimensions."
        )

    # Zarr shape is Z,Y,X.
    nz, ny, nx = source.shape

    input_size_xyz = np.array(
        [nx, ny, nz],
        dtype=np.int64,
    )

    input_spacing = np.asarray(
        input_spacing_xyz,
        dtype=np.float64,
    )

    input_origin = np.asarray(
        input_origin_xyz,
        dtype=np.float64,
    )

    # ---------------------------------------------------------------
    # Output geometry
    # ---------------------------------------------------------------

    if output_spacing_xyz is None:
        output_spacing_xyz = input_spacing_xyz

    if output_size_xyz is None or output_origin_xyz is None:

        calculated_size, calculated_spacing, calculated_origin = (
            calculate_output_geometry(
                input_size_xyz=tuple(input_size_xyz),
                input_spacing_xyz=input_spacing_xyz,
                input_origin_xyz=input_origin,
                transform=transform,
                output_spacing_xyz=output_spacing_xyz,
                padding_voxels=padding_voxels,
            )
        )

        if output_size_xyz is None:
            output_size_xyz = calculated_size

        if output_origin_xyz is None:
            output_origin_xyz = calculated_origin

    output_size = np.asarray(
        output_size_xyz,
        dtype=np.int64,
    )

    output_spacing = np.asarray(
        output_spacing_xyz,
        dtype=np.float64,
    )

    output_origin = np.asarray(
        output_origin_xyz,
        dtype=np.float64,
    )

    print("Input:")
    print(f"  Zarr shape Z,Y,X = {source.shape}")
    print(f"  XYZ size         = {tuple(input_size_xyz)}")
    print(f"  spacing          = {tuple(input_spacing)}")
    print(f"  origin            = {tuple(input_origin)}")

    print("\nOutput:")
    print(f"  XYZ size         = {tuple(output_size)}")
    print(f"  ZYX size         = "
          f"({output_size[2]}, {output_size[1]}, {output_size[0]})")
    print(f"  spacing          = {tuple(output_spacing)}")
    print(f"  origin           = {tuple(output_origin)}")

    if debug:
        exit(0)

    # ---------------------------------------------------------------
    # Create output Zarr
    # ---------------------------------------------------------------

    output_path = Path(output_zarr)

    if output_path.exists() and overwrite:
        import shutil
        shutil.rmtree(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_zarr} already exists. Use overwrite=True."
        )

    # Zarr uses Z,Y,X ordering.
    output_shape_zyx = (
        int(output_size[2]),
        int(output_size[1]),
        int(output_size[0]),
    )

    chunk_zyx = (
        int(output_chunk_xyz[2]),
        int(output_chunk_xyz[1]),
        int(output_chunk_xyz[0]),
    )

    output = zarr.open(
        output_zarr,
        mode="w",
        shape=output_shape_zyx,
        chunks=chunk_zyx,
        dtype=source.dtype,
        compressor=None,
    )

    # Store geometry as Zarr attributes.
    output.attrs["spacing_xyz"] = [
        float(v) for v in output_spacing
    ]

    output.attrs["origin_xyz"] = [
        float(v) for v in output_origin
    ]

    output.attrs["direction"] = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]

    output.attrs["source_shape_zyx"] = list(source.shape)
    output.attrs["transform_type"] = transform.GetName()

    # ---------------------------------------------------------------
    # Process output blocks
    # ---------------------------------------------------------------

    cx, cy, cz = output_chunk_xyz

    nx_out, ny_out, nz_out = output_size

    nblocks_x = math.ceil(nx_out / cx)
    nblocks_y = math.ceil(ny_out / cy)
    nblocks_z = math.ceil(nz_out / cz)

    total_blocks = (
        nblocks_x *
        nblocks_y *
        nblocks_z
    )

    print(
        f"\nProcessing {total_blocks:,} output blocks..."
    )

    block_number = 0

    for z0 in tqdm(
        range(0, nz_out, cz),
        desc="Z blocks",
    ):

        bz = min(cz, nz_out - z0)

        for y0 in range(0, ny_out, cy):

            by = min(cy, ny_out - y0)

            for x0 in range(0, nx_out, cx):

                bx = min(cx, nx_out - x0)

                output_start = np.array(
                    [x0, y0, z0],
                    dtype=np.int64,
                )

                output_block_size = np.array(
                    [bx, by, bz],
                    dtype=np.int64,
                )

                # ---------------------------------------------------
                # Find source ROI needed by this output block.
                # ---------------------------------------------------

                source_start, source_size = (
                    source_roi_for_output_block(
                        transform=transform,
                        output_start_xyz=output_start,
                        output_size_xyz=output_block_size,
                        output_spacing_xyz=output_spacing,
                        output_origin_xyz=output_origin,
                        input_spacing_xyz=input_spacing,
                        input_origin_xyz=input_origin,
                        input_size_xyz=input_size_xyz,
                        interpolation_radius=2,
                    )
                )

                # ---------------------------------------------------
                # Read only the required source ROI.
                # ---------------------------------------------------

                source_image = zarr_block_to_sitk(
                    array=source,
                    start_xyz=source_start,
                    size_xyz=source_size,
                    spacing_xyz=tuple(input_spacing),
                    origin_xyz=tuple(input_origin),
                )

                # ---------------------------------------------------
                # Physical origin of this output block.
                # ---------------------------------------------------

                block_origin = (
                    output_origin
                    + output_start * output_spacing
                )

                # ---------------------------------------------------
                # Resample with SimpleITK.
                # ---------------------------------------------------

                block = resample_block(
                    source_image=source_image,
                    transform=transform,
                    output_size_xyz=tuple(
                        int(v) for v in output_block_size
                    ),
                    output_spacing_xyz=tuple(
                        float(v) for v in output_spacing
                    ),
                    output_origin_xyz=tuple(
                        float(v) for v in block_origin
                    ),
                    interpolator=interpolator,
                    default_value=default_value,
                )

                # ---------------------------------------------------
                # Write block back to Zarr.
                #
                # SimpleITK returns Z,Y,X.
                # ---------------------------------------------------

                output[
                    z0:z0 + bz,
                    y0:y0 + by,
                    x0:x0 + bx,
                ] = block

                block_number += 1

    print("\nFinished.")
    print(f"Output written to: {output_zarr}")


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
    reg_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration"
    moving_zarr_path = os.path.join(base_path, moving, 'preps', 'C1', f'thumbnail_aligned.{downsample}.zarr')
    output_zarr = os.path.join(base_path, moving, 'preps', 'C1', f'{moving}_{fixed}_registered.zarr')
    xy_resolution = 0.325
    full_xy_resolution = xy_resolution * 64
    ds_xy_resolution = xy_resolution * downsample
    z_resolution = 20.0
    transform_path = os.path.join(reg_path, f"{moving}_{fixed}.tfm")

    if not os.path.exists(moving_zarr_path):
        print(f'Missing input zarr: {moving_zarr_path}')
        exit(0)
    if not os.path.exists(transform_path):
        print(f'Missing input zarr: {transform_path}')
        exit(0)


    # ------------------------------------------------------------
    # Load the affine transform that was generated during
    # registration.
    # ------------------------------------------------------------

    transform = sitk.ReadTransform(transform_path)


    # ------------------------------------------------------------
    # Large Zarr volume
    #
    # shape:
    #   Z = 485
    #   Y = 35500
    #   X = 65500
    #
    # spacing:
    #   X = 0.325 um
    #   Y = 0.325 um
    #   Z = 20 um
    # ------------------------------------------------------------

    apply_sitk_transform_to_zarr(
        input_zarr=moving_zarr_path,
        output_zarr=output_zarr,
        transform=transform,
        input_spacing_xyz=(0.325*downsample,0.325*downsample,20.0),
        output_spacing_xyz=(0.325*downsample, 0.325*downsample, 20.0),
        # 1024 x 1024 x 16 output blocks.
        #
        # The code only loads the source region required for
        # each block.
        output_chunk_xyz=(1024,1024,16),
        interpolator=sitk.sitkLinear,
        default_value=0,
        padding_voxels=10,
        overwrite=True,
        debug=debug
    )