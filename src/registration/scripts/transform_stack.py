"""
rigid_tiff_transform.py

Dependencies:
    pip install tifffile scikit-image numpy

Usage example (at bottom):
    transform_tiff_stack("input.tif", "output.tif", angle_deg=12.5, tx=5, ty=-3)
"""

from typing import Optional, Tuple
import numpy as np
import tifffile
from skimage.transform import AffineTransform, warp
import os


def _apply_rigid_2d(
    image: np.ndarray,
    angle_deg: float,
    tx: float,
    ty: float,
    center: Optional[Tuple[float, float]] = None,
    order: int = 1,
    cval=0,
    preserve_range=True,
) -> np.ndarray:
    """
    Apply 2D rigid transform (rotation + translation) to a single image.

    - image: 2D numpy array
    - angle_deg: rotation angle in degrees (positive = CCW)
    - tx, ty: translations in pixels (x -> columns, y -> rows)
    - center: (cx, cy) pixel coordinates about which to rotate. If None, rotates about image center.
    - order: interpolation order (0=nearest,1=bilinear,2=quadratic,3=cubic)
    - cval: fill value outside input image
    - preserve_range: if True, do not normalize intensity to [0,1]
    """
    if image.ndim != 2:
        raise ValueError("image must be 2D")

    h, w = image.shape
    if center is None:
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    else:
        cx, cy = center

    # Build AffineTransform for rotation about center then translation.
    # Sequence: translate(-center) -> rotate -> translate(center) -> translate(tx,ty)
    t_rotate = AffineTransform(rotation=np.deg2rad(angle_deg), translation=(0, 0))
    t_pre = AffineTransform(translation=(-cx, -cy))
    t_post = AffineTransform(translation=(cx + tx, cy + ty))
    transform = t_pre + t_rotate + t_post  # composition: pre -> rotate -> post

    # warp will call inverse_map on the transform to sample input coords
    warped = warp(
        image,
        inverse_map=transform.inverse,
        order=order,
        mode="constant",
        cval=cval,
        preserve_range=preserve_range,
        clip=False,  # we'll clip/cast later
    )

    return warped


def transform_tiff_stack(
    input_path: str,
    output_path: str,
    angle_deg: float = 0.0,
    tx: float = 0.0,
    ty: float = 0.0,
    center: Optional[Tuple[float, float]] = None,
    order: int = 1,
    cval=0,
    overwrite: bool = False,
):
    """
    Read a multi-page TIFF (or single-page), apply the same 2D rigid transform
    to every slice, and write a new TIFF stack.

    - input_path: path to input TIFF
    - output_path: path for output TIFF (overwritten if overwrite=True)
    - angle_deg, tx, ty: parameters of the rigid transform
    - center: optional rotation center (cx, cy)
    - order: interpolation order (0..3)
    - cval: fill value for outside areas
    - overwrite: if True, overwrite output_path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with tifffile.TiffFile(input_path) as tif:
        pages = tif.series[0].asarray()  # load full stack into memory
        # pages shape: (n_slices, h, w) or (h, w) for single
        data = pages

    # Normalize shape to (n_slices, h, w)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    elif data.ndim == 3:
        # already (Z, H, W)
        pass
    else:
        raise ValueError("Only supports 2D slices or 3D (z, y, x) stacks")

    n_slices, h, w = data.shape
    dtype = data.dtype
    is_integer = np.issubdtype(dtype, np.integer)
    info_min, info_max = np.nanmin(data), np.nanmax(data)

    # We'll preserve range and cast back to original dtype after transform
    transformed_stack = np.empty_like(data, dtype=np.float32)

    for i in range(n_slices):
        img = data[i].astype(np.float32)
        warped = _apply_rigid_2d(
            img,
            angle_deg=angle_deg,
            tx=tx,
            ty=ty,
            center=center,
            order=order,
            cval=cval,
            preserve_range=True,
        )
        transformed_stack[i] = warped

    # Clip & cast back to original dtype
    if is_integer:
        # integer type: clip to valid range
        iinfo = np.iinfo(dtype)
        clipped = np.clip(np.rint(transformed_stack), iinfo.min, iinfo.max).astype(dtype)
    else:
        # float: clip to original observed min/max to be safe (or keep as float)
        clipped = np.clip(transformed_stack, info_min, info_max).astype(dtype)

    # Write stack
    # If original was a single image, write single; else multi-page
    if clipped.shape[0] == 1:
        tifffile.imwrite(output_path, clipped[0])
    else:
        tifffile.imwrite(output_path, clipped, bigtiff=True)

    print(f"Saved transformed stack to: {output_path}")


if __name__ == "__main__":
    # simple example usage
    import argparse

    parser = argparse.ArgumentParser(description="Apply rigid transform to TIFF stack (2D per-slice).")
    parser.add_argument("--angle", type=float, default=0.0, help="Rotation (degrees, CCW)")
    parser.add_argument("--tx", type=float, default=0.0, help="Translation in x (pixels, positive -> right)")
    parser.add_argument("--ty", type=float, default=0.0, help="Translation in y (pixels, positive -> down)")
    parser.add_argument("--center", type=float, nargs=2, help="Rotation center: cx cy")
    parser.add_argument("--order", type=int, default=1, help="Interpolation order (0..3)")
    parser.add_argument("--cval", type=float, default=0.0, help="Fill value for outside areas")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output")
    args = parser.parse_args()

    data_path = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK78/preps/C1"
    input_dir = os.path.join(data_path, "thumbnail_aligned.ratto")
    output_dir = os.path.join(data_path, "thumbnail_aligned")

    for f in sorted(os.listdir(input_dir)):
        input = os.path.join(input_dir, f)
        output = os.path.join(output_dir, f)

        transform_tiff_stack(
            input,
            output,
            angle_deg=args.angle,
            tx=args.tx,
            ty=args.ty,
            center=tuple(args.center) if args.center else None,
            order=args.order,
            cval=args.cval,
            overwrite=args.overwrite,
        )
