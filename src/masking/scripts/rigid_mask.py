"""
Rigid registration of two microscopy TIFFs using SimpleElastix (SimpleITK).
- Auto-generates tissue masks from grayscale images (adaptive/local thresholding + morphological cleanup).
- Uses masks for Elastix (fixed & moving mask).
- Adjusts sampling (NumberOfSpatialSamples) to be robust to missing/damaged tissue.
- Outputs registered image and transform parameter map.

Dependencies (recommended):
    pip install SimpleITK scikit-image numpy scipy opencv-python

If scikit-image or opencv is missing, the code falls back to SimpleITK's Otsu thresholding.
"""

import os
import numpy as np
import SimpleITK as sitk

# Optional dependencies
try:
    from skimage.filters import threshold_local
    from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

from scipy import ndimage as ndi


def read_image(path):
    """Read image with SimpleITK (keeps original pixel type)."""
    img = sitk.ReadImage(path)
    # If multi-channel, convert to single-channel grayscale by extracting first channel
    if img.GetNumberOfComponentsPerPixel() > 1:
        img = sitk.VectorIndexSelectionCast(img, 0)
    return img


def generate_tissue_mask(sitk_image, block_size=51, offset=10, min_area=500):
    """
    Generate a binary tissue mask from a grayscale SimpleITK image.
    Attempts methods in order:
        1) skimage.threshold_local (adaptive)
        2) cv2.adaptiveThreshold
        3) SimpleITK Otsu (global)
    Then does morphological cleanup (opening/closing) and removes small objects.

    Parameters:
        sitk_image: SimpleITK.Image (grayscale)
        block_size: int, neighborhood for local threshold (must be odd)
        offset: int/float, subtracted from local mean for thresholding
        min_area: int, minimum connected component area (in pixels)
    Returns:
        mask_sitk: SimpleITK.Image (binary, unsigned char)
    """
    arr = sitk.GetArrayFromImage(sitk_image).astype(np.float32)  # z,y,x or y,x
    # If 3D volume, collapse to max projection for mask generation (common for microscopy slices)
    if arr.ndim == 3:
        # treat as z,y,x -> collapse along z to create 2D mask projection
        arr2d = np.max(arr, axis=0)
    else:
        arr2d = arr

    # Normalize to 0-255 for cv2
    def _to_uint8(a):
        a = a - np.nanmin(a)
        if np.nanmax(a) > 0:
            a = a / np.nanmax(a)
        a8 = (a * 255).astype(np.uint8)
        return a8

    mask2d = None

    if SKIMAGE_AVAILABLE:
        try:
            # threshold_local expects odd block_size
            if block_size % 2 == 0:
                block_size += 1
            th = threshold_local(arr2d, block_size, offset=offset)
            mask2d = arr2d > th
        except Exception:
            mask2d = None

    if mask2d is None and CV2_AVAILABLE:
        try:
            img8 = _to_uint8(arr2d)
            # cv2.adaptiveThreshold requires single-channel 8-bit image
            mask_cv = cv2.adaptiveThreshold(img8, 255,
                                            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            thresholdType=cv2.THRESH_BINARY,
                                            blockSize=block_size if block_size % 2 == 1 else block_size + 1,
                                            C=offset)
            mask2d = mask_cv > 0
        except Exception:
            mask2d = None

    if mask2d is None:
        # fallback: SimpleITK Otsu on the largest-projection image
        try:
            sitk_proj = sitk.GetImageFromArray(arr2d.astype(arr2d.dtype))
            otsu = sitk.OtsuThreshold(sitk_proj, 0, 1)
            mask2d = sitk.GetArrayFromImage(otsu).astype(bool)
        except Exception:
            # final fallback: simple global threshold at mean
            mask2d = arr2d > (np.nanmean(arr2d) * 0.5)

    # Morphological cleanup: open then close
    # Convert to boolean
    mask2d = mask2d.astype(bool)

    # Use scipy.ndimage binary operations (works for 2D)
    # First remove small holes: binary_closing then remove small objects
    mask2d = ndi.binary_opening(mask2d, structure=np.ones((3, 3)))
    mask2d = ndi.binary_closing(mask2d, structure=np.ones((5, 5)))

    # Remove small objects
    if min_area > 0:
        label_im, num = ndi.label(mask2d)
        sizes = ndi.sum(mask2d, label_im, range(1, num + 1))
        mask_clean = np.zeros_like(mask2d)
        for i, s in enumerate(sizes, start=1):
            if s >= min_area:
                mask_clean[label_im == i] = True
        mask2d = mask_clean

    # If original was 3D, lift mask back to 3D by repeating along z
    if arr.ndim == 3:
        z = arr.shape[0]
        mask3d = np.repeat(mask2d[np.newaxis, :, :], z, axis=0)
        mask = mask3d
    else:
        mask = mask2d

    # Convert back to SimpleITK (uint8)
    mask_sitk = sitk.GetImageFromArray((mask.astype(np.uint8)))
    # Ensure same spacing/origin/direction as input image (only if shapes match)
    try:
        if mask_sitk.GetSize() == sitk_image.GetSize():
            mask_sitk.CopyInformation(sitk_image)
        else:
            # If sizes don't match (common if collapsed), resample mask to image space
            mask_sitk = sitk.Resample(mask_sitk, sitk_image,
                                      sitk.Transform(), sitk.sitkNearestNeighbor,
                                      mask_sitk.GetOrigin(), mask_sitk.GetSpacing(),
                                      mask_sitk.GetDirection(), 0, sitk.sitkUInt8)
            mask_sitk.CopyInformation(sitk_image)
    except Exception:
        # best effort: set origin/spacing
        mask_sitk.CopyInformation(sitk_image)

    return mask_sitk


def cleanup_mask(mask_sitk, structure_radius=3, min_area=100):
    """Extra morphological cleanup for a binary SimpleITK mask using binary morphology and small-object removal."""
    arr = sitk.GetArrayFromImage(mask_sitk).astype(bool)
    # 2D or 3D cleanup
    selem = np.ones((2 * structure_radius + 1,) * (arr.ndim))
    arr = ndi.binary_opening(arr, structure=selem)
    arr = ndi.binary_closing(arr, structure=selem)
    # remove small connected components
    label_im, num = ndi.label(arr)
    sizes = ndi.sum(arr, label_im, range(1, num + 1))
    cleaned = np.zeros_like(arr)
    for i, s in enumerate(sizes, start=1):
        if s >= min_area:
            cleaned[label_im == i] = 1
    return sitk.GetImageFromArray(cleaned.astype(np.uint8))


def rigid_registration_with_masks(fixed_path,
                                  moving_path,
                                  output_prefix="reg_output",
                                  sample_fraction=0.02,
                                  block_size=51,
                                  offset=10,
                                  min_mask_area=500,
                                  debug=False):
    """
    Perform rigid registration with SimpleElastix, generating masks automatically and using them.

    Parameters:
        fixed_path, moving_path: paths to TIFF images
        output_prefix: prefix for output files (registered image and transform)
        sample_fraction: fraction of total voxels to use for spatial sampling (NumberOfSpatialSamples)
                         (use small fraction for big volumes; increase if registration unstable)
        block_size, offset: parameters for adaptive thresholding
        min_mask_area: minimum connected component area to keep in masks (pixels)
        debug: if True, write intermediate masks to disk
    Returns:
        result_image: SimpleITK.Image (registered moving image in fixed space)
        elastix_filter: the ElastixImageFilter after execution (can be inspected)
    """

    fixed = read_image(fixed_path)
    moving = read_image(moving_path)

    # If images are 3D volumes, we assume same orientation/spacing or resample externally beforehand

    # Generate masks
    fixed_mask_guess = generate_tissue_mask(fixed, block_size=block_size, offset=offset, min_area=min_mask_area)
    moving_mask_guess = generate_tissue_mask(moving, block_size=block_size, offset=offset, min_area=min_mask_area)

    # Additional cleanup
    fixed_mask = cleanup_mask(fixed_mask_guess, structure_radius=3, min_area=min_mask_area)
    moving_mask = cleanup_mask(moving_mask_guess, structure_radius=3, min_area=min_mask_area)

    #sitk.WriteImage(fixed_mask, output_prefix + "_fixed_mask.tif")
    #sitk.WriteImage(moving_mask, output_prefix + "_moving_mask.tif")
    sitk.WriteImage(sitk.Cast(fixed, sitk.sitkUInt16), "007.tif")
    #sitk.WriteImage(sitk.Cast(moving, sitk.sitkUInt16), output_prefix + "_moving_uint16.tif")

    # Prepare Elastix (SimpleElastix / ElastixImageFilter)
    elastix = sitk.ElastixImageFilter()
    #elastix.SetFixedImage(fixed)
    #elastix.SetMovingImage(moving)
    elastix.SetFixedImage(fixed_mask)
    elastix.SetMovingImage(moving_mask)

    # Use masks
    #elastix.SetFixedMask(fixed_mask)
    #elastix.SetMovingMask(moving_mask)

    # Get default rigid parameter map
    param_map = sitk.GetDefaultParameterMap("rigid")
    # Adjust sampling to be robust to missing/damaged tissue.
    # Compute desired NumberOfSpatialSamples as fraction of total voxels in the fixed image.
    total_voxels = np.prod(fixed.GetSize())
    n_samples = int(max(1000, min(int(total_voxels * sample_fraction), 2_000_000)))
    # Set number of spatial samples and sampler type
    param_map["NumberOfSpatialSamples"] = [str(n_samples)]
    param_map["ImageSampler"] = ["RandomCoordinate"]  # random sampling is more robust when tissue missing
    # Make sure we use a smaller maximum step length / more iterations if needed:
    # (these are default-ish for rigid but can be tuned)
    param_map["MaximumNumberOfIterations"] = ["1500"]
    param_map["ResultImageFormat"] = ["tif"]
    param_map["AutomaticTransformInitialization"] = ["true"]
    param_map["AutomaticTransformInitializationMethod"] = ["CenterOfGravity"]

    # param_map["FinalBSplineInterpolationOrder"] = ["1"]
    # Keep metric Rigid typical: use AdvancedMattesMutualInformation if present else Mattes
    # (we leave metric from default "rigid" param_map)

    print("Fixed size", fixed.GetSize(), "total voxels", total_voxels)
    print("Using NumberOfSpatialSamples =", param_map["NumberOfSpatialSamples"])
    print("Parameter map keys:", list(param_map.keys()))

    # Set parameter map to elastix
    elastix.SetParameterMap(param_map)
    elastix.PrintParameterMap()

    # Execute
    print("Starting Elastix rigid registration...")
    elastix.LogToConsoleOff()
    elastix.SetOutputDirectory(output_prefix + "_elastix_out")
    elastix.Execute()

    result_image = elastix.GetResultImage()

    # Save result image
    registered_path = "006.tif"
    sitk.WriteImage(sitk.Cast(result_image, sitk.sitkUInt16), registered_path)

    # Save transform parameter map(s)
    transform_maps = elastix.GetTransformParameterMap()
    # transform_maps is a list of parameter maps; write them into text files
    for i, tmap in enumerate(transform_maps):
        out_file = os.path.join(elastix.GetOutputDirectory(), f"TransformParameters.{i}.txt")
        with open(out_file, "w") as fh:
            for k, v in tmap.items():
                fh.write(f"({k} {' '.join(v)})\n")
        if debug:
            print("Wrote transform parameter map to:", out_file)

    return result_image, elastix


def rigid_registration_elastix(fixed_path, moving_path, output_dir="./elastix_output"):
    """
    Perform rigid registration between two TIF microscopy images using SimpleElastix.
    Returns rotation angle (degrees), x/y translation (pixels), and final metric value.
    """

    # --- Load TIFs ---
    fixed = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)

    # --- Normalize intensities to make registration more stable ---
    def normalize(img):
        arr = sitk.GetArrayFromImage(img)
        arr = (arr - np.percentile(arr, 1)) / (np.percentile(arr, 99) - np.percentile(arr, 1))
        arr = np.clip(arr, 0, 1)
        return sitk.GetImageFromArray(arr.astype(np.float32))
    
    #fixed = normalize(fixed)
    #moving = normalize(moving)

    # --- Optional: Create masks to ignore missing tissue (low intensity areas) ---
    def create_mask(img, threshold=0.5):
        arr = sitk.GetArrayFromImage(img)
        mask = (arr > threshold).astype(np.uint8)
        mask[mask > 0] = 255
        mask_img = sitk.GetImageFromArray(mask)
        mask_img.CopyInformation(img)
        return mask_img

    fixed_mask = create_mask(fixed)
    moving_mask = create_mask(moving)
    sitk.WriteImage(moving_mask, '006.mask.tif')
    sitk.WriteImage(fixed_mask, '007.mask.tif')

    # --- Setup Elastix ---
    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(fixed)
    elastix.SetMovingImage(moving)
    elastix.SetFixedMask(fixed_mask)
    elastix.SetMovingMask(moving_mask)
    elastix.SetOutputDirectory(output_dir)
    elastix.LogToConsoleOff()

    # --- Rigid registration parameters ---
    rigid_params = sitk.GetDefaultParameterMap("rigid")
    rigid_params["AutomaticTransformInitialization"] = ["true"]
    rigid_params["AutomaticTransformInitializationMethod"] = ["CenterOfGravity"]
    rigid_params["Metric"] = ["AdvancedMattesMutualInformation"]
    rigid_params["NumberOfSamplesForExactGradient"] = ["10000"]
    rigid_params["NumberOfResolutions"] = ["6"]
    rigid_params["MaximumNumberOfIterations"] = ["1500"]
    rigid_params["NumberOfSpatialSamples"] = ["10000"]
    rigid_params["Interpolator"] = ["BSplineInterpolator"]
    rigid_params["FinalBSplineInterpolationOrder"] = ["3"]
    rigid_params["CheckNumberOfSamples"] = ["true"]
    rigid_params["WriteResultImage"] = ["true"]

    elastix.SetParameterMap(rigid_params)
    elastix.Execute()

    # --- Get Transform Parameters ---
    transform_parameter_map = elastix.GetTransformParameterMap()[0]
    params = transform_parameter_map["TransformParameters"]

    # For 2D rigid, parameters = [angle(rad), x_trans, y_trans]
    angle_rad = float(params[0])
    tx = float(params[1])
    ty = float(params[2])
    angle_deg = np.degrees(angle_rad)


    print("Rigid registration complete.")
    print(f"Rotation (deg): {angle_deg:.3f}")
    print(f"Translation X: {tx:.3f}")
    print(f"Translation Y: {ty:.3f}")
    result_image = elastix.GetResultImage()

    # Save result image
    registered_path = "006.tif"
    sitk.WriteImage(sitk.Cast(result_image, sitk.sitkUInt16), registered_path)

    return angle_deg, tx, ty, output_dir


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Rigid registration with SimpleElastix (auto mask generation).")
    p.add_argument("--out", default="rigid_reg", help="Output prefix")
    p.add_argument("--sample_fraction", type=float, default=0.02,
                   help="Fraction of fixed voxels used for spatial sampling (0.01-0.05 typical).")
    p.add_argument("--block", type=int, default=51, help="Block size for adaptive threshold (odd).")
    p.add_argument("--offset", type=float, default=10.0, help="Offset for adaptive threshold.")
    p.add_argument("--min_area", type=int, default=500, help="Minimum mask component area in pixels.")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    data_path = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK79/preps/C1/thumbnail_cleaned"
    fixed_path = os.path.join(data_path, "007.tif")
    moving_path = os.path.join(data_path, "006.tif")

    rigid_registration_elastix(
        fixed_path=fixed_path,
        moving_path=moving_path,
        output_dir="./rigid_output"
    )