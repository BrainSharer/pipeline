import os
import SimpleITK as sitk
import numpy as np

def resample_image_to_spacing(image, out_spacing=(10.0,10.0,10.0), interpolator=sitk.sitkLinear):
    """Resample image preserving origin and direction, but changing spacing to out_spacing.
       Returns the resampled image (use same origin/direction as original)."""
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    orig_origin = image.GetOrigin()
    orig_direction = image.GetDirection()

    out_spacing = tuple(float(s) for s in out_spacing)
    out_size = [
        int(round(orig_size[i] * (orig_spacing[i] / out_spacing[i]))) for i in range(3)
    ]
    out_size = tuple(max(1, s) for s in out_size)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize(out_size)
    resampler.SetOutputOrigin(orig_origin)
    resampler.SetOutputDirection(orig_direction)
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(image)

def points_to_physical(points, image, points_in='index'):
    """
    Convert list/array of points into physical coordinates (same units as image spacing).
    points: (N,3) array-like
    image: SimpleITK.Image
    points_in: 'index' if points are voxel indices (i,j,k),
               'physical' if points already are physical coordinates (x,y,z in same units as spacing)
    returns: Nx3 numpy array of physical points
    """
    pts = np.asarray(points, dtype=float)
    phys = []
    if points_in == 'index':
        for p in pts:
            # If integer index use TransformIndexToPhysicalPoint. If non-integer, use ContinuousIndex->PhysicalPoint
            # here we use ContinuousIndexToPhysicalPoint to allow sub-voxel points as well
            phys.append(image.TransformContinuousIndexToPhysicalPoint(tuple(float(x) for x in p)))
    elif points_in == 'physical':
        phys = pts.tolist()
    else:
        raise ValueError("points_in must be 'index' or 'physical'")
    return np.asarray(phys, dtype=float)

def physical_to_fixed_index(phys_points, fixed_image):
    """Convert Nx3 physical points to (continuous) index coordinates in fixed_image"""
    idxs = []
    for p in phys_points:
        # You can use TransformPhysicalPointToIndex (returns int index) or TransformPhysicalPointToContinuousIndex
        idxs.append(fixed_image.TransformPhysicalPointToContinuousIndex(tuple(float(x) for x in p)))
    return np.asarray(idxs, dtype=float)

def transform_points_with_sitk_transform(phys_points, sitk_transform):
    """Apply a SimpleITK transform (sitk.Transform) to an array of physical points."""
    out = []
    for p in phys_points:
        out.append(tuple(sitk_transform.TransformPoint(tuple(float(x) for x in p))))
    return np.asarray(out, dtype=float)

# ---------------------------
# Example usage
# ---------------------------
# 1) load images and transform
regpath = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration"
fixed_path = os.path.join(regpath, "Allen", "Allen_10.0x10.0x10.0um_sagittal.nii")
fixed_img = sitk.ReadImage(fixed_path)

moving_path = os.path.join(regpath, "DK55", "DK55_10.0x10.0x10.0um_sagittal.nii")
moving_img = sitk.ReadImage(moving_path)

#moving_img = sitk.ReadImage("moving_original.nii.gz")   # original moving image (spacing 0.325,0.325,20)
#fixed_img  = sitk.ReadImage("fixed_resampled_10um.nii.gz")  # the fixed image used in registration (10um spacing)
# If you have the transform object (SimpleITK.Transform) saved as a file:
transform_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/DK55/DK55_Allen_10.0x10.0x10.0um.tfm"
sitk_transform = sitk.ReadTransform(transform_path)
# OR if you have the transform object in memory, use it directly

# 2) your fiducial points
# Example: points provided as voxel indices in moving image native resolution
moving_points_indices = [
    (31818/32, 33238/32, 252)
]

# --- Convert moving fiducials to physical coordinates ---
# Option A: If the affine was computed using images resampled to 10um spacing,
#           resample the moving image to the same spacing/origin/direction used in the registration
#           to guarantee the transform expects the same physical coordinate frame.
# If you know the resampled image origin/direction were preserved (common), you can skip resampling.
#print("Resampling moving image to 10um spacing...")
#moving_resampled = resample_image_to_spacing(moving_img, out_spacing=(10.0,10.0,10.0),
#                                            interpolator=sitk.sitkLinear)
#resampled_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/DK55/moving_resampled_10um.nii"

# Convert indices (which were drawn on the original moving image) to physical points **in the resampled image frame**.
# To do that we must map the original indices to physical using original image, then optionally
# if resampling preserved origin/direction, that same physical point is valid in resampled image.
# So a safer approach: convert original indices -> physical via original image, then use that physical point.
print("Converting moving points to physical coordinates...")
moving_phys = points_to_physical(moving_points_indices, moving_img, points_in='index')
del moving_img
# --- Apply transform (transform expects physical coordinates in the moving image frame used for registration) ---
# If the registration used the resampled moving image where the origin/direction = moving_resampled.GetOrigin()/GetDirection(),
# and resampling preserved origin/direction, then moving_phys is in the correct frame and can be transformed directly.
print("Applying transform to moving physical points...")
fixed_phys = transform_points_with_sitk_transform(moving_phys, sitk_transform.GetInverse())

# --- Convert transformed physical points into fixed image indices (continuous index recommended) ---
print("Converting transformed physical points to fixed image indices...")
fixed_indices = physical_to_fixed_index(fixed_phys, fixed_img)

# Print results
for i, (m_idx, m_phys, f_phys, f_idx) in enumerate(zip(moving_points_indices, moving_phys, fixed_phys, fixed_indices)):
    print(f"pt {i}: moving_index={m_idx}, \nmoving_phys={m_phys}, -> fixed_phys={f_phys}\n fixed_cont_index={f_idx}")
