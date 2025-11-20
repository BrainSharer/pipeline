import os
import SimpleITK as sitk
import numpy as np

# ---------------------------
# Example usage
# ---------------------------
# 1) load images and transform
regpath = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration"
fixed_path = os.path.join(regpath, "Allen", "Allen_10.0x10.0x10.0um_sagittal.nii")
if os.path.exists(fixed_path):
    fixed = sitk.ReadImage(fixed_path)
    print(f'Loading {fixed_path}')
    print(f'Fixed image size: {fixed.GetSize()}, spacing: {fixed.GetSpacing()}')
else:
    print(f'Could not load fixed image at {fixed_path}. Please check the path.  ')
    exit(1)

moving_path = os.path.join(regpath, "DK55", "DK55_10.4x10.4x20um_sagittal.nii")
if os.path.exists(moving_path):
    print(f'Loading {moving_path}')
    moving = sitk.ReadImage(moving_path)
    print(f'Moving image size: {moving.GetSize()}, spacing: {moving.GetSpacing()}')
else:
    print(f'Could not load moving image at {moving_path}. Please check the path.  ')
    exit(1)

#moving_img = sitk.ReadImage("moving_original.nii.gz")   # original moving image (spacing 0.325,0.325,20)
#fixed_img  = sitk.ReadImage("fixed_resampled_10um.nii.gz")  # the fixed image used in registration (10um spacing)
# If you have the transform object (SimpleITK.Transform) saved as a file:
transform_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/DK55/DK55_Allen_10.0x10.0x10.0um.tfm"
if os.path.exists(transform_path):
    print(f'Loading transform from {transform_path}')
    transform = sitk.ReadTransform(transform_path)
else:
    print(f'Could not load transform at {transform_path}. Please check the path.  ')
    exit(1)
# OR if you have the transform object in memory, use it directly

# 2) your fiducial points
# Example: points provided as voxel indices in moving image native resolution
# points
locations = [" 1. mid front tip"," 2. mid top"," 3. mid back bottom cerebellum",
             " 4. mid bottom in crux"," 5. left top"," 6. right bottom"]

moving_index = [
    [410, 738, 242],
    [1190, 416, 242],
    [1751, 766, 242],
    [1265, 939, 242],
    [1163, 616, 10],
    [1152, 820, 478]
]

# Moving image spacing (µm)
moving_spacing = (10.4, 10.4, 20.0)
# If you know moving image origin/direction, set them here:
moving_origin = (0.0, 0.0, 0.0)         # <-- change if different
moving_direction = (1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0)  # row-major 3x3


# --- Convert index -> physical (respecting origin/direction) ---
# If you have the actual moving SimpleITK image used in registration, prefer:
# moving_img = sitk.ReadImage("moving_used_in_registration.nii")
# phys_moving = moving_img.TransformContinuousIndexToPhysicalPoint(moving_index)

# Otherwise construct by hand (assumes direction is identity if you keep default)
ix, iy, iz = moving_index[0]
sx, sy, sz = moving_spacing
ox, oy, oz = moving_origin

# For general direction you should apply direction matrix; if identity:
phys_moving = (ix * sx, iy * sy, iz * sz)

print("Moving physical (µm):", phys_moving)


# Apply transform: physical (moving) -> physical (fixed)
phys_fixed = transform.TransformPoint(phys_moving)
print("Mapped moving (µm):", phys_fixed)


# Convert fixed physical -> fixed image index (continuous index -> round/clamp as you want)
try:
    fixed_index = fixed.TransformPhysicalPointToIndex(phys_fixed)   # integer index (raises if outside)
    print("Fixed index (integer):", fixed_index)
except RuntimeError:
    fixed_index_cont = fixed.TransformPhysicalPointToContinuousIndex(phys_fixed)
    fixed_index_rounded = tuple(int(round(v)) for v in fixed_index_cont)
    print("Fixed continuous index (outside bounds):", fixed_index_cont)
    print("Fixed index (rounded):", fixed_index_rounded)