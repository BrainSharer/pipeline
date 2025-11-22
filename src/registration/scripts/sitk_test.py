
# simpleitk_affine_registration.py
import os
import numpy as np
import SimpleITK as sitk
import tifffile

def load_tif_stack(folder, pattern=None, dtype=None):
    # Loads all tif files in folder sorted by filename.
    # If your filenames are not lexicographically ordered for z, modify sorting.
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                    if f.lower().endswith(('.tif', '.tiff'))])
    if pattern:
        files = [f for f in files if pattern in os.path.basename(f)]
    if not files:
        raise RuntimeError("No TIFFs found in folder: " + folder)
    # read into numpy stack (z, y, x)
    imgs = [tifffile.imread(f).astype(np.float32) for f in files]
    stack = np.stack(imgs, axis=0)
    return stack  # shape (Z, Y, X)

def numpy_to_sitk(vol, spacing, origin=(0.0,0.0,0.0), direction=None):
    # vol is (Z,Y,X) numpy
    vol = vol.astype(np.float32)
    sitk_img = sitk.GetImageFromArray(vol)   # returns image with same ordering: z,y,x -> sitk expects this
    # spacing in microns for SimpleITK expects (x, y, z)
    # But GetImageFromArray yields pixel order (x,y,z) ordering for SetSpacing: we must pass (sx, sy, sz)
    # SimpleITK image has size (x, y, z) ordering when interacting with spacing
    sitk_img.SetSpacing((spacing[0], spacing[1], spacing[2]))
    sitk_img.SetOrigin(origin)
    if direction is not None:
        sitk_img.SetDirection(direction)
    return sitk_img

def register_affine_rigid(fixed_img, moving_img, verbose=True):
    # Two-stage: first rigid, then affine, returning final transform (sitk.Transform)
    # Use mutual information for multi-modal intensity differences. If same modality, you might use correlation.
    # INITIALIZATION: center transform initializer
    # 3. Initial alignment using center of mass
    # ------------------------------------------------------------
    initial_transform = sitk.CenteredTransformInitializer(fixed_img,
                                                          moving_img,
                                                          sitk.AffineTransform(3),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    print("-> Running affine registration...")
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.01)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,
                                                  minStep=1e-5,
                                                  numberOfIterations=300,
                                                  relaxationFactor=0.5)
    registration.SetInitialTransform(initial_transform, inPlace=False)
    registration.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    affine_transform = registration.Execute(fixed_img, moving_img)
    print("Affine optimizer stop condition:", registration.GetOptimizerStopConditionDescription())
    print("Affine final metric value:", registration.GetMetricValue())
    print('type of transform', type(affine_transform))

    return affine_transform

def save_sitk_transform(transform, filename):
    sitk.WriteTransform(transform, filename)
    print("Wrote transform to:", filename)

def transform_points_sitk(transform, moving_img, fixed_img, points_index_list):
    # points_index_list: list of (x,y,z) in moving image voxel/index coordinates (0-based)
    # Convert index -> physical in moving image, apply transform, optionally map to fixed indices
    out_phys = []
    out_fixed_index = []
    for idx_point in points_index_list:
        # use continuous index -> physical point
        # NOTE: SimpleITK expects index as tuple (i,j,k) where ordering is (x,y,z)
        phys = moving_img.TransformContinuousIndexToPhysicalPoint(tuple(idx_point))
        phys_t = transform.TransformPoint(phys)  # physical coords in fixed space
        # convert physical into fixed continuous index
        fixed_idx = fixed_img.TransformPhysicalPointToContinuousIndex(phys_t)
        out_phys.append(tuple(phys_t))
        out_fixed_index.append(tuple(fixed_idx))
    return out_phys, out_fixed_index

if __name__ == "__main__":
    # ---------------- USER INPUT ----------------
    um = 25.0
    moving_brain = "DK55"
    regpath = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration"
    fixed_image_path = f"{regpath}/Allen/Allen_{um}x{um}x{um}um_sagittal.nii"      # path to Allen reference (10 um isotropic)
    moving_image_path = os.path.join(regpath, moving_brain, "DK55_10.4x10.4x20um_sagittal.nii")
    # Voxel spacings for moving image (microns)
    if not os.path.exists(fixed_image_path):
        raise RuntimeError("Fixed image NIfTI not found: " + fixed_image_path)
    # Example fiducials in moving image voxel coordinates (x,y,z) - replace with your list
    moving_fiducials = [
        (1062, 1062, 130),
        (1311, 644, 240)
    ]
    out_transform_path = os.path.join(regpath, moving_brain, f"{moving_brain}_to_Allen_affine.tfm")

    # --------------------------------------------
    if os.path.exists(moving_image_path):
        print("Loading moving image at:", moving_image_path)
        moving = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    else:
        print(f"No moving image NIfTI found. Loading TIFF stack from folder and creating NIfTI at: {moving_image_path}")
        exit(1)

    moving_spacing = moving.GetSpacing()  # (x,y,z) spacing in microns
    print("Moving image size (x,y,z):", moving.GetSize(), "spacing:", moving_spacing)

    print(f"Loading fixed (Allen) image from {fixed_image_path}")
    fixed = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    print("Fixed image size (x,y,z):", fixed.GetSize(), "spacing:", fixed.GetSpacing())

    print("Running registration (rigid -> affine)...")
    affine_tform = register_affine_rigid(fixed, moving, verbose=True)

    print("Saving transform...")
    save_sitk_transform(affine_tform, out_transform_path)

    print("Transforming fiducial points from moving -> fixed (Allen) ...")
    phys_points, fixed_indices = transform_points_sitk(affine_tform, moving, fixed, moving_fiducials)

    for i, (mp, pp, fi) in enumerate(zip(moving_fiducials, phys_points, fixed_indices)):
        print(f"fid {i}: moving_index={mp} -> fixed_phys={pp} -> fixed_index={fi}")

    print("Done.")
