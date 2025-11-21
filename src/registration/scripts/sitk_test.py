
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
    initial_transform = sitk.CenteredTransformInitializer(fixed_img,
                                                          moving_img,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    # Stage 1: Rigid (Euler3D)
    if verbose: print("-> Running rigid (Euler3D) registration...")
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.01)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,
                                                 minStep=1e-4,
                                                 numberOfIterations=200,
                                                 relaxationFactor=0.5)
    reg.SetInitialTransform(initial_transform, inPlace=False)
    reg.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    rigid_transform = reg.Execute(fixed_img, moving_img)
    if verbose:
        print("Rigid optimizer stop condition:", reg.GetOptimizerStopConditionDescription())
        print("Rigid final metric value:", reg.GetMetricValue())

    # Stage 2: Affine (to allow shears/scales if needed)
    if verbose: print("-> Running affine registration...")
    reg2 = sitk.ImageRegistrationMethod()
    reg2.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg2.SetMetricSamplingStrategy(reg2.RANDOM)
    reg2.SetMetricSamplingPercentage(0.01)
    reg2.SetInterpolator(sitk.sitkLinear)
    reg2.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,
                                                  minStep=1e-5,
                                                  numberOfIterations=300,
                                                  relaxationFactor=0.5)
    reg2.SetInitialTransform(sitk.AffineTransform(rigid_transform), inPlace=False)
    reg2.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    reg2.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    affine_transform = reg2.Execute(fixed_img, moving_img)
    if verbose:
        print("Affine optimizer stop condition:", reg2.GetOptimizerStopConditionDescription())
        print("Affine final metric value:", reg2.GetMetricValue())

    # Return the composite transform as an AffineTransform
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
    moving_tif_folder = f"/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{moving_brain}/preps/C1/thumbnail_aligned"          # folder with tifs (z order)
    fixed_image_path = f"{regpath}/Allen/Allen_{um}x{um}x{um}um_sagittal.nii"      # path to Allen reference (10 um isotropic)
    moving_image_path = os.path.join(regpath, moving_brain, f"{moving_brain}_moving_image.nii")
    # Voxel spacings for moving image (microns)
    if not os.path.exists(moving_tif_folder):
        raise RuntimeError("Moving TIFF folder not found: " + moving_tif_folder)
    if not os.path.exists(fixed_image_path):
        raise RuntimeError("Fixed image NIfTI not found: " + fixed_image_path)
    moving_spacing = (10.4, 10.4, 20.0)         # (x, y, z)
    # Example fiducials in moving image voxel coordinates (x,y,z) - replace with your list
    moving_fiducials = [
        (1094, 1066, 130),
        (1047, 662, 238)
    ]
    out_transform_path = os.path.join(regpath, moving_brain, f"{moving_brain}_to_Allen_affine.tfm")

    # --------------------------------------------
    if os.path.exists(moving_image_path):
        print("Moving image already exists at:", moving_image_path)
        moving = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    else:
        print("Loading moving TIFF stack...")
        vol = load_tif_stack(moving_tif_folder)  # shape (Z, Y, X)
        # SimpleITK expects array shape (z,y,x) -> GetImageFromArray will make sitk image with size (x,y,z)
        moving = numpy_to_sitk(vol, spacing=moving_spacing)
        sitk.WriteImage(moving, moving_image_path)
        print("Wrote moving image to:", moving_image_path)

    print("Moving image size (x,y,z):", moving.GetSize(), "spacing:", moving.GetSpacing())

    print("Loading fixed (Allen) image...")
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
