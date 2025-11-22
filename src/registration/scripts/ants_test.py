# antspy_affine_registration.py
import os
import pandas as pd
import numpy as np
import ants
import tifffile

def load_tif_stack_numpy(folder):
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                    if f.lower().endswith(('.tif', '.tiff'))])
    if not files:
        raise RuntimeError("No TIFFs found in folder: " + folder)
    imgs = [tifffile.imread(f).astype(np.float32) for f in files]
    stack = np.stack(imgs, axis=0)  # (Z,Y,X)
    # ANTs expects array as (X,Y,Z) or works with ants.from_numpy which expects arr with shape (z,y,x) and spacing param
    return stack

def ants_image_from_numpy(stack, spacing):
    # ANTs: specify spacing argument when creating image
    # ants.from_numpy expects array in (z,y,x) ordering
    im = ants.from_numpy(stack)
    # Set spacing as tuple (sx, sy, sz) in physical units
    im.set_spacing(spacing)
    return im

if __name__ == "__main__":
    um = 25.0
    moving_brain = "DK55"
    regpath = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration"
    #moving_tif_folder = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK55/preps/C1/thumbnail_aligned"
    fixed_image_path = os.path.join(regpath, "Allen", f"Allen_{um}x{um}x{um}um_sagittal.nii")
    if not os.path.exists(fixed_image_path):
        print("Fixed image NIfTI not found: " + fixed_image_path)
        exit(1)
    transform_path = os.path.join(regpath, moving_brain, f"{moving_brain}_to_Allen_affine.mat")
    moving_spacing = (10.4, 10.4, 20.0)  # (x,y,z) microns
    moving_points = np.array([
        (1062, 1062, 130),
        (1311, 644, 240)
    ])

    out_prefix = "ants_moving_to_allen"

    if os.path.exists(transform_path):
        reg_affine = ants.read_transform(transform_path)
        print("Loaded existing transform from:", transform_path)
    else:

        print("Loading...")
        #stack = load_tif_stack_numpy(moving_tif_folder)
        #moving = ants_image_from_numpy(stack, spacing=moving_spacing)
        moving_image_path = os.path.join(regpath, moving_brain, "DK55_10.4x10.4x20um_sagittal.nii")

        print(f"Loading moving image from NIfTI at: {moving_image_path}")
        moving = ants.image_read(moving_image_path)
        print(f"Loading fixed (Allen) image from {fixed_image_path}")
        fixed = ants.image_read(fixed_image_path)

        print("Running ANTs registration (rigid -> affine)...")
        # ants.registration can do multistage in one call with composite transforms, but we request affine specifically
        reg_rigid = ants.registration(fixed=fixed, moving=moving, type_of_transform='Rigid', 
                                    reg_iterations=(100,50,20))
        # initialize moving with rigid warp
        warped_rigid = reg_rigid['warpedmovout']
        # Now affine, using rigid as initial
        reg_affine = ants.registration(fixed=fixed, moving=moving, type_of_transform='Affine',
                                    initial_transform=reg_rigid['fwdtransforms'])  # pass fixed->moving transforms
        # reg_affine['fwdtransforms'] is the list of transform filenames (ants format)
        print("Affine registration complete.")

        # Save transforms (forward transforms are filepaths)
        print("Transforms saved by ANTs:", reg_affine['fwdtransforms'])
        ants.write_transform(reg_affine['fwdtransforms'][0], transform_path)
        print("Wrote transform to:", transform_path)

    # Transform fiducial points:
    # ants.apply_transforms_to_points expects points as Nx3 numpy (with columns x,y,z)
    # Convert to DataFrame as required by ANTs
    pts = pd.DataFrame(moving_points, columns=['x', 'y', 'z'])
    print("Moving points DataFrame:\n", pts.head())

    # Apply transforms to points. The transform maps moving->fixed. Use same transforms used to warp moving.
    #transformed_pts = ants.apply_transforms_to_points(3, pts, transformlist=reg_affine['fwdtransforms'], whichtoinvert=None)
    transformed_pts = ants.apply_transforms_to_points(3, pts, transformlist=reg_affine, whichtoinvert=None)
    print("Points mapped to fixed (physical coordinates):")
    print(transformed_pts)

    # If you want them in fixed voxel indices:
    # Convert physical points to continuous index using fixed origin/spacing
    fixed_spacing = fixed.spacing
    fixed_origin = fixed.origin
    # ANTs uses numpy origin/spacing in image header. Continuous index = (phys - origin) / spacing (but consider direction if present)
    # Here we'll assume identity direction
    fixed_indices = (transformed_pts - np.array(fixed_origin)) / np.array(fixed_spacing)
    print("Fixed-image voxel (continuous) coordinates:")
    print(fixed_indices)

    # If you want to write transforms to custom filenames, reg_affine['fwdtransforms'] are already file paths you can copy/move.
