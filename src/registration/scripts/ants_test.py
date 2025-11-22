# antspy_affine_registration.py
import os
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
    moving_tif_folder = "moving_tifs/"
    fixed_image_path = "allen_10um.nii.gz"
    moving_spacing = (10.4, 10.4, 20.0)  # (x,y,z) microns
    moving_fiducials = [
        (150.3, 200.1, 10.0),
        (300.0, 120.0, 12.0),
        (400.5, 350.2, 20.0),
    ]
    out_prefix = "ants_moving_to_allen"

    print("Loading...")
    stack = load_tif_stack_numpy(moving_tif_folder)
    moving = ants_image_from_numpy(stack, spacing=moving_spacing)
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

    # Transform fiducial points:
    # ants.apply_transforms_to_points expects points as Nx3 numpy (with columns x,y,z)
    pts = np.array(moving_fiducials)
    # Apply transforms to points. The transform maps moving->fixed. Use same transforms used to warp moving.
    transformed_pts = ants.apply_transforms_to_points(3, pts, transformlist=reg_affine['fwdtransforms'], whichtoinvert=None)
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
