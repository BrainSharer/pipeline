# antspy_affine_registration.py
import os
import pandas as pd
import numpy as np
import ants
import tifffile
import shutil

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
    moving_image_path = os.path.join(regpath, moving_brain, "DK55_10.4x10.4x20um_sagittal.nii")
    fixed_image_path = os.path.join(regpath, "Allen", f"Allen_{um}x{um}x{um}um_sagittal.nii")
    if not os.path.exists(moving_image_path):
        print("Moving image NIfTI not found: " + moving_image_path)
        exit(1)
    if not os.path.exists(fixed_image_path):
        print("Fixed image NIfTI not found: " + fixed_image_path)
        exit(1)
    transform_path = os.path.join(regpath, moving_brain, f"{moving_brain}ToAllen_affine.mat")
    inv_transform_path = os.path.join(regpath, moving_brain, f"AllenTo{moving_brain}_affine.mat")
    moving_spacing = (10.4, 10.4, 20.0)  # (x,y,z) microns
    moving_points = np.array([
        (1062, 1062, 130),
        (1311, 644, 240)
    ])

    coords_are_voxel_indices = True

    # Whether to resample the moving image onto the fixed image grid before registration.
    # This often simplifies interpretation when atlas and subject spacings differ.
    resample_moving_to_fixed = True

    # Type of registration. We produce an AFFINE transform as requested.
    registration_type = "Affine"  # options include 'Rigid', 'Affine', 'SyN', etc.
    print("Reading images...")
    fixed = ants.image_read(fixed_image_path)
    moving = ants.image_read(moving_image_path)
    print("Fixed image spacing:", fixed.spacing)
    print("Moving image spacing:", moving.spacing)
    print("Fixed image shape:", fixed.shape)
    print("Moving image shape:", moving.shape)   
    if resample_moving_to_fixed:
        print("Resampling moving image to fixed image grid/spacing...")
        # ants.resample_image_to_target will resample 'moving' to have the same spacing/size/origin as 'fixed'
        moving_rs = ants.resample_image_to_target(moving, fixed, interp_type='linear')
    else:
        moving_rs = moving     
    print(f"Running ANTs registration (type_of_transform='{registration_type}')")
    reg_affine = ants.registration(fixed=fixed,
                        moving=moving_rs,
                        type_of_transform=registration_type)  # will produce a dict of outputs


    print(f"{registration_type} registration complete.")

    # Save transforms (forward transforms are filepaths)
    print("Transforms saved by ANTs:", reg_affine['fwdtransforms'])
    shutil.copyfile(reg_affine['fwdtransforms'][0], transform_path)
    shutil.copyfile(reg_affine['invtransforms'][0], inv_transform_path)
    print("Wrote transform to:", transform_path)

    # Transform fiducial points:
    # ants.apply_transforms_to_points expects points as Nx3 numpy (with columns x,y,z)
    # Convert to DataFrame as required by ANTs
    pts = pd.DataFrame(moving_points, columns=['x', 'y', 'z'])
    print("Moving points DataFrame:\n", pts.head())
    if coords_are_voxel_indices:
        print("Converting fiducial voxel indices -> physical coordinates (moving image)...")
        physical_points = []
        for idx in moving_points:
            # ants.transform_index_to_physical_point expects integer indices or a tuple
            # Use the original moving image coordinates (not resampled) so index mapping is correct
            pt_phys = ants.transform_index_to_physical_point(moving, tuple(map(int, idx)))
            physical_points.append(pt_phys)
        physical_points = np.array(physical_points)  # shape (N,3)
    else:
        # assume the array already contains physical coordinates
        physical_points = moving_points

    print("Physical coordinates of fiducials (moving space):")
    for p in physical_points:
        print(p)