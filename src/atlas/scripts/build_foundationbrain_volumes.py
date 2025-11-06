import argparse
import json
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import center_of_mass
from pathlib import Path

from typing import List, Dict, Any, Tuple

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.draw import polygon as sk_polygon


PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.atlas.atlas_utilities import register_volume, resample_image
from settings import data_path as DATA_PATH, atlas as ATLAS


def save_volume_origin(animal, structure, volume, xyz_offsets):
    x, y, z = xyz_offsets

    #volume = np.swapaxes(volume, 0, 2)
    volume = np.rot90(volume, axes=(0,1))
    volume = np.flip(volume, axis=0)

    OUTPUT_DIR = os.path.join(DATA_PATH, 'atlas_data', animal)
    volume_filepath = os.path.join(OUTPUT_DIR, 'structure', f'{structure}.npy')
    os.makedirs(os.path.join(OUTPUT_DIR, 'structure'), exist_ok=True)
    np.save(volume_filepath, volume)
    origin_filepath = os.path.join(OUTPUT_DIR, 'origin', f'{structure}.txt')
    os.makedirs(os.path.join(OUTPUT_DIR, 'origin'), exist_ok=True)
    np.savetxt(origin_filepath, (x,y,z))


def load_json_vertices(jsonpath: str) -> List[Dict[str, Any]]:
    
    #sqlController = SqlController(animal)
    contours = []
    if not os.path.exists(jsonpath):
        print(f'{jsonpath} does not exist')
        sys.exit()
    with open(jsonpath) as f:
        raw = json.load(f)

    structures = list(raw.keys())
    scaling_factor = 14.464/25.0  # hardcoded for now
    for structure in structures:
        if structure in ['SC']:
            onestructure = raw[structure]
            print(f'Working on {jsonpath} {structure}')
            for z, points in onestructure.items():
                scaled_points = [[sp[0]*scaling_factor,sp[1]*scaling_factor] for sp in points]
                contours.append({"structure": structure, "z": int(z), "points": scaled_points})
   
    return contours


def rasterize_contours_to_mask(contours: List[Dict[str, Any]], volume_shape: Tuple[int,int,int]) -> np.ndarray:
    """
    Build a 3D binary mask (z,y,x) from contours.
    contours: list of {"structure":str, "z":int, "points":[[x,y], [...]]}
    volume_shape: (z_dim, y_dim, x_dim) - integer tuple

    Returns: numpy array dtype=uint8 with 1 for inside polygons.
    """
    z_dim, y_dim, x_dim = volume_shape
    mask = np.zeros(volume_shape, dtype=np.uint8)

    # group contours by z slice
    from collections import defaultdict
    by_z = defaultdict(list)
    for c in contours:
        by_z[c["z"]].append(c["points"])

    for z_idx, polygon_list in by_z.items():
        if not (0 <= z_idx < z_dim):
            # contour outside volume z-range: skip with warning
            print(f"Warning: contour z={z_idx} outside volume (0..{z_dim-1}), skipping.")
            continue

        # create slice mask for this z
        slice_mask = np.zeros((y_dim, x_dim), dtype=np.uint8)

        for pts in polygon_list:
            if len(pts) < 3:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]

            # Clip coordinates to integer pixel grid
            # Rasterization expects row (y) and col (x)
            rr, cc = sk_polygon(ys, xs, shape=slice_mask.shape)
            slice_mask[rr, cc] = 1

        mask[z_idx] = slice_mask

    return mask


# -----------------------
# Utilities for SimpleITK <-> numpy conversions
# -----------------------

def sitk_to_numpy(img: sitk.Image) -> np.ndarray:
    """Return numpy array in z,y,x order from SimpleITK image."""
    arr = sitk.GetArrayFromImage(img)  # SITK returns array with shape (z,y,x)
    return arr

def numpy_to_sitk(img_np: np.ndarray, reference_img: sitk.Image, is_binary=True) -> sitk.Image:
    """
    Convert numpy array (z,y,x) to a SimpleITK image using the reference_img spacing/origin/direction.
    If is_binary, cast to sitk.sitkUInt8 to save space.
    """
    img_sitk = sitk.GetImageFromArray(img_np.astype(np.uint8 if is_binary else img_np.dtype))
    img_sitk.SetSpacing(reference_img.GetSpacing())
    img_sitk.SetOrigin(reference_img.GetOrigin())
    img_sitk.SetDirection(reference_img.GetDirection())
    return img_sitk


# -----------------------
# Mask averaging + cleanup
# -----------------------

def average_and_binarize(masks_list: List[np.ndarray], threshold: float = 0.5, closing_radius_voxels: int = 2) -> np.ndarray:
    """
    masks_list: list of binary numpy arrays shape (z,y,x)
    threshold: fraction threshold for converting mean -> binary
    closing_radius_voxels: radius for morphological closing to fill holes
    """
    for m in masks_list:
        print("Mask shape:", m.shape, "dtype:", m.dtype, "unique vals:", np.unique(m))


    images = [sitk.GetImageFromArray(img) for img in masks_list]
    reference_image_index, reference_image = max(enumerate(images), key=lambda img: np.prod(img[1].GetSize()))
    # Resample all images to the reference
    resampled_images = [resample_image(img, reference_image) for img in images]
    registered_images = [register_volume(img, reference_image) for img in resampled_images if img != reference_image]

    mean_mask = np.mean(registered_images, axis=0)
    bin_mask = (mean_mask >= threshold).astype(np.uint8)

    output_dir = "./registered_output"
    out_path = os.path.join(output_dir, "bin_mask.nii")
    sitk.WriteImage(sitk.GetImageFromArray(bin_mask), out_path)
    print(f"Wrote binary mask before closing -> {out_path}")



    if closing_radius_voxels > 2222220:
        structure = ndimage.generate_binary_structure(3, 2)
        # approximate spherical by iterative binary_dilation then binary_erosion (closing)
        # use binary_closing with iterations approximating radius
        bin_mask = ndimage.binary_closing(bin_mask, structure=structure, iterations=closing_radius_voxels).astype(np.uint8)

    bin_mask[bin_mask > 0] = 255  # ensure binary
    ids, counts = np.unique(bin_mask, return_counts=True)
    print(f'bin mask shape={bin_mask.shape} dtype={bin_mask.dtype}')
    print('ids', ids)
    print('counts', counts)
    return bin_mask


# -----------------------
# Compute center of mass in physical coordinates
# -----------------------

def com_voxel_to_physical(com_voxel: Tuple[float,float,float], reference_img: sitk.Image) -> Tuple[float,float,float]:
    """
    Convert center-of-mass in voxel coordinates (z,y,x) to physical coordinates (x,y,z).
    SimpleITK expects index order [x,y,z] or continuous index? We will convert:
      - SITK's TransformContinuousIndexToPhysicalPoint expects index (i,j,k) in (x,y,z) order.
    com_voxel is from scipy.ndimage.center_of_mass which returns in (z,y,x) ordering.
    We must convert ordering accordingly.
    """
    zc, yc, xc = com_voxel
    cont_index = [float(xc), float(yc), float(zc)]  # (x,y,z)
    phys = reference_img.TransformContinuousIndexToPhysicalPoint(cont_index)
    return phys  # (x_phys, y_phys, z_phys)


# -----------------------
# Build affine transform from source COM -> target COM
# -----------------------

def build_translation_affine(source_com_phys: Tuple[float,float,float], target_com_phys: Tuple[float,float,float]) -> sitk.AffineTransform:
    """
    Build a 3D affine transform with identity linear part and translation that maps source_com -> target_com.
    The returned transform is in physical space.
    """
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(np.eye(3).ravel().tolist())
    # translation = target - source
    tx = target_com_phys[0] - source_com_phys[0]
    ty = target_com_phys[1] - source_com_phys[1]
    tz = target_com_phys[2] - source_com_phys[2]
    print('translation for affine is {tx=} {ty=} {tz=}')
    transform.SetTranslation((tx, ty, tz))
    return transform


# -----------------------
# Resample mask with transform into fixed image space
# -----------------------

def resample_mask_to_reference(mask_sitk: sitk.Image, ref_img: sitk.Image, transform: sitk.Transform, default_value=0) -> sitk.Image:
    """
    Resample mask_sitk into reference image space using transform.
    Use nearest neighbor interpolation because it's a binary mask.
    """
    resampled = sitk.Resample(mask_sitk, ref_img, transform, sitk.sitkNearestNeighbor, default_value)
    # ensure binary (threshold tiny interpolation artifacts)
    resampled = sitk.BinaryThreshold(resampled, lowerThreshold=1, upperThreshold=255, insideValue=255, outsideValue=0)
    return resampled


# -----------------------
# Top-level pipeline
# -----------------------

def register_subvolumes_to_fixed(
    fixed_image_path: str,
    moving_image_paths: List[str],
    json_paths_per_brain: List[str],
    structure_list: List[str],
    fixed_structure_masks: Dict[str, str] = None,
    output_dir: str = ".",
    average_threshold: float = 0.5
):
    """
    Main pipeline.

    - fixed_image_path: path to fixed SimpleITK image (tif/nifti/...).
    - moving_image_paths: list of 3 file paths to each brain's corresponding image (order matches json_paths_per_brain)
    - json_paths_per_brain: list of 3 json files, each containing contours for potentially multiple structures
    - structure_list: list of structure names to extract & average (e.g., ["sup_colliculus","inf_colliculus"])
    - fixed_structure_masks: optional dict mapping structure name -> path to a binary image for that structure in fixed space.
                             If omitted, script will try to look inside a JSON for the fixed image contours (not implemented here).
    - output_dir: where outputs are written.
    """
    os.makedirs(output_dir, exist_ok=True)

    # load fixed image
    fixed_img = sitk.ReadImage(fixed_image_path)
    fixed_np_shape = sitk_to_numpy(fixed_img).shape  # (z,y,x)

    # load moving images
    if len(moving_image_paths) != len(json_paths_per_brain):
        raise ValueError("moving_image_paths must correspond 1:1 with json_paths_per_brain")

    num_brains = len(json_paths_per_brain)

    moving_imgs = [sitk.ReadImage(p) for p in moving_image_paths]
    moving_shapes = [sitk_to_numpy(mi).shape for mi in moving_imgs]  # each shape is (z,y,x)

    # parse JSONs once
    parsed_jsons = [load_json_vertices(jp) for jp in json_paths_per_brain]

    # For each brain, create per-structure binary masks in its own grid
    # We'll store masks as structure -> list of masks (one per brain)
    structure_masks_per_brain = {s: [] for s in structure_list}

    for b in range(num_brains):
        contours = parsed_jsons[b]
        # group contours by structure for this brain
        from collections import defaultdict
        by_structure = defaultdict(list)
        for c in contours:
            struct = c["structure"]
            if struct in structure_list:
                by_structure[struct].append({"z": c["z"], "points": c["points"]})

        # for structures that are missing for this brain, produce empty mask
        shape = moving_shapes[b]
        for struct in structure_list:
            contours_for_struct = by_structure.get(struct, [])
            if len(contours_for_struct) == 0:
                print(f"Brain {b}: no contours found for structure '{struct}'. Using empty mask.")
                mask = np.zeros(shape, dtype=np.uint8)
            else:
                mask = rasterize_contours_to_mask(contours_for_struct, shape)
                mask[mask > 0] = 255  # ensure binary
                ids, counts = np.unique(mask, return_counts=True)
                print(f"Brain {b}: rasterized mask for structure '{struct}' shape={mask.shape} ids={ids} counts={counts}")
            structure_masks_per_brain[struct].append(mask)

    # For each structure, average across the brains and binarize
    avg_masks = {}
    for struct, masks in structure_masks_per_brain.items():
        # ensure exactly num_brains masks
        if len(masks) != num_brains:
            raise RuntimeError("Mask counts mismatch.")
        avg_mask = average_and_binarize(masks_list=masks, threshold=average_threshold, closing_radius_voxels=2)
        avg_masks[struct] = avg_mask

        # Save averaged mask (in the coordinate system of the first moving image for convenience)
        ref_img_for_struct = moving_imgs[0]  # uses spacing/origin/direction of first brain
        avg_img_sitk = numpy_to_sitk(avg_mask, reference_img=ref_img_for_struct, is_binary=True)
        out_path = os.path.join(output_dir, f"avg_mask_{struct}.nii")
        sitk.WriteImage(avg_img_sitk, out_path)
        print(f"Wrote averaged mask for {struct} -> {out_path}")

    # Compute COMs for averaged moving masks in physical space
    avg_coms_phys = {}
    for struct, avg_mask in avg_masks.items():
        # Compute center of mass in voxel coords (z,y,x)
        com_voxel = ndimage.center_of_mass(avg_mask)
        print(f'COM for {struct} is {com_voxel}')
        if np.isnan(com_voxel[0]):
            raise RuntimeError(f"Structure {struct} averaged mask empty -> cannot compute COM.")
        # Convert using the reference moving image (we used moving_imgs[0] to give spacing/origin)
        phys = com_voxel_to_physical(com_voxel, reference_img=moving_imgs[0])
        avg_coms_phys[struct] = phys
        print(f"{struct} averaged COM (phys): {phys}")

    # Load fixed structure COMs (from fixed binary masks) and compute COMs
    fixed_coms_phys = {}
    for struct in structure_list:
        if fixed_structure_masks and struct in fixed_structure_masks:
            fixed_mask_img = sitk.ReadImage(fixed_structure_masks[struct])
            fixed_mask_np = sitk_to_numpy(fixed_mask_img)
            com_voxel = ndimage.center_of_mass(fixed_mask_np)
            if np.isnan(com_voxel[0]):
                raise RuntimeError(f"Fixed mask for {struct} empty -> cannot compute COM.")
            phys = com_voxel_to_physical(com_voxel, reference_img=fixed_img)
            fixed_coms_phys[struct] = phys
            print(f"{struct} fixed COM (phys): {phys}")
        else:
            raise RuntimeError(f"No fixed structure mask provided for '{struct}'. Provide fixed_structure_masks dict mapping structure->path.")

    # For each structure, build an affine transform mapping averaged moving->fixed based on COM
    transforms_per_structure = {}
    resampled_masks_per_structure = {}
    for struct in structure_list:
        src = avg_coms_phys[struct]
        tgt = fixed_coms_phys[struct]
        transform = build_translation_affine(src, tgt)
        transforms_per_structure[struct] = transform

        # convert averaged mask (which is in the grid of moving_imgs[0]) to sitk and resample to fixed image space
        avg_mask_img = numpy_to_sitk(avg_masks[struct], reference_img=moving_imgs[0], is_binary=True)
        # Because our transform maps moving_phys -> fixed_phys, but sitk.Resample applies transform sample location mapping,
        # this is correct usage: Resample(moving, reference, transform)
        resampled = resample_mask_to_reference(avg_mask_img, fixed_img, transform)
        resampled_masks_per_structure[struct] = sitk.GetArrayFromImage(resampled).astype(np.uint8)

        out_path = os.path.join(output_dir, f"registered_mask_{struct}.nii")
        sitk.WriteImage(resampled, out_path)
        print(f"Wrote registered mask for {struct} -> {out_path}")

    # Compose a super volume in fixed image space that contains all registered subvolumes (logical OR)
    super_vol = np.zeros(fixed_np_shape, dtype=np.uint8)
    for struct, reg_mask_np in resampled_masks_per_structure.items():
        # reg_mask_np has shape (z,y,x) matching fixed image
        if reg_mask_np.shape != fixed_np_shape:
            raise RuntimeError(f"Registered mask for {struct} has shape {reg_mask_np.shape} but fixed image shape is {fixed_np_shape}")
        print(f'registered mask for {struct} shape={reg_mask_np.shape} ids={np.unique(reg_mask_np)} dtype={reg_mask_np.dtype}')
        reg_mask_np[reg_mask_np > 0] = 255  # ensure binary
        super_vol = np.maximum(super_vol, reg_mask_np)

    super_sitk = numpy_to_sitk(super_vol, reference_img=fixed_img, is_binary=True)
    super_path = os.path.join(output_dir, "super_volume_registered_structures.nii")
    sitk.WriteImage(super_sitk, super_path)
    print(f"Wrote super volume -> {super_path}")

    return {
        "avg_masks": avg_masks,
        "avg_coms_phys": avg_coms_phys,
        "fixed_coms_phys": fixed_coms_phys,
        "transforms": transforms_per_structure,
        "resampled_masks": resampled_masks_per_structure,
        "super_volume_path": super_path
    }




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=False)
    parser.add_argument('--debug', help='Enter debug True|False', required=False, default='true')
    parser.add_argument('--um', type=float, default=10.0, help='Resolution in microns (default: 10.0)')
    args = parser.parse_args()
    um = args.um
    
    args = parser.parse_args()
    animal = args.animal
    debug = bool({'true': True, 'false': False}[str(args.debug).lower()])
    if animal is None:
        animals = ['MD585', 'MD589', 'MD594']
    else:
        animals = [animal]

    """"
    for animal in animals:
        create_volumes_and_origins(animal, debug)
    """

    # Example placeholders - replace with your real file paths & structure list
    # fixed image (the target space)
    um = str(um)
    reg_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration"
    fixed_image_path = os.path.join(reg_path, f"Allen/Allen_{um}x{um}x{um}um_sagittal.nii")
    moving_image_paths = [
        os.path.join(reg_path, 'MD585', f'MD585_{um}x{um}x{um}um_sagittal.nii'),
        os.path.join(reg_path, 'MD589', f'MD589_{um}x{um}x{um}um_sagittal.nii'),
        os.path.join(reg_path, 'MD594', f'MD594_{um}x{um}x{um}um_sagittal.nii')
    ]
    atlas_path = "/net/birdstore/Active_Atlas_Data/data_root/atlas_data"
    json_paths_per_brain = [
        os.path.join(atlas_path, 'MD585', 'aligned_padded_structures.json'),
        os.path.join(atlas_path, 'MD589', 'aligned_padded_structures.json'),
        os.path.join(atlas_path, 'MD594', 'aligned_padded_structures.json')
    ]
    # structures to process
    structures = ["SC"]

    # mapping structure name -> fixed binary mask file path (these must be aligned to the fixed image)
    fixed_structure_masks = {
        "SC": "/net/birdstore/Active_Atlas_Data/data_root/atlas_data/Allen/structure/SC.nii",
    }

    out = register_subvolumes_to_fixed(
        fixed_image_path=fixed_image_path,
        moving_image_paths=moving_image_paths,
        json_paths_per_brain=json_paths_per_brain,
        structure_list=structures,
        fixed_structure_masks=fixed_structure_masks,
        output_dir="./registered_output",
        average_threshold=0.5
    )

