#!/usr/bin/env python3
"""
allen_extract_to_nifti.py

Downloads an Allen CCF annotation volume (NRRD), downloads the structure graph (JSON),
creates a mask for a named structure (optionally including all descendants), and writes
a NIfTI file with the correct spacing, origin and direction.

Requires:
  pip install requests simpleitk numpy tqdm

Example:
  python allen_extract_to_nifti.py --structure "Hippocampal formation" --resolution 10 --out hippocampus_10um.nii.gz
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import math
from tqdm import tqdm
from scipy.ndimage import center_of_mass

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.core.reference_space_cache import ReferenceSpaceCache


PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.atlas.atlas_manager import AtlasToNeuroglancer
from library.utilities.atlas import allen_structures
from library.utilities.utilities_process import read_image, write_image

def get_center_of_mass(um, debug):
    # --- step 0 clean up
    data_path = "/net/birdstore/Active_Atlas_Data/data_root/atlas_data/Allen"
    com_path = os.path.join(data_path, 'com')
    structure_path = os.path.join(data_path, 'structure')
    if os.path.exists(com_path):
        shutil.rmtree(com_path)
    os.makedirs(com_path, exist_ok=True)
    if os.path.exists(structure_path):
        shutil.rmtree(structure_path)
    os.makedirs(structure_path, exist_ok=True)
    # --- Step 1: Initialize Allen Reference Space ---
    # Downloaded annotation volume and structure tree will be cached here
    resolution = um  # microns
    rspc = ReferenceSpaceCache(
        manifest='mouse_ccf_manifest.json',
        resolution=resolution,
        reference_space_key='annotation/ccf_2017'
    )

    # --- Step 2: Load annotation volume and structure tree ---
    allen_volume, meta = rspc.get_annotation_volume()
    print("Annotation volume shape (x,y,z):", allen_volume.shape)
    registered_full = np.zeros_like(allen_volume, dtype=np.uint32)
    structure_tree = rspc.get_structure_tree()

    # --- Step 3: Choose a structure ID ---
    # Example: Primary visual area (VISp) = 385
    for abbreviation, structure_id in allen_structures.items():
        if debug:
            if abbreviation not in ['SC']:  # just a few test structures
                continue
        if isinstance(structure_id, list):
            structure_id = structure_id[0]
        print(f"\nProcessing structure: {abbreviation} (ID: {structure_id})")
        # --- Step 4: Get structure and its substructures ---
        structure = structure_tree.get_structures_by_id([structure_id])[0]
        descendant_ids = structure_tree.descendant_ids([structure_id])[0]
        structure_ids = [structure_id] + descendant_ids  # include subregions
        print(f"Including {len(structure_ids)} structure IDs: {structure_ids}")
        print(f'descendants: {[structure_tree.get_structures_by_id([sid])[0]["name"] for sid in descendant_ids]}')
        print('descendant ids, ', descendant_ids)

        # --- Step 5: Create binary mask ---

        mask = np.isin(allen_volume, structure_ids)
        registered_full = np.maximum(registered_full, mask)
        midpoint = allen_volume.shape[2] // 2
        if abbreviation.endswith('L'):
            print(f'Left side {abbreviation}')
            mask = mask[:,:,0:midpoint]
        elif abbreviation.endswith('R'):
            print(f'Right side  {abbreviation}')
            mask = mask[:,:,midpoint:]
        else:
            print(f'Singular object {abbreviation}')

        # --- Step 6: Compute center of mass (in voxel coordinates) ---
        center_vox = np.array(center_of_mass(mask))

        # --- Step 7: Convert to microns in CCF coordinates ---
        # CCF space origin is at the corner, each voxel = resolution microns
        center_um = center_vox * resolution

        print(f"\tStructure: {structure['name']} ({structure['acronym']})")
        print(f"\tCenter of mask (voxel indices): {center_vox}")
        print(f"\tCenter of mask (microns): {center_um}")
        comfile_path = os.path.join(com_path, f'{abbreviation}.txt')
        np.savetxt(comfile_path, center_um)
        print(f"\tWrote COM file to {comfile_path}")

    # Save the registered full mask
    registered_full_path = os.path.join(structure_path, 'registered_full.tif')
    write_image(registered_full_path, registered_full.astype(np.uint32))
    print(f"Wrote registered full mask to {registered_full_path}")


def load_allen_nrrd(um):
    nrrd_path = f"/home/eddyod/programming/pipeline/mouse_connectivity/annotation/ccf_2017/annotation_{um}.nrrd"
    if not os.path.exists(nrrd_path):
        print(f"NRRD file not found: {nrrd_path}")
        sys.exit(1)

    # read NRRD with SimpleITK (preserves spacing, origin, direction)
    print("Reading atlas NRRD with SimpleITK (this preserves spacing/origin/direction)...")
    atlas_img = sitk.ReadImage(str(nrrd_path))
    atlas_arr = sitk.GetArrayFromImage(atlas_img).astype(np.int64)  # shape: [z,y,x]
    print("Atlas shape (z,y,x):", atlas_arr.shape)
    print("Atlas spacing:", atlas_img.GetSpacing())
    print("Atlas origin:", atlas_img.GetOrigin())
    print("Atlas direction:", atlas_img.GetDirection())
    return atlas_img.GetSpacing(), atlas_img.GetOrigin(), atlas_img.GetDirection()


def parse_masks(um, debug):
    """
    mcc = MouseConnectivityCache(resolution=um)
    rsp = mcc.get_reference_space()
    allen_volume = rsp.annotation
    """

    rspc = ReferenceSpaceCache(
        manifest='mouse_ccf_manifest.json',
        resolution=um,
        reference_space_key='annotation/ccf_2017'
    )

    # --- Step 2: Load annotation volume and structure tree ---
    allen_volume, meta = rspc.get_annotation_volume()
    structure_tree = rspc.get_structure_tree()

    print('Shape of allen_volume', allen_volume.shape)
    registered_full = np.zeros_like(allen_volume, dtype=np.uint32)
    midpoint = int(allen_volume.shape[2] / 2)
    print('Mid z', midpoint)
    # Pn looks like one mass in Allen

    data_path = "/net/birdstore/Active_Atlas_Data/data_root/atlas_data/Allen"
    com_path = os.path.join(data_path, 'com')
    structure_path = os.path.join(data_path, 'structure')
    if os.path.exists(com_path):
        shutil.rmtree(com_path)
    os.makedirs(com_path, exist_ok=True)
    if os.path.exists(structure_path):
        shutil.rmtree(structure_path)
    os.makedirs(structure_path, exist_ok=True)

    ids = {}

    for abbreviation, structure_id in allen_structures.items():
        if debug:
            if abbreviation not in ['SC', 'VLL_L', 'VLL_R']:  # just a few test structures
                continue

        bad_keys = ['RMC_L', 'RMC_R']

        if type(structure_id) == list:
            sid = structure_id
        else:
            sid = [structure_id]
        if sid[0] > 1000:
            continue

        all_ids = []
        for s in sid:
            structure = structure_tree.get_structures_by_id([s])[0]
            descendant_ids = structure_tree.descendant_ids([s])[0]
            structure_ids = [s] + descendant_ids  # include subregions
            all_ids.extend(structure_ids)

        allen_id = sid[0]
        structure_mask = np.isin(allen_volume, all_ids)

        #structure_mask = rsp.make_structure_mask(sid, direct_only=True)
        #structure_mask = np.swapaxes(structure_mask,0,2)
        structure_mask = structure_mask.astype(np.uint32)
        structure_mask[structure_mask > 0] = allen_id
        registered_full = np.maximum(registered_full, structure_mask)


        if abbreviation.endswith('L'):
            print(f'Left side {abbreviation}')
            structure_mask = structure_mask[:,:,0:midpoint]
        elif abbreviation.endswith('R'):
            print(f'Right side  {abbreviation}')
            structure_mask = structure_mask[:,:,midpoint:]
        else:
            print(f'Singular object {abbreviation}')

        try:
            x_coords, y_coords, z_coords = np.where(structure_mask == allen_id)
        except ValueError as e:
            print(f"\tNo voxels found for structure {abbreviation} with IDs {allen_id}, skipping...")
            continue
        try:
            min_x = np.min(x_coords)
            max_x = np.max(x_coords)
            min_y = np.min(y_coords)
            max_y = np.max(y_coords)
            min_z = np.min(z_coords)
            max_z = np.max(z_coords)
        except ValueError as e:
            print(f"\tNo voxels found for structure {abbreviation} with IDs {allen_id}, skipping...")
            continue
        print(f'\tbbox (x,y,z): {min_x}:{max_x}, {min_y}:{max_y}, {min_z}:{max_z}')

        x,y,z = center_of_mass(structure_mask)

        if math.isnan(x) or math.isnan(y) or math.isnan(z):
            print(f"\tNo voxels found for structure {abbreviation} with IDs {sid}, skipping...")
            continue
        ids[abbreviation] = allen_id

        x *= um
        y *= um
        if abbreviation.endswith('R'):
            z = (z + midpoint) * um
        else:
            z *= um
        center_um = np.array([x, y, z])
    
        print(f"\tStructure: {structure['name']} ({structure['acronym']})")
        print(f"\tCenter of mass (x,y,z): {x}, {y}, {z}")
        print(f'\tIDs {sid} mask dtype={structure_mask.dtype} shape={structure_mask.shape} unique values={np.unique(structure_mask)}')

        comfile_path = os.path.join(com_path, f'{abbreviation}.txt')
        np.savetxt(comfile_path, center_um)
        print(f"\tWrote COM file to {comfile_path}")
    # Save the registered full mask
    registered_full_path = os.path.join(structure_path, 'registered_full.tif')
    write_image(registered_full_path, registered_full.astype(np.uint32))
    print(f"Wrote registered full mask to {registered_full_path}")
    json.dump(ids, open(os.path.join(structure_path, 'ids.json'), 'w'))
    return ids, registered_full


def create_neuroglancer(um, ids=None, atlas_volume=None):
    data_path = "/net/birdstore/Active_Atlas_Data/data_root/atlas_data/Allen"
    structure_path = os.path.join(data_path, 'structure')
    registered_full_path = os.path.join(structure_path, 'registered_full.tif')
    if atlas_volume is None:
        atlas_volume = read_image(registered_full_path)
    print(f"Pre Shape {atlas_volume.shape=} {atlas_volume.dtype=}")
    if ids is None:
        ids = json.load(open(os.path.join(structure_path, 'ids.json'), 'r'))

    atlas_name = "Allen_DK_structures_10.0um"
    structure_path = os.path.join("/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/structures", atlas_name)
    if os.path.exists(structure_path):
        print(f"Removing existing neuroglancer data at {structure_path}")
        shutil.rmtree(structure_path)
    os.makedirs(structure_path, exist_ok=True)
    atlas_box_scales = np.array((um*1000, um*1000, um*1000))
    neuroglancer = AtlasToNeuroglancer(volume=atlas_volume, scales=atlas_box_scales)
    neuroglancer.init_precomputed(path=structure_path)
    neuroglancer.add_segment_properties(ids)
    neuroglancer.add_downsampled_volumes()
    neuroglancer.add_segmentation_mesh()
    print(f"Neuroglancer data created at {structure_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Allen CCF structure to NIfTI (preserving spacing/direction/origin).")
    parser.add_argument('--um', type=float, default=10.0, help='Resolution in microns (default: 10)')
    parser.add_argument('--debug', required=False, default='false', type=str)

    args = parser.parse_args()
    um = args.um
    debug = bool({'true': True, 'false': False}[args.debug.lower()])    
    #load_allen_nrrd()
    #get_center_of_mass(um, debug) # right is wrong, left looks good and singular structures look good.
    ids, atlas_volume = parse_masks(um, debug) # Both VLL_L and VLL_R look good.
    create_neuroglancer(um, ids, atlas_volume)