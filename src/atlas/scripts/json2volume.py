
import json
import os
from pathlib import Path

import cv2
import numpy as np

from scipy.ndimage import center_of_mass
from skimage.draw import polygon
from tqdm import tqdm
import tifffile

XY_RES = 0.325 * 32
Z_RES = 20.0
ATLAS_SHAPE = (
    1000,
    1820,
    1140
)
ATLAS_RES = 10.0
DATA_DIR = "/net/birdstore/Active_Atlas_Data/data_root/atlas_data"


def polygons_to_um(polygons):

    result = {}

    for z, verts in polygons.items():

        xyz = np.zeros((len(verts), 3))

        xyz[:, 0] = verts[:, 0] * XY_RES
        xyz[:, 1] = verts[:, 1] * XY_RES
        xyz[:, 2] = z * Z_RES

        result[z] = xyz

    return result


def polygon_centroid(vertices):

    x = vertices[:, 0]
    y = vertices[:, 1]

    a = np.sum(
        x*np.roll(y,-1) -
        np.roll(x,-1)*y
    )

    a *= 0.5

    cx = np.sum(
        (x + np.roll(x,-1))
        *
        (x*np.roll(y,-1)
        -
        np.roll(x,-1)*y)
    )

    cy = np.sum(
        (y + np.roll(y,-1))
        *
        (x*np.roll(y,-1)
        -
        np.roll(x,-1)*y)
    )

    cx /= (6*a)
    cy /= (6*a)

    return np.array([cx, cy])

def structure_center_of_mass(polygons_um):

    centroids = []
    weights = []

    for z, verts in polygons_um.items():

        xy = verts[:, :2]

        centroid = polygon_centroid(xy)

        x = xy[:,0]
        y = xy[:,1]

        area = 0.5 * np.abs(
            np.sum(
                x*np.roll(y,-1)
                -
                np.roll(x,-1)*y
            )
        )

        centroids.append(
            [centroid[0],
             centroid[1],
             z * Z_RES]
        )

        weights.append(area)

    centroids = np.asarray(centroids)
    weights = np.asarray(weights)

    com = np.average(
        centroids,
        axis=0,
        weights=weights
    )

    return com

def estimate_affine(src, dst):

    if src.shape != dst.shape:
        print(f"Source and destination shapes do not match: {src.shape} vs {dst.shape}")
        exit(0)
    n = src.shape[0]
    X = np.hstack([src, np.ones((n,1))])
    T, _, _, _ = np.linalg.lstsq(X, dst, rcond=None)
    affine = np.eye(4)
    affine[:3,:4] = T.T

    return affine

def transform_vertices(vertices, affine):

    pts = np.hstack([vertices, np.ones((len(vertices),1))])
    transformed = (affine @ pts.T).T

    return transformed[:, :3]


def rasterize_structure(polygons_um, affine):

    mask = np.zeros(
        (1140, 1000, 1820),
        dtype=np.uint8
    )

    for z, verts in polygons_um.items():

        transformed = transform_vertices(
            verts,
            affine
        )

        atlas_xyz = transformed / ATLAS_RES

        z_vox = int(
            round(
                np.mean(atlas_xyz[:,2])
            )
        )
        plane = np.zeros((1000, 1820), dtype=np.uint8)
        #print(f'plane shape: {plane.shape} dtype: {plane.dtype} z_vox: {z_vox}')
        #print(f'shape of atlas_xyz: {atlas_xyz.shape} dtype: {atlas_xyz.dtype} z_vox: {z_vox}')
        pts = np.asarray(
                atlas_xyz[:, :2],
                dtype=np.int32
            )
        #print(pts)
        cv2.fillPoly(
                plane,
                [pts],
                255
            )


        mask[z_vox, :, :] = plane
        #ids, counts = np.unique(pts, return_counts=True) 
        #print(f"shape of pts: {pts.shape} dtype: {pts.dtype} unique ids in plane: {ids} counts: {counts}")

        """
        rr, cc = polygon(
            atlas_xyz[:,1],
            atlas_xyz[:,0],
            shape=ATLAS_SHAPE[1:]
        )

        mask[z_vox, rr, cc] = 1
        """

    return mask

def load_structure_json(json_file):

    with open(json_file) as f:
        data = json.load(f)

    slices = {}

    if isinstance(data, dict):

        for z, verts in data.items():
            slices[int(z)] = np.asarray(verts, dtype=np.float64)

    else:

        for item in data:
            for z, verts in item.items():
                slices[int(z)] = np.asarray(verts, dtype=np.float64)

    return slices

def process_brain(
        input_dir,
        output_dir,
        allen_coms):

    structure_coms = []
    structure_names = []
    polygons_all = []

    files = sorted(Path(input_dir).glob("*.json"))
    print(f"Found {len(files)} structure JSON files in {input_dir}")

    for f in tqdm(files):

        poly = load_structure_json(f)
        poly_um = polygons_to_um(poly)
        com = structure_center_of_mass(poly_um)
        structure_coms.append(com)
        structure_names.append(f.stem)

        polygons_all.append(poly_um)

    structure_coms = np.asarray(structure_coms)

    affine = estimate_affine(structure_coms, allen_coms)
    print(f'Estimated affine:\n{affine}')

    os.makedirs(output_dir,exist_ok=True)

    for name, poly_um, com in zip(
            structure_names,
            polygons_all,
            structure_coms):

        #if name not in ["SC", "IC", '7n_L', '7n_R']:
        #    continue
        mask = rasterize_structure(poly_um,affine)
        mask_path = os.path.join(output_dir, "mask")
        origin_path = os.path.join(output_dir, "origin")
        com_path = os.path.join(output_dir, "com")
        os.makedirs(mask_path, exist_ok=True)
        os.makedirs(origin_path, exist_ok=True)
        os.makedirs(com_path, exist_ok=True)

        mask_file_path = os.path.join(mask_path, f"{name}.npy")
        origin_file_path = os.path.join(origin_path, f"{name}.txt")
        com_file_path = os.path.join(com_path, f"{name}.txt")


        transformed_com = (
            affine
            @ np.append(com,1)
        )[:3]

        origin = np.array(
            np.where(mask)
        ).min(axis=1)

        #np.save(mask_file_path, mask)
        tifffile.imwrite(mask_file_path.replace(".npy", ".tif"), mask.astype(np.uint8), bigtiff=True, compression="LZW")
        np.savetxt(origin_file_path, origin)
        np.savetxt(com_file_path, transformed_com)
        print(f"Saved structure data to {Path(output_dir)}")

    return affine

def build_data(output_path):
    allen_path = os.path.join(DATA_DIR, "Allen", "com")
    os.makedirs(output_path, exist_ok=True)
    animals = ["MD585", "MD589", "MD594"]

    for animal in animals:

        jsonpath = os.path.join(
            DATA_DIR, animal, "aligned_padded_structures.json"
        )
        if not os.path.exists(jsonpath):
            print(f"{jsonpath} does not exist")
            exit(0)
        with open(jsonpath) as f:
            aligned_dict = json.load(f)

        brain_path = os.path.join(output_path, animal)
        os.makedirs(brain_path, exist_ok=True)

        structures = sorted(aligned_dict.keys())

        for structure in structures:
            json_data_list = []
            structure_path = os.path.join(brain_path, 'data', f'{structure}.json')
            if os.path.exists(structure_path):
                continue
            allen_com_path = os.path.join(allen_path, f"{structure}.txt")
            if not os.path.exists(allen_com_path):
                continue
            polygons = aligned_dict[structure]
            for k, v in polygons.items():
                json_data_list.append({k:v})

            with open(structure_path, "w", encoding="utf-8") as json_file:
                json.dump(json_data_list, json_file, indent=4)


def build_allen_coms():
    output_path = os.path.join(DATA_DIR, "Allen", "allen_structure_coms.npy")
    if os.path.exists(output_path):
        print(f"{output_path} already exists, skipping building Allen COMs")
        return
    com_path = os.path.join(DATA_DIR, "Allen", "com")
    coms = sorted(os.listdir(com_path))
    array_list = []
    for com in coms:
        print(f"Processing {com}")
        com_file = os.path.join(com_path, com)
        com_data = np.loadtxt(com_file)
        print(f"{com} COM: {com_data}")
        array_list.append(com_data)

    allen_coms = np.array(array_list)
    print(f"Allen COMs shape: {allen_coms.shape} dtype: {allen_coms.dtype}")
    np.save(output_path, allen_coms)

def create_probability_atlas(
        registered_dirs,
        output_dir):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    files = sorted(
        Path(registered_dirs[0])
        .glob("*.npy")
    )

    for f in tqdm(files, desc="Creating probability atlas"):

        if "_meta" in f.name:
            continue

        masks = []

        for d in registered_dirs:
            masks.append(np.load(Path(d) / f.name))

        masks = np.stack(masks)

        probability = masks.mean(axis=0)

        np.save(
            Path(output_dir)
            / f.name,
            probability.astype(
                np.float32
            )
        )

def main():
    animals = ["MD585", "MD589", "MD594"]
    allen_coms_path = os.path.join(DATA_DIR, "Allen", "allen_structure_coms.npy")
    allen_coms = np.load(allen_coms_path)
    print(f"Loading Allen COMs from {allen_coms_path} with shape {allen_coms.shape} and dtype {allen_coms.dtype}")

    registered_dirs = []

    for animal in animals:  # Process only the first animal
        input_animal_dir = os.path.join(DATA_DIR, animal, 'data')
        output_animal_dir = os.path.join(DATA_DIR, animal, 'registered')
        if not os.path.exists(input_animal_dir):
            print(f"{input_animal_dir} does not exist, skipping {animal}")
            exit(0)

        process_brain(
            input_animal_dir,
            output_animal_dir,
            allen_coms
        )
        print(f"Processed brain {input_animal_dir} and saved registered structures to {output_animal_dir}")

        registered_dirs.append(output_animal_dir)
        print(f"Registered directories so far: {registered_dirs}")

    probability_path = os.path.join(DATA_DIR, "probability_atlas")
    os.makedirs(probability_path, exist_ok=True)
    #create_probability_atlas(registered_dirs,probability_path)



if __name__ == "__main__":
    build_data(DATA_DIR)
    build_allen_coms()
    main()