"""
This script will take a source brain (where the data comes from) and an image brain 
(the brain whose images you want to display unstriped) and align the data from the point brain
to the image brain. It first aligns the point brain data to the atlas, then that data
to the image brain. It prints out the data by default and also will insert
into the database if given a layer name.
"""
import os
import numpy as np
from collections import defaultdict
from skimage.filters import gaussian
from scipy.ndimage import center_of_mass, zoom

import SimpleITK as sitk
from tqdm import tqdm

from library.atlas.atlas_utilities import adjust_volume, apply_affine_transform, average_images, compute_affine_transformation, list_coms, register_volume
from library.image_manipulation.filelocation_manager import data_path
from library.utilities.atlas import volume_to_polygon, save_mesh
from library.utilities.atlas import singular_structures


class BrainMerger():

    def __init__(self, animal):
        self.animal = animal
        self.symmetry_list = singular_structures
        self.coms_to_merge = defaultdict(list)
        self.origins_to_merge = defaultdict(list)
        self.volumes_to_merge = defaultdict(list)
        self.volumes = {}
        self.origins = {}
        self.data_path = os.path.join(data_path, 'atlas_data', self.animal)
        
        self.com_path = os.path.join(self.data_path, 'com')
        self.origin_path = os.path.join(self.data_path, 'origin')
        self.mesh_path = os.path.join(self.data_path, 'mesh')
        self.volume_path = os.path.join(self.data_path, 'structure')

        self.volumes = {}
        self.coms = {}
        self.origins = {}
        self.margin = 50
        self.threshold = 0.25  # the closer to zero, the bigger the structures
        # a value of 0.01 results in very big close fitting structures

        os.makedirs(self.com_path, exist_ok=True)
        os.makedirs(self.origin_path, exist_ok=True)
        os.makedirs(self.mesh_path, exist_ok=True)
        os.makedirs(self.volume_path, exist_ok=True)


    def pad_volume(self, size, volume):
        size_difference = size - volume.shape
        xr, yr, zr = ((size_difference)/2).astype(int)
        xl, yl, zl = size_difference - np.array([xr, yr, zr])
        return np.pad(volume, [[xl, xr], [yl, yr], [zl, zr]])

    def merge_volumes(self, structure, volumes):

        lvolumes = len(volumes)
        if lvolumes == 1:
            return volumes[0]
        elif lvolumes > 1:
            average_volume = average_images(volumes, structure)
            return average_volume
        else:
            print(f'{structure} has no volumes to merge')
            return None

    @staticmethod
    def get_mean_coordinates(xyz):
        return np.mean(xyz, axis=0)

    def save_brain_coms_meshes_origins_volumes(self):
        origins_mean = self.get_mean_coordinates(list(self.origins.values()))
        scales = (1.464, 1.464, 2)
        origins_mean = origins_mean * scales
        desc = f"Saving {self.animal} coms/meshes/origins/volumes"

        for structure, volume in tqdm(self.volumes.items(), desc=desc, disable=False):
            volume = np.swapaxes(volume, 0, 2)
            volume = zoom(volume, scales)
            com = center_of_mass(volume)
            origin = self.origins[structure] * scales
            com += origin

            com_filepath = os.path.join(self.com_path, f'{structure}.txt')
            origin_filepath = os.path.join(self.origin_path, f'{structure}.txt')
            volume_filepath = os.path.join(self.volume_path, f'{structure}.npy')

            np.savetxt(com_filepath, com)
            np.savetxt(origin_filepath, origin)
            np.save(volume_filepath, volume)

            #mesh STL file
            mesh_origin = origin - origins_mean
            aligned_structure = volume_to_polygon(volume=volume, origin=mesh_origin, times_to_simplify=3)
            mesh_filepath = os.path.join(self.mesh_path, f'{structure}.stl')
            save_mesh(aligned_structure, mesh_filepath)

    def save_atlas_coms_meshes_origins_volumes(self):
        origins = {structure: self.get_mean_coordinates(origin) for structure, origin in self.origins_to_merge.items()}
        origins_array = np.array(list(origins.values()))
        origins_mean = self.get_mean_coordinates(origins_array)
        desc = "Saving atlas coms/meshes/origins/volumes"
        for structure in tqdm(self.volumes.keys(), desc=desc):
            volume = self.volumes[structure]
            mesh_volume = adjust_volume(volume, 100)
            origin = origins[structure] - origins_mean
            
            origin_filepath = os.path.join(self.origin_path, f'{structure}.txt')
            volume_filepath = os.path.join(self.volume_path, f'{structure}.npy')

            np.savetxt(origin_filepath, origin)
            np.save(volume_filepath, volume)

            #mesh
            aligned_structure = volume_to_polygon(volume=mesh_volume, origin=origin, times_to_simplify=3)
            mesh_filepath = os.path.join(self.mesh_path, f'{structure}.stl')
            save_mesh(aligned_structure, mesh_filepath)

            
    def fetch_allen_origins(self):
        structures = {
            '3N_L': (354.00, 147.00, 216.00),
            '3N_R': (354.00, 147.00, 444.00),
            '4N_L': (381.00, 147.00, 214.00),
            '4N_R': (381.00, 147.00, 442.00),
            '5N_L': (393.00, 195.00, 153.00),
            '5N_R': (393.00, 195.00, 381.00),
            '6N_L': (425.00, 204.00, 204.00),
            '6N_R': (425.00, 204.00, 432.00),
            '7N_L': (415.00, 256.00, 153.00),
            '7N_R': (415.00, 256.00, 381.00),
            '7n_L': (407.00, 199.00, 157.00),
            '7n_R': (407.00, 199.00, 385.00),
            'AP': (495.00, 193.00, 217.00),
            'Amb_L': (454.00, 258.00, 167.00),
            'Amb_R': (454.00, 258.00, 395.00),
            'DC_L': (424.00, 177.00, 114.00),
            'DC_R': (424.00, 177.00, 342.00),
            'IC': (369.00, 44.00, 141.00),
            'LC_L': (424.00, 161.00, 185.00),
            'LC_R': (424.00, 161.00, 413.00),
            'LRt_L': (464.00, 262.00, 150.00),
            'LRt_R': (464.00, 262.00, 378.00),
            'PBG_L': (365.00, 141.00, 138.00),
            'PBG_R': (365.00, 141.00, 366.00),
            'Pn_L': (342.00, 139.00, 119.00),
            'Pn_R': (342.00, 139.00, 347.00),
            'RtTg': (353.00, 185.00, 161.00),
            'SC': (329.00, 41.00, 161.00),
            'SNC_L': (313.00, 182.00, 148.00),
            'SNC_R': (313.00, 182.00, 376.00),
            'SNR_L': (310.00, 175.00, 137.00),
            'SNR_R': (310.00, 175.00, 365.00),
            'Sp5C_L': (495.00, 202.00, 136.00),
            'Sp5I_L': (465.00, 202.00, 127.00),
            'Sp5I_R': (465.00, 202.00, 355.00),
            'Sp5O_L': (426.00, 207.00, 137.00),
            'Sp5O_R': (426.00, 207.00, 365.00),
            'VLL_L': (361.00, 149.00, 137.00),
            'VLL_R': (361.00, 149.00, 365.00),
        }
        return structures
    
    @staticmethod
    def calculate_distance(self, com1, com2):
        return (np.linalg.norm(com1 - com2))
