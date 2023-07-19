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
from scipy.ndimage import center_of_mass

from library.image_manipulation.filelocation_manager import data_path
from library.utilities.algorithm import brain_to_atlas_transform, umeyama
from library.utilities.atlas import volume_to_polygon, save_mesh
from library.utilities.atlas import singular_structures
from library.registration.brain_structure_manager import BrainStructureManager


class BrainMerger():

    def __init__(self):
        self.symmetry_list = singular_structures
        self.volumes_to_merge = defaultdict(list)
        self.origins_to_merge = defaultdict(list)
        atlas = 'atlasV8'
        self.data_path = os.path.join(data_path, 'atlas_data', atlas)
        self.volume_path = os.path.join(self.data_path, 'structure')
        self.origin_path = os.path.join(self.data_path, 'origin')
        self.mesh_path = os.path.join(self.data_path, 'mesh')
        self.volumes = {}
        self.margin = 50
        self.threshold = 0.25  # the closer to zero, the bigger the structures
        # a value of 0.01 results in very big close fitting structures

    def pad_volume(self, size, volume):
        size_difference = size - volume.shape
        xr, yr, zr = ((size_difference)/2).astype(int)
        xl, yl, zl = size_difference - np.array([xr, yr, zr])
        return np.pad(volume, [[xl, xr], [yl, yr], [zl, zr]])

    def get_merged_landmark_probability(self, structure, volumes):
        lvolumes = len(volumes)
        if lvolumes == 1:
            print(f'{structure} has only one volume')
            return volumes[0]
        elif lvolumes > 1:
            sizes = np.array([vi.shape for vi in volumes])
            volume_size = sizes.max(0) + self.margin
            volumes = [self.pad_volume(volume_size, vi) for vi in volumes]
            volumes = list([(v > 0).astype(np.int32) for v in volumes])

            merged_volume = np.sum(volumes, axis=0)
            merged_volume_prob = merged_volume / float(np.max(merged_volume))
            # increasing the STD makes the volume smoother
            # Smooth the probability
            average_volume = gaussian(merged_volume_prob, 3.0)
            color = 1
            average_volume[average_volume > self.threshold] = color
            average_volume[average_volume != color] = 0
            average_volume = average_volume.astype(np.uint8)
            return average_volume
        else:
            print(f'{structure} has no volumes to merge')
            return None

    def save_atlas_origins_and_volumes_and_meshes(self):
        """Saves everything to disk
        """
        os.makedirs(self.origin_path, exist_ok=True)
        os.makedirs(self.volume_path, exist_ok=True)
        os.makedirs(self.mesh_path, exist_ok=True)

        average_origins = {structure: np.mean(origin, axis=0) for structure, origin in self.origins_to_merge.items()}
        origins = self.transform_origins(average_origins)


        for structure in self.volumes.keys():
            x, y, z = origins[structure]
            volume = self.volumes[structure]
            com = center_of_mass(volume)
            origin_array = np.array(list(origins.values())) - com
            centered_origin = (x, y, z) - origin_array.mean(0)
            aligned_structure = volume_to_polygon(
                volume=volume, origin=centered_origin, times_to_simplify=3)
            origin_filepath = os.path.join(
                self.origin_path, f'{structure}.txt')
            volume_filepath = os.path.join(
                self.volume_path, f'{structure}.npy')
            mesh_filepath = os.path.join(self.mesh_path, f'{structure}.stl')
            if 'SC' in structure:
                print(origin_filepath)
                print(volume_filepath)
                print(mesh_filepath)
            np.savetxt(origin_filepath, (x, y, z))
            np.save(volume_filepath, volume)
            save_mesh(aligned_structure, mesh_filepath)

    def calculate_distance(self, com1, com2):
        return (np.linalg.norm(com1 - com2))

    def transform_origins(self, moving_origins):
        """moving origins VS fixed COMs
        fixed_coms have to be adjusted to subtract half the volume size
        so they become origins
        """
        fixed_brain = BrainStructureManager('Allen')
        fixed_coms = fixed_brain.get_coms(annotator_id=1)

        common_keys = fixed_coms.keys() & moving_origins.keys()
        brain_regions = sorted(moving_origins.keys())

        #fixed_points_list = []
        fixed_volume_list = []
        fixed_volume_dict = {}
        for structure in brain_regions:
            if structure in common_keys:
                #fixed_points_list.append(fixed_coms[structure])
                volume_com = center_of_mass(self.volumes[structure])
                fixed_volume_list.append(volume_com)
                fixed_volume_dict[structure] = volume_com

        fixed_volume_arr = np.array(fixed_volume_list)
        
        fixed_points = np.array([fixed_coms[s] for s in brain_regions if s in common_keys]) / 25 - fixed_volume_arr
        moving_points = np.array([moving_origins[s] for s in brain_regions if s in common_keys])
        fixed_point_dict = {s:fixed_coms[s] for s in brain_regions if s in common_keys}
        moving_point_dict = {s:moving_origins[s] for s in brain_regions if s in common_keys}

        assert fixed_points.shape == moving_points.shape, 'Shapes do not match'
        assert len(fixed_points.shape) == 2, f'Dimensions are wrong fixed shape={fixed_points.shape} commonkeys={common_keys}'
        assert fixed_points.shape[0] > 2, 'Not enough points'

        transformed_coms = {}
        R, t = umeyama(moving_points.T, fixed_points.T)
        for structure, origin in moving_origins.items():
            point = brain_to_atlas_transform(origin, R, t)
            transformed_coms[structure] = point


        distances = []
        for structure in common_keys:
            (x,y,z) = fixed_point_dict[structure]
            x = (x/25 - fixed_volume_dict[structure][0])
            y = (x/25 - fixed_volume_dict[structure][1])
            z = (x/25 - fixed_volume_dict[structure][2])
            fixed_point = np.array([x,y,z])    
            moving_point = np.array(moving_point_dict[structure])
            reg_point = brain_to_atlas_transform(moving_point, R, t)
            d = self.calculate_distance(fixed_point, reg_point)
            distances.append(d)
            print(f'{structure} distance={round(d,2)}')
        
        print(f'length={len(distances)} mean={round(np.mean(distances))} min={round(min(distances))} max={round(max(distances))}')

        return transformed_coms
