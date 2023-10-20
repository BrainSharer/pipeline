"""
This is the last script for creating the atlas
This will create a precomputed volume of the Active Brain Atlas which
you can import into neuroglancer
"""
import argparse
import os
import sys
import json
import numpy as np


np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
import shutil
from taskqueue import LocalTaskQueue
import igneous.task_creation as tc
from cloudvolume import CloudVolume
from pathlib import Path
from skimage import io
from scipy.ndimage import center_of_mass

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.controller.sql_controller import SqlController
from library.controller.structure_com_controller import StructureCOMController
from library.registration.algorithm import brain_to_atlas_transform, umeyama
from library.utilities.atlas import allen_structures
from library.utilities.utilities_process import SCALING_FACTOR

class NumpyToNeuroglancer():
    viewer = None

    def __init__(self, volume, scales, offset=[0, 0, 0], layer_type='segmentation'):
        self.volume = volume
        self.scales = scales
        self.offset = offset
        self.layer_type = layer_type
        self.precomputed_vol = None

    def init_precomputed(self, path):
        info = CloudVolume.create_new_info(
            num_channels = self.volume.shape[3] if len(self.volume.shape) > 3 else 1,
            layer_type = self.layer_type,
            data_type = str(self.volume.dtype),  # Channel images might be 'uint8'
            encoding = 'raw',                    # raw, jpeg, compressed_segmentation, fpzip, kempressed
            resolution = self.scales,            # Voxel scaling, units are in nanometers
            voxel_offset = self.offset,          # x,y,z offset in voxels from the origin
            chunk_size = [64,64,64],           # units are voxels
            volume_size = self.volume.shape[:3], # e.g. a cubic millimeter dataset
        )
        self.precomputed_vol = CloudVolume(f'file://{path}', mip=0, info=info, compress=True, progress=True)
        self.precomputed_vol.commit_info()
        self.precomputed_vol[:, :, :] = self.volume[:, :, :]

    def add_segment_properties(self, ids):

        if self.precomputed_vol is None:
            raise NotImplementedError('You have to call init_precomputed before calling this function.')
        self.precomputed_vol.info['segment_properties'] = 'names'
        self.precomputed_vol.commit_info()
        segment_properties_path = os.path.join(self.precomputed_vol.layer_cloudpath.replace('file://', ''), 'names')
        os.makedirs(segment_properties_path, exist_ok=True)
        info = {
            "@type": "neuroglancer_segment_properties",
            "inline": {
                "ids": [str(label) for label in ids.values()],
                "properties": [{
                    "id": "label",
                    "type": "label",
                    "values": [str(id) for id in ids.keys()]
                }]
            }
        }
        with open(os.path.join(segment_properties_path, 'info'), 'w') as file:
            json.dump(info, file, indent=2)

    def add_downsampled_volumes(self):
        if self.precomputed_vol is None:
            raise NotImplementedError('You have to call init_precomputed before calling this function.')
        tq = LocalTaskQueue(parallel=2)
        tasks = tc.create_downsampling_tasks(self.precomputed_vol.layer_cloudpath, compress=True)
        tq.insert(tasks)
        tq.execute()

    def add_segmentation_mesh(self):
        if self.precomputed_vol is None:
            raise NotImplementedError('You have to call init_precomputed before calling this function.')

        tq = LocalTaskQueue(parallel=2)
        tasks = tc.create_meshing_tasks(self.precomputed_vol.layer_cloudpath, mip=0, compress=True) # The first phase of creating mesh
        tq.insert(tasks)
        tq.execute()

        # It should be able to incoporated to above tasks, but it will give a weird bug. Don't know the reason
        tasks = tc.create_mesh_manifest_tasks(self.precomputed_vol.layer_cloudpath) # The second phase of creating mesh
        tq.insert(tasks)
        tq.execute()

class AtlasCreator:
    def __init__(self, animal='Atlas', debug=False):
        self.animal = animal
        self.debug = debug
        self.DATA_PATH = '/net/birdstore/Active_Atlas_Data/data_root'
        self.fixed_brain = 'Allen'
        self.ATLAS_PATH = os.path.join(self.DATA_PATH, 'atlas_data', animal)
        self.REG_PATH = os.path.join(self.DATA_PATH, 'brains_info/registration')
        self.OUTPUT_DIR = f'/var/www/brainsharer/structures/{animal}'
        if os.path.exists(self.OUTPUT_DIR):
            shutil.rmtree(self.OUTPUT_DIR)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.sqlController = SqlController(self.fixed_brain)

    def get_transform_to_align_brain(self):
        if 'Atlas' in self.animal:
            fixed = 'Allen'
            midbrain_keys = {'SC', 'SNC_L', 'SNC_R', '7N_L', '7N_R', 'SpV_L', 'SpV_R'}
            testing_structures = {'PBG_R', '3N_L', 'PBG_L', '4N_L', '4N_R', '3N_R'}
            structureController = StructureCOMController(fixed)
            fixed_coms = structureController.get_COM('Allen', annotator_id=1)
            moving_coms = structureController.get_COM('Atlas', annotator_id=1)
            common_keys = sorted(fixed_coms.keys() & moving_coms.keys() & testing_structures)
            fixed_points = np.array([fixed_coms[s] for s in common_keys])
            moving_points = np.array([moving_coms[s] for s in common_keys])
            if fixed_points.shape != moving_points.shape or len(fixed_points.shape) != 2 or fixed_points.shape[0] < 3:
                print(f'Error calculating transform {self.animal} {fixed_points.shape} {moving_points.shape} {common_keys}')
                print(f'Length fixed coms={len(fixed_coms.keys())} # moving coms={len(moving_coms.keys())}')
                sys.exit()

            # Divide by the Allen um
            fixed_points /= 25
            moving_points /= 25
            self.R, self.t = umeyama(moving_points.T, fixed_points.T)
        else:
            self.R = np.eye(3)
            self.t = np.zeros((3,1))
        


    def get_allen_id(self, color, structure):
        try:
            allen_color = allen_structures[structure]
        except KeyError:
            allen_color = color

        if type(allen_color) == list:
            allen_color = allen_color[0]
        
        return allen_color
    
    def create_atlas(self, save, ng):
        # origin is in animal scan_run.resolution coordinates
        # volume is in 10um
        color = 1000
        self.get_transform_to_align_brain()
        if 'Atlas' in self.animal:
            target_path = os.path.join(self.REG_PATH, 'Allen_25um_sagittal.tif')
            target_img = io.imread(target_path)
            height = target_img.shape[0]
            width = target_img.shape[1] + 150
            z_length = target_img.shape[2]
            xy_resolution = 25 * 1000 # Allen isotropic
            z_resolution = xy_resolution
        else:
            dir_path = os.path.join(self.DATA_PATH, 'pipeline_data', self.animal, 'preps/CH1/thumbnail_aligned')
            z_length = len(os.listdir(dir_path))
            self.sqlController = SqlController(self.animal)
            height = int(round(self.sqlController.scan_run.height / SCALING_FACTOR))
            width = int(round(self.sqlController.scan_run.width / SCALING_FACTOR))
            xy_resolution = self.sqlController.scan_run.resolution * SCALING_FACTOR * 1000
            z_resolution = self.sqlController.scan_run.zresolution * 1000


        
        atlas_box_size=(width, height, z_length)
        print(f'box size={atlas_box_size} xy_resolution={xy_resolution} z_resolution={z_resolution}')
        atlas_volume = np.zeros(atlas_box_size, dtype=np.uint32)
        origin_dir = os.path.join(self.ATLAS_PATH, 'origin')
        volume_dir = os.path.join(self.ATLAS_PATH, 'structure')
        if not os.path.exists(origin_dir):
            print(f'{origin_dir} does not exist, exiting.')
            sys.exit()
        if not os.path.exists(volume_dir):
            print(f'{volume_dir} does not exist, exiting.')
            sys.exit()
        atlas_box_scales = np.array([xy_resolution, xy_resolution, z_resolution])
        print(f'atlas box size={atlas_box_size} shape={atlas_volume.shape}')
        print(f'origin dir {origin_dir}')
        print(f'origin dir {volume_dir}')
        print(f'Using data from {self.ATLAS_PATH}')
        origins = sorted(os.listdir(origin_dir))
        volumes = sorted(os.listdir(volume_dir))
        print(f'Working with {len(origins)} origins and {len(volumes)} volumes.')
        ids = {}
        
        for origin_file, volume_file in zip(origins, volumes):
            if Path(origin_file).stem != Path(volume_file).stem:
                print(f'{Path(origin_file).stem} and {Path(volume_file).stem} do not match')
            structure = Path(origin_file).stem
            allen_color = self.get_allen_id(color, structure)
            color += 2

            #if structure != 'SC':
            #    continue

            origin = np.loadtxt(os.path.join(origin_dir, origin_file))
            x,y,z = brain_to_atlas_transform(origin, self.R, self.t)

            volume = np.load(os.path.join(volume_dir, volume_file))
            volume = volume.astype(np.uint32)
            volume[volume > 0] = allen_color
            xs,ys,zs = np.where(volume != 0)
            preshape = volume.shape
            try:
                volume = volume[min(xs):max(xs)+1,min(ys):max(ys)+1,min(zs):max(zs)+1] 
            except:
                pass
            ids[structure] = allen_color
            row_start = int(round(x))
            col_start = int(round(y))
            z_start = int(round(z))
            row_end = row_start + volume.shape[0]
            col_end = col_start + volume.shape[1]

            if 'Atlas' not in self.animal:
                z_indices = [z for z in range(volume.shape[2]) if z % 2 == 0]
                volume = volume[:, :, z_indices]
            z_end = z_start + volume.shape[2]
            if debug:
                volume_ids, counts = np.unique(volume, return_counts=True)
                print(f'{structure} preshape={preshape} postshape={volume.shape}  \
                    x: {row_start}->{row_end}, y: {col_start}->{col_end}, z: {z_start}->{z_end}', end=" ")
                print(f'color={allen_color} ids={volume_ids} counts={counts}')
            #continue
            try:
                atlas_volume[row_start:row_end, col_start:col_end, z_start:z_end] += volume
            except ValueError as ve:
                print(f'Error adding {structure} to atlas: {ve}')
        
        print(f'Shape of atlas volume {atlas_volume.shape} dtype={atlas_volume.dtype}')
        
        save_volume = np.swapaxes(atlas_volume, 0, 2)
        if self.debug:
            atlas_volume_ids, counts = np.unique(atlas_volume, return_counts=True)
            print('ids')
            print(atlas_volume_ids)
            print('counts')
            print(counts)
        #atlas_volume = np.rot90(atlas_volume, axes=(0, 1))
        #print(f'Shape of atlas volume {atlas_volume.shape} after swapping 0 and 2')
        if save:
            #save_volume = np.swapaxes(save_volume, 1, 2)
            outpath = f'/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/DKAtlas_25um_sagittal.tif'
            io.imsave(outpath, save_volume)
        if ng:
            neuroglancer = NumpyToNeuroglancer(atlas_volume, atlas_box_scales, offset=[0,0,0])
            neuroglancer.init_precomputed(self.OUTPUT_DIR)
            neuroglancer.add_segment_properties(ids)
            neuroglancer.add_downsampled_volumes()
            neuroglancer.add_segmentation_mesh()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Atlas')
    parser.add_argument('--animal', required=False, default='Atlas')
    parser.add_argument('--debug', required=False, default='true', type=str)
    parser.add_argument('--save', required=False, default='false', type=str)
    parser.add_argument('--ng', required=False, default='false', type=str)
    args = parser.parse_args()
    debug = bool({'true': True, 'false': False}[args.debug.lower()])    
    save = bool({'true': True, 'false': False}[args.save.lower()])    
    ng = bool({'true': True, 'false': False}[args.ng.lower()])    
    animal = args.animal
    atlasCreator = AtlasCreator(animal, debug)
    atlasCreator.create_atlas(save, ng)
