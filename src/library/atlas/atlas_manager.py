"""
This is the last script for creating the atlas
This will create a precomputed volume of the Active Brain Atlas which
you can import into neuroglancer
"""
import os
import sys
import json
import numpy as np


np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
from taskqueue import LocalTaskQueue
import igneous.task_creation as tc
from cloudvolume import CloudVolume
from pathlib import Path

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())


class AtlasToNeuroglancer():
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
        print(f'Creating downsampled volumes in {self.precomputed_vol.layer_cloudpath}')
        tasks = tc.create_downsampling_tasks(self.precomputed_vol.layer_cloudpath, compress=True, num_mips=1)
        tq.insert(tasks)
        tq.execute()

    def add_segmentation_mesh(self):
        if self.precomputed_vol is None:
            raise NotImplementedError('You have to call init_precomputed before calling this function.')

        tq = LocalTaskQueue(parallel=2)
        tasks = tc.create_meshing_tasks(self.precomputed_vol.layer_cloudpath, mip=0, compress=True, sharded=False) # The first phase of creating mesh
        tq.insert(tasks)
        tq.execute()

        # It should be able to incorporated to above tasks, but it will give a weird bug. Don't know the reason
        tasks = tc.create_mesh_manifest_tasks(self.precomputed_vol.layer_cloudpath) # The second phase of creating mesh
        tq.insert(tasks)
        tq.execute()
