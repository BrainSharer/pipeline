"""
William, this is the last script for creating the atlas

This will create a precomputed volume of the Active Brain Atlas which
you can import into neuroglancer
"""
import os
import numpy as np
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
import shutil
from cloudvolume import CloudVolume

from atlas_manager import Atlas
from library.utilities.neuroglancer_manager import NumpyToNeuroglancer
from library.utilities.assembler import AtlasAssembler
from brain_structure_manager import BrainStructureManager


class NgConverter(NumpyToNeuroglancer):
    def __init__(self, volume=None, scales=None, offset=[0, 0, 0], layer_type='segmentation'):
        if volume is not None:
            self.volume = np.pad(volume, [[1, 0], [1, 0], [1, 0]])
        self.scales = scales
        self.offset = offset
        self.layer_type = layer_type
        self.precomputed_vol = None

    def init_precomputed(self, path):
        info = CloudVolume.create_new_info(
            num_channels=self.volume.shape[3] if len(
                self.volume.shape) > 3 else 1,
            layer_type=self.layer_type,
            # Channel images might be 'uint8'
            data_type=str(self.volume.dtype),
            # raw, jpeg, compressed_segmentation, fpzip, kempressed
            encoding='raw',
            resolution=self.scales,            # Voxel scaling, units are in nanometers
            # x,y,z offset in voxels from the origin
            voxel_offset=self.offset - np.array([1, 1, 1]),
            chunk_size=[64, 64, 64],           # units are voxels
            # e.g. a cubic millimeter dataset
            volume_size=self.volume.shape[:3],
        )
        self.precomputed_vol = CloudVolume(
            f'file://{path}', mip=0, info=info, compress=True, progress=True)
        self.precomputed_vol.commit_info()
        self.precomputed_vol[:, :, :] = self.volume

    def create_neuroglancer_files(self, output_dir, segment_properties):
        self.reset_output_path(output_dir)
        self.init_precomputed(output_dir)
        self.add_segment_properties(segment_properties)
        self.add_downsampled_volumes()
        self.add_segmentation_mesh()

    def reset_output_path(self, output_dir):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)


class NgSegmentMaker(NgConverter):
    def __init__(self, debug=False, out_folder='atlas_test', *arg, **kwarg):
        NgConverter.__init__(self, *arg, **kwarg)
        self.OUTPUT_DIR = os.path.join('/home/httpd/html/data', self.atlas)
        self.debug = debug

    def get_atlas_resolution(self):
        self.fixed_brain = BrainStructureManager('MD589', sql=True)
        resolution = self.fixed_brain.get_resolution()
        SCALE = 32
        return int(resolution * SCALE * 1000)

    def get_animal_resolution(self):
        resolution = self.get_resolution()
        return int(resolution * self.DOWNSAMPLE_FACTOR * 1000)


class AtlasNgMaker(Atlas, NgSegmentMaker):
    def __init__(self, atlas_name, debug=False, out_folder='atlas_test', threshold=0.9, sigma=3.0, offset=[0, 0, 0]):
        Atlas.__init__(self, atlas_name)
        NgSegmentMaker.__init__(
            self, debug, out_folder=out_folder, offset=offset)
        self.assembler = AtlasAssembler(
            atlas_name, threshold=threshold, sigma=sigma)
        self.resolution = self.get_atlas_resolution()

    def create_atlas_neuroglancer(self):
        self.volume = self.assembler.combined_volume
        self.scales = [self.resolution, self.resolution, 20]
        segment_properties = self.get_segment_properties()
        self.create_neuroglancer_files(self.OUTPUT_DIR, segment_properties)

"""
class BrainNgMaker(BrainStructureManager, NgSegmentMaker):
    def __init__(self, animal, debug=False, out_folder='animal_folder', *args, **kwargs):
        BrainStructureManager.__init__(self, animal)
        NgSegmentMaker.__init__(
            self, debug, out_folder=out_folder, *args, **kwargs)
        self.assembler = BrainAssembler(animal)
        self.resolution = self.get_animal_resolution()

    def create_brain_neuroglancer(self):
        self.volume = self.assembler.combined_volume
        self.scales = [self.resolution, self.resolution, 20]
        segment_properties = self.get_segment_properties()
        self.create_neuroglancer_files(self.OUTPUT_DIR, segment_properties)
"""

if __name__ == '__main__':
    atlas = 'atlasV8'
    debug = False
    maker = AtlasNgMaker(atlas, debug, threshold=0.9,
                         out_folder='new_atlas', sigma=3.0)
    maker.assembler.assemble_all_structure_volume()
    maker.create_atlas_neuroglancer()
