import argparse
import sys
from pathlib import Path

from tqdm import tqdm

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.atlas.atlas_utilities import NEW_ATLAS, ORIGINAL_ATLAS
from library.atlas.brain_structure_manager import BrainStructureManager
from library.atlas.brain_merger import BrainMerger


class AtlasManager():

    def __init__(self, animal, um=10, affine=False, debug=False):

        self.animal = animal
        self.brainManager = BrainStructureManager(animal, um, affine, debug)
        self.atlasMerger = BrainMerger(animal)
        self.debug = debug
        self.um = um
        self.foundation_brains = ['MD589', 'MD594', 'MD585']

    # 1st step, this parses the original CSV files and creates the JSON files
    def create_brain_json(self):
        for animal in self.foundation_brains:
            brainManager = BrainStructureManager(animal, self.um, self.debug)
            brainManager.create_brain_json(animal, self.debug)

    # 2nd step, this takes the JSON files and creates the brain volumes and origins
    def create_brain_volumes_and_origins(self):
        for animal in self.foundation_brains:
            brainMerger = BrainMerger(animal)
            self.brainManager.create_brain_volumes_and_origins(brainMerger, animal, self.debug)
            brainMerger.save_brain_origins_and_volumes_and_meshes()

    # optional step, this draws the brains from the cleaned images so you can check the placement of the volumes
    def test_brain_volumes_and_origins(self):
        for animal in self.foundation_brains:
            brainManager = BrainStructureManager(animal, self.um, self.debug)
            brainManager.test_brain_volumes_and_origins(animal)

    def merge_origin_creation(self):
        polygon_annotator_id = 1
        animal_users = [['MD585', polygon_annotator_id], ['MD589', polygon_annotator_id], ['MD594', polygon_annotator_id]]
        for animal, polygon_annotator_id in sorted(animal_users):
            #print(f'animal={animal} annotator={polygon_annotator_id}')
            self.brainManager.polygon_annotator_id = polygon_annotator_id
            # The fixed brain is, well, fixed. 
            # All foundation polygon brain data is under: Edward ID=1
            # All foundation COM brain data is under: Beth ID=2
            self.brainManager.fixed_brain = BrainStructureManager('MD589', debug)
            self.brainManager.fixed_brain.com_annotator_id = 2
            self.brainManager.com_annotator_id = 2
            self.brainManager.compute_origin_and_volume_for_brain_structures(self.brainManager, self.atlasMerger, 
                                                                        animal, polygon_annotator_id)

        for structure in tqdm(self.atlasMerger.volumes_to_merge, desc='Merging volumes', disable=False):
            volumes = self.atlasMerger.volumes_to_merge[structure]
            volume = self.atlasMerger.merge_volumes(structure, volumes)
            self.atlasMerger.volumes[structure]= volume

        if len(self.atlasMerger.origins_to_merge) > 0:
            print('Finished filling up volumes and origins')
            if self.animal == NEW_ATLAS:
                self.brainManager.rm_existing_dir(self.brainManager.origin_path)
                self.brainManager.rm_existing_dir(self.brainManager.volume_path)
                self.brainManager.rm_existing_dir(self.brainManager.mesh_path)
            self.atlasMerger.save_atlas_origins_and_volumes_and_meshes()
            self.atlasMerger.evaluate(self.animal)
            print('Finished saving data to disk.')
        else:
            print('No data to save')

    def create_neuroglancer_volume(self):
        self.brainManager.create_neuroglancer_volume()

    def save_atlas_volume(self):
        self.brainManager.save_atlas_volume()

    def update_atlas_coms(self):
        self.brainManager.update_atlas_coms()

    def list_coms(self):
        self.brainManager.list_coms_by_atlas()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Atlas')


    parser.add_argument('--animal', required=True, type=str)
    parser.add_argument('--debug', required=False, default='false', type=str)
    parser.add_argument('--affine', required=False, default='false', type=str)
    parser.add_argument('--um', required=False, default=10, type=int)
    
    
    parser.add_argument('--task', required=True, type=str)

    args = parser.parse_args()
    animal = str(args.animal).strip()
    task = str(args.task).strip().lower()
    debug = bool({'true': True, 'false': False}[args.debug.lower()])    
    affine = bool({'true': True, 'false': False}[args.affine.lower()])    
    um = args.um

    if task == 'update_coms' and affine:
        print('Cannot update COMs with affine set to True')
        sys.exit()

    if animal == NEW_ATLAS or animal == ORIGINAL_ATLAS:
        print(f'Working on {animal}')
    else:
        print(f'{animal} is not a valid animal. Choose one of these:')
        print(f'\t{NEW_ATLAS}')
        print(f'\t{ORIGINAL_ATLAS}')
        sys.exit()

    pipeline = AtlasManager(animal, um, affine, debug)

    function_mapping = {'create_brain_json': pipeline.create_brain_json,
                        'draw_brains': pipeline.test_brain_volumes_and_origins,
                        'create_volumes': pipeline.create_brain_volumes_and_origins,
                        'merge_volumes': pipeline.merge_origin_creation,
                        'neuroglancer': pipeline.create_neuroglancer_volume,
                        'save_atlas': pipeline.save_atlas_volume,
                        'update_coms': pipeline.update_atlas_coms,
                        'list_coms': pipeline.list_coms,
    }

    if task in function_mapping:
        function_mapping[task]()
    else:
        print(f'{task} is not a correct task. Choose one of these:')
        for key in function_mapping.keys():
            print(f'\t{key}')



