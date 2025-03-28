import argparse
import sys
from pathlib import Path
from timeit import default_timer as timer
from tqdm import tqdm

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.atlas.atlas_utilities import NEW_ATLAS, ORIGINAL_ATLAS
from library.atlas.brain_structure_manager import BrainStructureManager
from library.atlas.brain_merger import BrainMerger


class AtlasManager():

    def __init__(self, animal, task, um=10, affine=False, debug=False):

        self.animal = animal
        self.brainManager = BrainStructureManager(animal, um, affine, debug)
        self.atlasMerger = BrainMerger(animal)
        self.task = task
        self.debug = debug
        self.um = um
        self.foundation_brains = ['MD589', 'MD594', 'MD585']
        #self.foundation_brains = ['MD594']

    def create_brain_json(self):
        """
        # 1st step, this parses the original CSV files and creates the JSON files
        """
        for animal in self.foundation_brains:
            brainManager = BrainStructureManager(animal, self.um, self.debug)
            brainManager.create_brain_json(animal, self.debug)

    def create_brain_volumes_and_origins(self):
        """
        # 2nd step, this takes the JSON files and creates the brain volumes and origins
        """
        start_time = timer()
        for animal in self.foundation_brains:
            brainMerger = BrainMerger(animal)
            self.brainManager.create_brain_volumes_origins(brainMerger, animal, self.debug)
            if not self.debug:
                brainMerger.save_brain_coms_meshes_origins_volumes()

        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        print(f"{self.task} took {total_elapsed_time} seconds")

    def merge_foundation_origin_creation(self):
        """
        # 3rd step, this merges the volumes and origins from the foundation brains into the new atlas
        # The fixed brain is, well, fixed. 
        # All foundation polygon brain data is under: Edward ID=1
        # All foundation COM brain data is under: Beth ID=2"
        """
        start_time = timer()
        polygon_annotator_id = 1
        foundation_animal_users = [['MD585', polygon_annotator_id], ['MD589', polygon_annotator_id], ['MD594', polygon_annotator_id]]
        for animal, polygon_annotator_id in sorted(foundation_animal_users):
            self.brainManager.polygon_annotator_id = polygon_annotator_id
            self.brainManager.fixed_brain = BrainStructureManager('MD589', debug)
            self.brainManager.fixed_brain.com_annotator_id = 2
            self.brainManager.com_annotator_id = 2
            self.brainManager.create_brains_origin_volume(self.atlasMerger, animal, self.brainManager.fixed_brain)

        structures = ['TG_L', 'TG_R']
        for structure in structures:
            self.brainManager.create_brains_origin_volume_from_polygons(self.atlasMerger, structure, self.debug)
        

        for structure in tqdm(self.atlasMerger.volumes_to_merge, desc='Merging atlas origins/volumes', disable=False):
            volumes = self.atlasMerger.volumes_to_merge[structure]
            volume = self.atlasMerger.merge_volumes(structure, volumes)
            self.atlasMerger.volumes[structure]= volume

        if len(self.atlasMerger.origins_to_merge) > 0:
            self.atlasMerger.save_atlas_meshes_origins_volumes()
        else:
            print('No data to save')

        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        print(f"{self.task} took {total_elapsed_time} seconds")

    def test_brain_volumes_and_origins(self):
        """
        # optional step, this draws the brains from the cleaned images so you can check the placement of the volumes    
        """
        for animal in self.foundation_brains:
            brainManager = BrainStructureManager(animal, self.um, self.debug)
            brainManager.test_brain_volumes_and_origins(animal)

    def create_neuroglancer_volume(self):
        self.brainManager.create_neuroglancer_volume()

    def save_atlas_volume(self):
        self.brainManager.save_atlas_volume()

    def update_atlas_coms(self):
        self.brainManager.update_atlas_coms()

    def list_coms(self):
        self.brainManager.list_coms_by_atlas()

    def validate(self):
        self.brainManager.validate_volumes()

    def evaluate(self):
        self.brainManager.evaluate()



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

    pipeline = AtlasManager(animal, task, um, affine, debug)

    function_mapping = {'create_brain_json': pipeline.create_brain_json,
                        'draw': pipeline.test_brain_volumes_and_origins,
                        'create_volumes': pipeline.create_brain_volumes_and_origins,
                        'merge_volumes': pipeline.merge_foundation_origin_creation,
                        'neuroglancer': pipeline.create_neuroglancer_volume,
                        'save_atlas': pipeline.save_atlas_volume,
                        'update_coms': pipeline.update_atlas_coms,
                        'list_coms': pipeline.list_coms,
                        'validate': pipeline.validate,
                        'evaluate': pipeline.evaluate,
    }

    if task in function_mapping:
        function_mapping[task]()
    else:
        print(f'{task} is not a correct task! Choose one of these:')
        for key in function_mapping.keys():
            print(f'\t{key}')



