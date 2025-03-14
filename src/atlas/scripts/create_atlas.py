import argparse
import sys
from pathlib import Path

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.registration.brain_structure_manager import BrainStructureManager
from library.registration.brain_merger import BrainMerger


class AtlasManager():

    def __init__(self, animal, um=10, affine=False, debug=False):

        self.animal = animal
        self.brainManager = BrainStructureManager(animal, um, affine, debug)
        self.debug = debug
        self.um = um


    def volume_origin_creation(self):
        brainMerger = BrainMerger(debug)

        polygon_annotator_id = 1
        animal_users = [['MD585', polygon_annotator_id], ['MD589', polygon_annotator_id], ['MD594', polygon_annotator_id]]
        #animal_users = [['MD589', polygon_annotator_id]]
        for animal, polygon_annotator_id in sorted(animal_users):
            if 'test' in animal or 'Atlas' in animal or 'Allen' in animal:
                continue
            if polygon_annotator_id == 2 and animal == 'MD589':
                continue
            print(f'animal={animal} annotator={polygon_annotator_id}')
            self.brainManager.polygon_annotator_id = polygon_annotator_id
            # The fixed brain is, well, fixed. 
            # All foundation polygon brain data is under: Edward ID=1
            # All foundation COM brain data is under: Beth ID=2
            self.brainManager.fixed_brain = BrainStructureManager('MD589', debug)
            self.brainManager.fixed_brain.com_annotator_id = 2
            self.brainManager.com_annotator_id = 2
            self.brainManager.compute_origin_and_volume_for_brain_structures(self.brainManager, brainMerger, 
                                                                        polygon_annotator_id)

        
        for structure in brainMerger.volumes_to_merge:
            volumes = brainMerger.volumes_to_merge[structure]
            volume = brainMerger.merge_volumes(structure, volumes)
            brainMerger.volumes[structure]= volume

        if len(brainMerger.origins_to_merge) > 0:
            print('Finished filling up volumes and origins')
            brainMerger.save_atlas_origins_and_volumes_and_meshes()
            brainMerger.save_coms_to_db()
            brainMerger.evaluate(region)
            brainMerger.save_brain_area_data()
            print('Finished saving data to disk and to DB.')
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


    parser.add_argument('--animal', required=False, default='atlasV8')
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

    pipeline = AtlasManager(animal, um, affine, debug)

    function_mapping = {'create_volumes': pipeline.volume_origin_creation,
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



