"""This program will create an atlas from the original anatomist's annotations and create a new Atlas named AtlasV8. \
      The tasks are run in sequence. Data is saved to disk on birdstore, so you only need to rerun the tasks if you change the data.

- python src/atlas/scripts/create_atlas.py --task json --debug false
- python src/atlas/scripts/create_atlas.py --task create --debug false
- python src/atlas/scripts/create_atlas.py --task merge --debug false
- python src/atlas/scripts/create_atlas.py --task neuroglancer --debug false --affine true

Explanation for the tasks:

- json - This parses the original anatomist's annotations and creates JSON data. This makes it easier \
    for the later steps.
- create - This take the JSON data and creates the origins and volumes of each of the 3 foundation brains. \
    The origin is the upper left corner of the volume box. Origins at 0,0,0 start on the very first section in the upper left corner.
- merge - This takes each of the 3 foundation brains ands merges them into one volume. It initially uses elastix \
    to align the volumes with a affine transformation. The mean of these aligned images is then used to create the volume. \
    The volume will contain only zeros or the Allen color. This process will also take any polygons stored in the database, \
    e.g., the TG_L and TG_R, and merge them into a volume. 
- neuroglancer - This takes the merged volume and moves the origins into Allen space. A neuroglancer view is then created from \
    all these merged volumes.
"""

import argparse
import sys
from pathlib import Path
from timeit import default_timer as timer
from tqdm import tqdm


PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.atlas.atlas_utilities import list_coms, load_transformation
from library.atlas.brain_structure_manager import BrainStructureManager
from library.atlas.brain_merger import BrainMerger
from library.utilities.utilities_process import SCALING_FACTOR


class AtlasManager():

    def __init__(self, animal, task, um=10, affine=False, scaling_factor=SCALING_FACTOR, debug=False):

        self.animal = animal
        self.brainManager = BrainStructureManager(animal, um, affine, scaling_factor, debug)
        self.atlasMerger = BrainMerger(animal)
        self.task = task
        self.debug = debug
        self.um = um
        self.foundation_brains = ['MD589', 'MD594', 'MD585']

    def create_brain_json(self):
        """
        # 1st step, this parses the original CSV files and creates the JSON files
        All data is downsampled by 1/32=SCALING_FACTOR. This is also the same size
        as the downsampled images.
        """
        if self.debug:
            self.foundation_brains = ['MD585']
        for animal in self.foundation_brains:
            brainManager = BrainStructureManager(animal, self.um, self.debug)
            brainManager.create_brain_json(animal, self.debug)

    def create_brain_volumes_and_origins(self):
        """
        # 2nd step, this takes the JSON files and creates the brain volumes and origins
        """
        start_time = timer()
        self.brainManager.fixed_brain = BrainStructureManager('MD589', self.debug)
        if self.debug:
            self.foundation_brains = ['MD585']
        for animal in self.foundation_brains:
            brainMerger = BrainMerger(animal)
            self.brainManager.create_foundation_brain_volumes_origins(brainMerger, animal, self.debug)
            if not self.debug:
                brainMerger.save_foundation_brain_coms_meshes_origins_volumes()

        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        print(f"{self.task} took {total_elapsed_time} seconds")


    def merge_all(self):
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
            self.brainManager.create_foundation_brains_origin_volume(self.atlasMerger, animal)
        
        # Note, for DK78, The C1 source is C1.v1
        
        
        #brains = ['MD585']
        for animal in self.foundation_brains:
            #transform = load_transformation(animal, self.um, self.um)
            transform = None
            structure_coms = list_coms(animal)
            structures = sorted(structure_coms.keys())
            #structures = ['SC']
            for structure in structures:
                print(f'Processing {animal} {structure}')
                self.brainManager.create_brains_origin_volume_from_polygons(self.atlasMerger, animal, structure, transform, self.debug)

        
        other_brains = ['DK78', 'DK79']
        other_structures = ['TG_L', 'TG_R']

        for animal in other_brains:
            transform = load_transformation(animal, self.um, self.um)
            for structure in other_structures:
                self.brainManager.create_brains_origin_volume_from_polygons(self.atlasMerger, animal, structure, transform, self.debug)


        for structure in tqdm(self.atlasMerger.volumes_to_merge, desc='Merging atlas origins/volumes', disable=self.debug):
            volumes = self.atlasMerger.volumes_to_merge[structure]
            volume = self.atlasMerger.merge_volumes(structure, volumes)
            self.atlasMerger.volumes[structure]= volume

        if len(self.atlasMerger.origins_to_merge) > 0:
            self.atlasMerger.save_atlas_meshes_origins_volumes(self.um)
        else:
            print('No data to save')

        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        print(f"{self.task} took {total_elapsed_time} seconds")

    def test_brain_volumes_and_origins(self):
        """
        # optional step, this draws the brains from the cleaned images so you can check the placement of the volumes
        # The output is in /net/birdstore/Active_Atlas_Data/data_root/pipeline_data/MDXXX/preps/C1/drawn    
        """
        brainManager = BrainStructureManager(self.animal, self.um, self.debug)
        brainManager.test_brain_volumes_and_origins(self.animal)

    def create_precomputed(self):
        """
        # optional step, this draws the brains from the cleaned images so you can check the placement of the volumes
        # The output is in /net/birdstore/Active_Atlas_Data/data_root/pipeline_data/MDXXX/preps/C1/drawn    
        """
        brainManager = BrainStructureManager(self.animal, self.um, self.debug)
        brainManager.create_cloud_volume()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Atlas')


    parser.add_argument('--animal', required=False, default='AtlasV8', type=str)
    parser.add_argument('--debug', required=False, default='false', type=str)
    parser.add_argument('--affine', required=False, default='false', type=str)
    parser.add_argument('--um', required=False, default=10.0, type=float)
    parser.add_argument('--scaling_factor', required=False, default=SCALING_FACTOR, type=float)
    
    
    parser.add_argument('--task', required=True, type=str)

    args = parser.parse_args()
    animal = str(args.animal).strip()
    task = str(args.task).strip().lower()
    debug = bool({'true': True, 'false': False}[args.debug.lower()])    
    affine = bool({'true': True, 'false': False}[args.affine.lower()])
    scaling_factor = float(args.scaling_factor)   
    um = args.um

    if task == 'update_coms' and affine:
        print('Cannot update COMs with affine set to True')
        sys.exit()

        
    pipeline = AtlasManager(animal, task, um, affine, scaling_factor, debug)

    function_mapping = {'json': pipeline.create_brain_json,
                        'draw': pipeline.test_brain_volumes_and_origins,
                        'create': pipeline.create_brain_volumes_and_origins,
                        'merge': pipeline.merge_all,
                        'neuroglancer': pipeline.brainManager.create_neuroglancer_volume,
                        'save_atlas': pipeline.brainManager.save_atlas_volume,
                        'create_atlas': pipeline.brainManager.create_atlas_volume,
                        'update_coms': pipeline.brainManager.update_atlas_coms,
                        'list_coms': pipeline.brainManager.list_coms_by_atlas,
                        'validate': pipeline.brainManager.validate_volumes,
                        'evaluate': pipeline.brainManager.evaluate,
                        'status': pipeline.brainManager.report_status,
                        'update_volumes': pipeline.brainManager.fetch_create_volumes,
                        'precomputed': pipeline.create_precomputed,
                        'atlas2allen': pipeline.brainManager.atlas2allen,
    }

    if task in function_mapping:
        function_mapping[task]()
    else:
        print(f'{task} is not a correct task! Choose one of these:')
        for key in function_mapping.keys():
            print(f'\t{key}')



 