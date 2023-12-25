import argparse
import sys
from pathlib import Path

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.registration.brain_structure_manager import BrainStructureManager
from library.registration.brain_merger import BrainMerger
from library.controller.polygon_sequence_controller import PolygonSequenceController
from library.controller.structure_com_controller import StructureCOMController

# get average brain the same scale as atlas
# put the dk atlas on the average brain

def volume_origin_creation(region, debug=False):
    brainMerger = BrainMerger(debug)
    structureController = StructureCOMController('MD589')
    polygonController = PolygonSequenceController('MD589')
    sc_sessions = structureController.get_active_sessions()
    pg_sessions = polygonController.get_available_volumes_sessions()
    animal_users = set()
    # structures with COMs to calculate R and t
    animals = [session.FK_prep_id for session in sc_sessions]
    # structures with polygon data
    # We want to draw data from the polygons, but we can only use polygon data that also 
    # has a COM
    for session in pg_sessions:
        if session.FK_prep_id in animals:
            animal_users.add((session.FK_prep_id, session.FK_user_id))

    
    animal_users = list(animal_users)
    polygon_annotator_id = 1
    #animal_users = [['MD585', polygon_annotator_id], ['MD589', polygon_annotator_id], ['MD594', polygon_annotator_id]]
    #animal_users = [['MD589', polygon_annotator_id]]
    for animal_user in sorted(animal_users):
        animal = animal_user[0]
        polygon_annotator_id = animal_user[1]
        if 'test' in animal or 'Atlas' in animal or 'Allen' in animal:
            continue
        if polygon_annotator_id == 2 and animal == 'MD589':
            continue
        print(f'animal={animal} annotator={polygon_annotator_id}')
        brainManager = BrainStructureManager(animal, 'all', debug)
        brainManager.polygon_annotator_id = polygon_annotator_id
        # The fixed brain is, well, fixed. 
        # All foundation polygon brain data is under: Edward ID=1
        # All foundation COM brain data is under: Beth ID=2
        brainManager.fixed_brain = BrainStructureManager('MD589', debug)
        brainManager.fixed_brain.com_annotator_id = 2
        brainManager.com_annotator_id = 2
        brainManager.compute_origin_and_volume_for_brain_structures(brainManager, brainMerger, 
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Atlas')
    parser.add_argument('--animal', required=False, default='atlasV8')
    parser.add_argument('--debug', required=False, default='false', type=str)
    parser.add_argument('--region', required=False, default='all', type=str)
    args = parser.parse_args()
    debug = bool({'true': True, 'false': False}[args.debug.lower()])    
    region = args.region.lower()
    regions = ['midbrain', 'all', 'brainstem']
    if region not in regions:
        print(f'regions is wrong {region}')
        print(f'use one of: {regions}')
        sys.exit()
    volume_origin_creation(region, debug)
