import argparse
import sys
from pathlib import Path

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.registration.brain_structure_manager import BrainStructureManager
from library.registration.brain_merger import BrainMerger
from library.controller.polygon_sequence_controller import PolygonSequenceController
from library.controller.structure_com_controller import StructureCOMController



def volume_origin_creation(debug=False):
    structureController = StructureCOMController('MD589')
    polygonController = PolygonSequenceController('MD589')
    sc_sessions = structureController.get_active_sessions()
    pg_sessions = polygonController.get_available_volumes_sessions()
    animal_users = set()
    for session in sc_sessions:
        animal_users.add((session.FK_prep_id, session.FK_user_id))
    for session in pg_sessions:
        animal_users.add((session.FK_prep_id, session.FK_user_id))

    
    animal_users = list(animal_users)
    brainMerger = BrainMerger(debug)
    animal_users = [['MD585',3], ['MD589',3], ['MD594',3]]
    for animal_user in sorted(animal_users):
        animal = animal_user[0]
        polygon_annotator_id = animal_user[1]
        if 'test' in animal or 'Atlas' in animal:
            continue
        brainManager = BrainStructureManager(animal, debug)
        brainManager.polygon_annotator_id = polygon_annotator_id
        brainManager.fixed_brain = BrainStructureManager('Allen', debug)
        brainManager.fixed_brain.com_annotator_id = 1
        brainManager.com_annotator_id = 2
        brainManager.compute_origin_and_volume_for_brain_structures(brainManager, brainMerger, 
                                                                    polygon_annotator_id=polygon_annotator_id)
        brainManager.save_brain_origins_and_volumes_and_meshes()

    if debug:
        return
    
    for structure in brainMerger.volumes_to_merge:
        volumes = brainMerger.volumes_to_merge[structure]
        volume = brainMerger.get_merged_landmark_probability(structure, volumes)
        brainMerger.volumes[structure]= volume

    if len(brainMerger.origins_to_merge) > 0:
        print('Finished filling up volumes and origins')
        brainMerger.save_atlas_origins_and_volumes_and_meshes()
        brainMerger.save_coms_to_db()
        brainMerger.evaluate()
        print('Finished saving data to disk and to DB.')
    else:
        print('No data to save')

"""
def evaluate_registration(self):
    mcc = MouseConnectivityCache(resolution=25)
    rsp = mcc.get_reference_space()
    allen_structure_id = 661 # facial nucleus
    sc_dict = {'Superior colliculus, sensory related': 302,
                'Superior colliculus, optic layer': 851,
                'Superior colliculus, superficial gray layer': 842,
                'Superior colliculus, zonal layer': 834}
    for structure, allen_structure_id in allen_structures.items():
        structure_mask = rsp.make_structure_mask([allen_structure_id], direct_only=False)
        structure_mask = np.swapaxes(structure_mask, 0, 2)
        atlaspath = os.path.join(self.atlas_path, 'atlasV8.tif')
        atlasImage = io.imread(atlaspath)
        atlasImage[atlasImage != allen_structure_id] = 0
        #print('atlas ', atlasImage.shape)
        #print('structure_mask', structure_mask.shape)
        structure_mask_padded = np.pad(structure_mask, ((0,0), (0,100), (0, 100)), 'constant')
        #print('padded ', structure_mask_padded.shape)

        #break            
        dice_coefficient = dice(structure_mask_padded, atlasImage)
        print(f'Structure: {structure} dice coefficient={dice_coefficient}')
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Atlas')
    parser.add_argument('--animal', required=False, default='atlasV8')
    parser.add_argument('--debug', required=False, default='false', type=str)
    args = parser.parse_args()
    debug = bool({'true': True, 'false': False}[args.debug.lower()])    
    volume_origin_creation(debug)
