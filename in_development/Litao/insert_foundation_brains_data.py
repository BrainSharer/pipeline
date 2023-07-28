import argparse
import os, sys
import json
from tqdm import tqdm
import string
import random
import csv

HOME = os.path.expanduser("~")
PATH = os.path.join(HOME, 'programming/pipeline_utility/src')
sys.path.append(PATH)
from pipeline.lib.file_location import DATA_PATH
from pipeline.lib.sqlcontroller import SqlController

def random_string() -> str:
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=40))


def fetch_n_insert(animal):
    sqlController = SqlController(animal)
    SCALING_FACTOR = 32.0
    xy_scale = 0.452 # same resolution for all foundation brains
    z_scale = 20 # same for all foundation brains
    FK_owner_id = 16 # Lauren
    FK_input_id = 1 # manual
    label = 'Atlas_Structures'
    ATLAS = 'atlasV8'
    CSVPATH = os.path.join(DATA_PATH, 'atlas_data', ATLAS, animal)
    jsonpath = os.path.join(CSVPATH,  'aligned_padded_structures.json')
    print(jsonpath)
    with open(jsonpath) as f:
        aligned_dict = json.load(f)    
    
    # load structure IDs and colors into dictionary which is much faster
    # than querying the DB every row
    structure_dict = sqlController.get_structures_dict()
    csvpath = os.path.join(HOME, 'sql', f'{label}.csv')
    f = open(csvpath, 'w')
    writer = csv.writer(f, delimiter=',', lineterminator='\r\n', quoting=csv.QUOTE_ALL)
    active = 1
    for structure, points in tqdm(aligned_dict.items()):
        if structure in structure_dict:
            structure_info = structure_dict[structure]
            FK_structure_id = structure_info[2]
            for z, vertices in points.items():
                section = int(z) * z_scale
                
                segment_id = random_string()
                for vertex in vertices:
                    x = int(vertex[0] * SCALING_FACTOR * xy_scale)
                    y = int(vertex[1] * SCALING_FACTOR * xy_scale)
                    #sqlController.add_layer_data_row(animal, FK_owner_id, FK_input_id, \
                    #                                 coordinates, FK_structure_id, label, segment_id=segment_id)
                    row = (str(animal), FK_owner_id, FK_input_id, FK_structure_id, 
                           x, y, section, str(label), str(segment_id), active) 
                    writer.writerow(row)
    f.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=True)
    

    args = parser.parse_args()
    animal = args.animal
    fetch_n_insert(animal)
