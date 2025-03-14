import concurrent.futures
import sys
from pathlib import Path

import numpy as np
from itertools import combinations
from timeit import default_timer as timer

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.atlas.atlas_utilities import apply_affine_transform, list_coms, compute_affine_transformation

def absolute_sum(l):
    la = np.array(l)
    return np.sum(la, axis=0)

def sum_square_com(com):
    ss = np.sqrt(sum([s*s for s in com]))
    return ss

def generate_combinations(lst, ncombos=5):
    """
    Generate all combinations of at least 3 elements from the given list.
    
    :param lst: List of elements
    :return: List of tuples containing the combinations
    """
    return list(combinations(lst, ncombos ))

def find_best_combo(combo):
    atlas_src = np.array([atlas_all[s] for s in combo])
    allen_src = np.array([allen_all[s] for s in combo])
    sss = []
    error = {}
    matrix = compute_affine_transformation(atlas_src, allen_src)
    for structure in common_keys:
        atlas0 = np.array(atlas_all[structure])
        allen0 = np.array(allen_all[structure]) 
        transformed = apply_affine_transform(atlas0, matrix)
        transformed = [round(x,8) for x in transformed]
        difference = [round(a - b, 8) for a, b in zip(transformed, allen0)]
        ss = sum_square_com(difference)
        sss.append(ss)
    error[combo] = sum(sss)

    return error

def create_combos():
    start_time = timer()
    ncombos = 3
    combinations_list = generate_combinations(common_keys, ncombos=5)
    print(f'Trying {len(combinations_list)} different combinations')
    end_time = timer()
    total_elapsed_time = round((end_time - start_time), 2)
    print(f'Elapsed time to get combos: {total_elapsed_time} seconds')

    start_time = timer()
    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(find_best_combo, file_key) for file_key in combinations_list]

        for future in concurrent.futures.as_completed(futures):
            try:
                one_error = future.result()
                errors.append(one_error)
            except Exception as e:
                print(f"Task failed: {e}")    

    end_time = timer()
    total_elapsed_time = round((end_time - start_time), 2)
    print(f'Elapsed time to get combos: {total_elapsed_time} seconds')
    r = sorted(errors, key=lambda item: list(item.values())[0], reverse=True)
    for k in r[:3]:
        print(f'{k}')


class VariableManager:
    def __init__(self, starting_keys):
        self.variables = starting_keys
        self.base_score = 947.2633003961902
    
    def update_variables(self, key, score):
        """
        Adds a new variable if the score exceeds the threshold_add.
        Removes a variable if the score drops below the threshold_remove.
        """
        
        if score < self.base_score:
            self.variables.append(key)
            print(f"Added: {key} = {score} and len of keys: {len(self.variables)}")
        
        if score > self.base_score and key in self.variables:
            self.variables.remove(key)
            print(f"Removed: {key}")
    
    def get_variables(self):
        return self.variables


atlas_all = list_coms('Atlas')
allen_all = list_coms('Allen')
common_keys = list(atlas_all.keys() & allen_all.keys())
atlas_common = np.array([atlas_all[s] for s in common_keys])
allen_common = np.array([allen_all[s] for s in common_keys])

starting_keys = ['LRt_L', 'LRt_R', 'SC', 'SNC_R', 'SNC_L']
#starting_keys = common_keys
atlas_src = np.array([atlas_all[s] for s in starting_keys])
allen_src = np.array([allen_all[s] for s in starting_keys])
#vm = VariableManager(starting_keys)

remaining_keys = set(common_keys) - set(starting_keys)
for key in remaining_keys:
    atlas_src = np.vstack([atlas_src, atlas_all[key]])
    allen_src = np.vstack([allen_src, allen_all[key]])
    sss = []
    error = {}
    matrix = compute_affine_transformation(atlas_src, allen_src)
    for structure in common_keys:
        atlas0 = np.array(atlas_all[structure])
        allen0 = np.array(allen_all[structure]) 
        transformed = apply_affine_transform(atlas0, matrix)
        transformed = [round(x,8) for x in transformed]
        difference = [round(a - b, 8) for a, b in zip(transformed, allen0)]
        ss = sum_square_com(difference)
        sss.append(ss)
    error = sum(sss)
    print(f'{key} = {error}')
    # Example Usage
    #vm.update_variables(key, error)  # Adds a variable
    if error < 947.2633003961902:
        starting_keys.append(key)
    else:
        atlas_src = np.array([atlas_all[s] for s in starting_keys])
        allen_src = np.array([allen_all[s] for s in starting_keys])

print(f'Best combo: {starting_keys}')






