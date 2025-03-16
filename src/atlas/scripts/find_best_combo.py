import argparse
import concurrent.futures
import sys
from pathlib import Path

import numpy as np
from itertools import combinations
from timeit import default_timer as timer

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.atlas.atlas_utilities import apply_affine_transform, list_coms, compute_affine_transformation


atlas_all = list_coms('Atlas')
allen_all = list_coms('Allen')
common_keys = list(atlas_all.keys() & allen_all.keys())


class VariableManager:
    def __init__(self, ncombos, cpus):
        self.ncombos = ncombos
        self.cpus = cpus


    
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


    def create_combos(self):
        start_time = timer()
        starting_rms = 999

        combos = generate_combinations(common_keys, ncombos)
        print(f'Trying {len(combos)} different combinations')
        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        print(f'Elapsed time to get combos: {total_elapsed_time} seconds')

        start_time = timer()
        errors = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.cpus) as executor:
                # Map function to items in the list concurrently
                for found_combo, rms in executor.map(find_best_combo, combos):
                    if rms < starting_rms:
                        starting_rms = rms
                        print(f'New best combo: {found_combo} with {rms=}') 
                        errors.append([found_combo, rms])

        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        print(f'Elapsed time to get combos: {total_elapsed_time} seconds')

        #r = sorted(errors, key=lambda item: list(item.values())[0], reverse=True)
        for combo, rms in errors:
            print(f'{combo=} {rms=}')


def find_best_combo(combo):
    sss = []
    atlas_src = np.array([atlas_all[s] for s in combo])
    allen_src = np.array([allen_all[s] for s in combo])

    matrix = compute_affine_transformation(atlas_src, allen_src)
    for structure in common_keys:
        atlas0 = np.array(atlas_all[structure])
        allen0 = np.array(allen_all[structure]) 
        transformed = apply_affine_transform(atlas0, matrix)
        transformed = [round(x,8) for x in transformed]
        difference = [round(a - b, 8) for a, b in zip(transformed, allen0)]
        ss = sum_square_com(difference)
        sss.append(ss)

    return (combo, sum(sss) / len(common_keys))


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



def do_this():
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Atlas')
    parser.add_argument('--ncombos', required=True, type=int)
    parser.add_argument('--cpus', required=True, type=int)
    
    args = parser.parse_args()
    cpus = args.cpus
    ncombos = args.ncombos

    pipeline = VariableManager(ncombos, cpus)
    pipeline.create_combos()







