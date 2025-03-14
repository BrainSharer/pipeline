import sys
from pathlib import Path

import numpy as np
from itertools import combinations

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.atlas.atlas_utilities import apply_affine_transform, list_coms, compute_affine_transformation

def absolute_sum(l):
    la = np.array(l)
    return np.sum(la, axis=0)

def sum_square_com(com):
    ss = np.sqrt(sum([s*s for s in com]))
    return ss

def generate_combinations(lst, ncombos):
    """
    Generate all combinations of at least 3 elements from the given list.
    
    :param lst: List of elements
    :return: List of tuples containing the combinations
    """
    #return list(combinations(lst, ncombos ))

    result = []
    for r in range(ncombos, 10):
        result.extend(combinations(lst, r))
    return result


atlas_all = list_coms('Atlas')
allen_all = list_coms('Allen')
common_keys = list(atlas_all.keys() & allen_all.keys())
atlas_common = np.array([atlas_all[s] for s in common_keys])
allen_common = np.array([allen_all[s] for s in common_keys])

ncombos = 3
combinations_list = generate_combinations(common_keys, ncombos=ncombos)
print(f'Found {len(combinations_list)} for ncombos={ncombos}')

error = {}

for combo in combinations_list[:10000]:
    atlas_src = np.array([atlas_all[s] for s in combo])
    allen_src = np.array([allen_all[s] for s in combo])
    df_list = []
    sss = []
    matrix = compute_affine_transformation(atlas_src, allen_src)
    for structure in common_keys:
        atlas0 = np.array(atlas_all[structure])
        allen0 = np.array(allen_all[structure]) 
        transformed = apply_affine_transform(atlas0, matrix)
        transformed = [round(x,8) for x in transformed]
        difference = [round(a - b, 8) for a, b in zip(transformed, allen0)]
        ss = sum_square_com(difference)
        row = [structure, atlas0, allen0, transformed, difference, ss]
        df_list.append(row)
        sss.append(ss)
    error[combo] = sum(sss)



sorted_combos = {k: v for k, v in sorted(error.items(), key=lambda item: item[1])}
results = list(sorted_combos.items())[:5]

for result in results:
    print(result)
