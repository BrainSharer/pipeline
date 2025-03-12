import sys
from pathlib import Path
import numpy as np

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.registration.algorithm import umeyama
from library.atlas.atlas_utilities import compute_affine_transformation, compute_affine_transformation_centroid, list_coms

atlas_structures = list_coms('Atlas')
allen_structures = list_coms('Allen')

common_keys = atlas_structures.keys() & allen_structures.keys()
atlas_src = np.array([atlas_structures[s] for s in common_keys])
allen_src = np.array([allen_structures[s] for s in common_keys])


print(f'orignal atlas src shape={atlas_src.shape} dtype={atlas_src.dtype} mean = {np.mean(atlas_src, axis=0)} min={np.min(atlas_src, axis=0)} max={np.max(atlas_src, axis=0)}')
print(f'orignal allen src shape={allen_src.shape} dtype={allen_src.dtype} mean = {np.mean(allen_src, axis=0)} min={np.min(allen_src, axis=0)} max={np.max(allen_src, axis=0)}')



"""
print('umeyama')
A, t = umeyama(atlas_src.T, allen_src.T, with_scaling=True)
transformation_matrix = np.hstack( [A, t ])
transformation_matrix = np.vstack([transformation_matrix, np.array([0, 0, 0, 1])])
print(np.array2string(transformation_matrix, separator=', '))
print()"

transform = compute_affine_transformation(atlas_src, allen_src)
print('compute affine transformation')
print(np.array2string(transform, separator=', '))
print()
"""

print('compute affine transformation centroid')
A, t, transformation = compute_affine_transformation_centroid(atlas_src, allen_src)
print(np.array2string(transformation, separator=', '))

print()
for structure in common_keys:
    print(f'allen {structure}: {allen_structures[structure]}', end="\t")
    print(f'atlas {structure}: {atlas_structures[structure]}')

#print(np.array2string(atlas_src, separator=', '))


