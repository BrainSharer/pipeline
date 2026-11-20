import argparse
import math
from pathlib import Path
import sys
from typing import List, Tuple


import numpy as np
PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.image_manipulation.precomputed_manager import NgPrecomputedMaker

def _compute_chunk_size(res: List[float], size: List[int]) -> Tuple[int, int, int]:
    """Finds a power-of-two chunk shape balanced in physical space."""
    chunk = [1, 1, 1]
    target_chunk_voxels: int = 64
    while math.prod(chunk) < target_chunk_voxels:
        # Only expand axes that haven't exceeded the volume's current dimension
        candidates = [i for i in range(3) if chunk[i] < size[i]]
        if not candidates:
            break

        # Expand the axis with the smallest physical span (chunk_dim * voxel_res)
        smallest_phys_axis = min(candidates, key=lambda i: chunk[i] * res[i])
        chunk[smallest_phys_axis] *= 2

    return tuple(chunk)



def print_pyramid(factors, resolutions, chunks):
    print(f"{'Mip':<10} {'Factor':<30} {'Resolution (µm)':<30} {'Chunk'}")
    print("-" * 90)
    for i, (r, s, c) in enumerate(zip(factors, resolutions, chunks)):
        print(f"{i:<10} {str(r):<30} {str(s):<30} {c}")

if __name__ == "__main__":
    # Example usage:
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--xy", help="xy resolution", required=True, type=float)
    parser.add_argument("--z", help="z resolution", required=True, type=float)

    args = parser.parse_args()
    x = args.xy
    y = args.xy
    z = args.z

    resolution = [x, y, z]  # microns

    #scales, resolutions, chunks = compute_mipmaps(resolution)
    #print(f'resolution: {resolutions}')
    #print(f'scales: {scales}')
    #print(f'chunks: {chunks}')

    #for mip, (scale, resolution, chunk) in enumerate(zip(scales, resolutions, chunks)):
    #    print(f'{mip=} Scale factor: {scale} Resolution: {resolution} Chunk: {chunk}')
    base_chunk_size = (64,64,64)

    
    resolutions, scales, chunks = NgPrecomputedMaker.compute_mipmaps(base_resolution=resolution, base_chunk_size=base_chunk_size)

    while True:
        chunk_size = _compute_chunk_size(resolution, [60000,35000,450])


    print_pyramid(resolutions, scales, chunks)