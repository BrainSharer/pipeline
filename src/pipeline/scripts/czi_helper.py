import numpy as np
from aicsimageio import AICSImage
import os
import sys
import argparse
from pathlib import Path

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.image_manipulation.czi_manager import CZIManager
from library.utilities.utilities_process import write_image



def run_main(animal, czi_file, scene=0):

    czi_file_path = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/czi'
    infile = os.path.join(czi_file_path, czi_file)
    czi_aics = AICSImage(infile)
    czi = CZIManager(infile)

    total_scenes = czi_aics.scenes
    scales = [1.0, 1/32]
    channels = czi_aics.dims.C
    print(f"{czi_file} has {len(total_scenes)} scenes and {channels} channels")
    for scale in scales:
        for idx, scenei in enumerate(total_scenes):
            if idx == scene:
                czi_aics.set_scene(scenei)
                print(f"{scale=} Scene {idx=} {scenei=}", end=" ")
                data = czi.get_scene(scale=scale, scene_index=idx, channel=1)
                print(f'shape={data.shape} data type={data.dtype}')
                outpath = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps/C1/test'
                os.makedirs(outpath, exist_ok=True)
                outfile = os.path.join(outpath, f'{czi_file}_{scenei}_{str(scale)}.tif')
                write_image(outfile, data)
                print(f'{outpath=}')
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True)
    parser.add_argument("--czi", help="Enter the filename", required=True, type=str)
    parser.add_argument("--scene", help="Enter the scene idx", required=False, default=0, type=int)

    args = parser.parse_args()

    animal = args.animal
    czi = args.czi
    scene = args.scene

    run_main(animal, czi, scene)



