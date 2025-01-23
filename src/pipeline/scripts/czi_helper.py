import numpy as np
from aicsimageio import AICSImage
import os
import sys
import argparse
from pathlib import Path
PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.image_manipulation.czi_manager import CZIManager



def run_main(animal, czi_file):

    czi_file_path = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/czi'
    infile = os.path.join(czi_file_path, czi_file)
    czi_aics = AICSImage(infile)
    #czi = CziFile(infile)
    czi = CZIManager(infile)

    #metadata = czi_aics.metadata  # returns the metadata object for this file format (XML, JSON, etc.)
    #print(metadata)
    #for element in metadata.findall('*'):
    #    for e2 in element.findall('*'):
    #        print(e2)




    #dimensions = czi.get_dims_shape() 
    #print(f'dimensions {dimensions}')
    #dims = czi.dims  # BSCZYX
    #print(f'dims={dims}')
    #sizes = czi.size  # (1, 40, 4, 60, 1300, 1900)
    #print(f'sizes={sizes}')
    total_scenes = czi_aics.scenes
    scales = [1]
    channels = czi_aics.dims.C
    print(f"{czi_file} has {len(total_scenes)} scenes and {channels} channels")
    for scale in scales:
        for idx, scenei in enumerate(total_scenes):
            czi_aics.set_scene(scenei)
            print(f"{scale=} Scene {idx=} {scenei=}", end=" ")
            data = czi.get_scene(scale=scale, scene_index=idx, channel=1)
            print(f'shape={data.shape} data type={data.dtype}')
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True)
    parser.add_argument("--czi", help="Enter the filename", required=True, type=str)

    args = parser.parse_args()

    animal = args.animal
    czi = args.czi

    run_main(animal, czi)



