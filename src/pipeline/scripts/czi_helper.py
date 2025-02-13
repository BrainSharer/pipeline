import numpy as np
from aicsimageio import AICSImage
from aicspylibczi import CziFile
from pylibCZIrw import czi as pyczi

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

def run_mosaic(animal, czi_file, scene=0):
    inpath = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/czi/{czi_file}'
    czi = CziFile(inpath)

    # Get the shape of the data
    #dimensions = czi.dims  # 'STCZMYX'

    #czi.get_dims_shape()  # [{'X': (0, 924), 'Y': (0, 624), 'Z': (0, 1), 'C': (0, 1), 'T': (0, 1), 'M': (0, 2), 'S': (0, 1)}]
    #print(f'dims_shape={czi.get_dims_shape()}')

    #czi.is_mosaic()  # True
    print(f'is_mosaic={czi.is_mosaic()}')
    # Mosaic files ignore the S dimension and use an internal mIndex to reconstruct, the scale factor allows one to generate a manageable image
    scene = czi.get_scene_bounding_box(0)
    print(f'scenex={scene.x} scene.y={scene.y} scene.w={scene.w} scene.h={scene.h}')

    data = czi.read_mosaic(C=0, scale_factor=1.0, region=(scene.x, scene.y, scene.w, scene.h))
    print(f'mosaic_data shape={data.shape} data type={data.dtype}')
    outpath = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps/C1/test'
    os.makedirs(outpath, exist_ok=True)
    outfile = os.path.join(outpath, f'{czi_file}_{str(scene)}_{str(1)}.tif')
    write_image(outfile, data)

def run_pyczi(animal, czi_file, scene=0):
    inpath = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/czi/{czi_file}'
    with pyczi.open_czi(inpath) as czidoc:
        # get the image dimensions as a dictionary, where the key identifies the dimension
        total_bounding_box = czidoc.total_bounding_box
        print(f'total_bounding_box={total_bounding_box}')

        # get the total bounding box for all scenes
        total_bounding_rectangle = czidoc.total_bounding_rectangle
        print(f'total_bounding_rectangle={total_bounding_rectangle}')
        # get the bounding boxes for each individual scene
        scenes_bounding_rectangle = czidoc.scenes_bounding_rectangle
        print(f'scenes_bounding_rectangle={scenes_bounding_rectangle}')
        bounding_box = scenes_bounding_rectangle[0]
        print(f'bounding_box={bounding_box}')
        data = czidoc.read(plane={"T": 0, "Z": 0, "C": 0}, zoom=1.0, roi=(bounding_box.x, bounding_box.y, bounding_box.w, bounding_box.h))
        print(f'image2d shape={data.shape} data type={data.dtype} ndim={data.ndim}')
        outpath = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps/C1/test'
        os.makedirs(outpath, exist_ok=True)
        outfile = os.path.join(outpath, f'{czi_file}_{str(scene)}_{str(1)}.tif')
        write_image(outfile, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True)
    parser.add_argument("--czi", help="Enter the filename", required=True, type=str)
    parser.add_argument("--scene", help="Enter the scene idx", required=False, default=0, type=int)

    args = parser.parse_args()

    animal = args.animal
    czi = args.czi
    scene = args.scene

    #run_main(animal, czi, scene)
    #run_mosaic(animal, czi, scene)
    run_pyczi(animal, czi, scene)


