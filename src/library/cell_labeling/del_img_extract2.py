# CREATED: 23-OCT-2023
# LAST EDIT: 25-OCT-2023
# AUTHORS: DUANE RINEHART, MBA (drinehart@ucsd.edu)
# CREATES LOOPS THROUGH OME-Zarr FOLDER TO EXTRACT HIGH RESOLUTION IMAGES FROM FORMAT AND STORE IN SEPARATE FOLDER BY SECTION (FILENAME)

import os
import zarr
import ome_zarr
from pathlib import Path
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import imageio
import argparse
import concurrent.futures as cf
from joblib import Parallel, delayed

base_path_linux = Path('/net/birdstore/')
base_path_win = Path('z:/')

if os.name == 'nt':
    base_path = base_path_win
else:
    base_path = base_path_linux


def collectArguments():
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True)
    parser.add_argument("--channel", help="Enter channel", required=False, default=1)
    args = parser.parse_args()
    return args


def compute_and_save(img, full_output_file):
    print(f'CURRENT SECTION/IMAGE: {full_output_file}')
    cur_image = img.compute()
    imageio.imsave(full_output_file, cur_image)


def main():
    args = collectArguments()
    animal = args.animal
    channel = args.channel

    zarr_filename = str(channel) + '.zarr'

    zarr_path = Path(base_path, 'drinehart/allen_brain_aws/', animal, zarr_filename)
    output_path = Path(base_path, 'drinehart/allen_brain_aws/output/', animal, channel)

    print('USING PARAMETERS:')
    print(f'INPUT OME-Zarr PATH: {zarr_path}')
    print(f'OUTPUT PATH: {output_path}')
    print(f'ANIMAL: {animal}')
    print(f'CHANNEL: {channel}')

    # Open the OME-Zarr file
    store = parse_url(zarr_path, mode="r").store
    reader = Reader(parse_url(zarr_path))
    nodes = list(reader())
    image_node = nodes[0]  # first node is image pixel data
    dask_data = image_node.data
    total_sections = dask_data[0].shape[2]
    print(f'total sections: {total_sections}')
    
    args2 = []
    # Loop over the images in the file, access the full resolution image data and store
    for section, img in enumerate(dask_data[0][0][0]):  # time, channel, z coordinate

        filename = str(section).zfill(4) + '.tiff'
        full_output_file = Path(output_path, filename)
        Path(output_path).mkdir(parents=True, exist_ok=True)

        if section < 1:
            print(f'individual image shape {img.shape}')

        args2.append((img, full_output_file))
            
            
    Parallel(n_jobs=20)(delayed(compute_and_save)(i[0], i[1]) for i in args2)
    

if __name__ == "__main__":
    main()
