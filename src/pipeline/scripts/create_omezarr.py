import argparse
import os
import sys
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import zarr
from ome_zarr.writer import write_multiscale
import zarr
import dask.array as da

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.utilities.utilities_process import SCALING_FACTOR
from library.utilities.dask_utilities import imreads
from library.controller.sql_controller import SqlController
from library.image_manipulation.filelocation_manager import FileLocationManager

def get_transformations(axis_names, resolution, n_levels) -> tuple[dict,dict]:
    '''
    GENERATES META-INFO FOR PYRAMID

    :param axis_names:
    :param resolution:
    :param n_levels:
    :return: list[dict,dict]
    '''

    transformations = []
    for scale_level in range(n_levels):
        scale = []
        for ax in axis_names:
            if ax in resolution:
                scale.append(resolution[ax] * 2**scale_level)
            else:
                scale.append(resolution.get(ax, 1))
        transformations.append([{"scale": scale, "type": "scale"}])
    return transformations


def create_omezarr(animal, downsample, debug):
    sqlController = SqlController(animal)
    fileLocationManager = FileLocationManager(animal)
    xy_resolution = sqlController.scan_run.resolution
    z_resolution = sqlController.scan_run.zresolution
    if downsample:
        scaling_factor = SCALING_FACTOR
        chunk_factor = 2
        INPUT = os.path.join(fileLocationManager.prep, 'C1', 'thumbnail_aligned')
        mips = 4
    else:
        scaling_factor = 1
        chunk_factor = 1
        INPUT = os.path.join(fileLocationManager.prep, 'C1', 'full_aligned')
        mips = 7
    if not os.path.exists(INPUT):
        print(f'Missing: {INPUT}')
        sys.exit()
    files = os.listdir(INPUT)
    if len(files) < 5:
        print(f'Not enough files in: {INPUT}')
        sys.exit()


    axes = [{'name': 'x', 'type': 'space', 'unit': 'micrometer'},
            {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
            {'name': 'z', 'type': 'space', 'unit': 'micrometer'}]
    axis_names = [a['name'] for a in axes]

    stacked = imreads(INPUT)

    # like numpy.mean, but maintains dtype
    def mean_dtype(arr, **kwargs):
        return np.mean(arr, **kwargs).astype(arr.dtype)

    chunk_dict = {
        1: [128//chunk_factor, 128//chunk_factor, 64//chunk_factor],
        2: [64//chunk_factor, 64//chunk_factor, 64//chunk_factor],
        3: [64//chunk_factor, 64//chunk_factor, 64//chunk_factor],
        4: [64//chunk_factor, 64//chunk_factor, 64//chunk_factor],
        5: [32//chunk_factor, 32//chunk_factor, 32//chunk_factor],
        6: [32//chunk_factor, 32//chunk_factor, 32//chunk_factor],
        7: [32//chunk_factor, 32//chunk_factor, 32//chunk_factor],
    }

    stacked = stacked.rechunk((128//chunk_factor, 128//chunk_factor, 64))
    downsampled_stack = [stacked]
    start_time = timer()
    for mip in (range(1, mips)):
        chunks = chunk_dict[mip]
        axis_dict = {0:2, 1:2, 2:2}
        print(f'scaled {mip} chunks={chunks} axis_dict={axis_dict}')
        scaled = da.coarsen(mean_dtype, downsampled_stack[-1], axis_dict, trim_excess=True).rechunk(chunks)
        downsampled_stack.append(scaled)
    end_time = timer()
    elapsed_time = round((end_time - start_time), 2)
    print(f'Creating downsampled mip took {elapsed_time} seconds')

    n_levels = len(downsampled_stack)
    resolution = {'x': xy_resolution*scaling_factor, 'y': xy_resolution*scaling_factor, 'z': z_resolution}
    transformations = get_transformations(axis_names, resolution, n_levels)
    if debug:
        print(f'transformations={transformations}')

    # Open the zarr group manually
    storefile = 'C1.zarr'
    storepath = os.path.join(fileLocationManager.www, 'neuroglancer_data', storefile)
    store = zarr.NestedDirectoryStore(storepath)
    root = zarr.group(store=store, overwrite=True)
    root.attrs['omero'] = {}
    if debug:
        print(root.info)

    # Use OME write multiscale; this actually computes the dask arrays but does so
    # in a memory-efficient way.
    write_multiscale(downsampled_stack, group=root, axes=axes, coordinate_transformations=transformations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=True)
    parser.add_argument("--debug", help="debug", required=False, default=False)
    parser.add_argument("--downsample", help="downsample", required=False, default=True)
    args = parser.parse_args()
    animal = args.animal
    downsample = bool({"true": True, "false": False}[str(args.downsample).lower()])
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    
    create_omezarr(animal, downsample, debug)
