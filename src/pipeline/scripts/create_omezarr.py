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
import xarray as xr
from datetime import datetime, timedelta
PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.utilities.utilities_process import SCALING_FACTOR
from library.utilities.dask_utilities import get_transformations, imreads, mean_dtype
from library.controller.sql_controller import SqlController
from library.image_manipulation.filelocation_manager import FileLocationManager

def sizeof_fmt(num, suffix="B"):
    #ref: https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def get_meta_data(axes, transformations, transformation_method: str, description, perf_lab):
    # META-DATA COMPILATION ('provenance' FROM precomputed FORMAT)

    #TODO: write_multiscales_metadata() from ome_zarr?
    meta_datasets = {'coordinateTransformations': transformations}

    now = datetime.now()  # datetime object containing current date and time
    dt_string = now.strftime("%m-%d-%Y %H:%M:%S")
    processing_meta = {'description': description, 'method': transformation_method, 'owners': perf_lab,
            'processing': {'date': dt_string}, 'kwargs': {'multichannel': True}}

    meta_multiscales = {'axes': axes, 'datasets': meta_datasets, 'metadata': processing_meta}
    return meta_multiscales

def get_storage_opts(axis_names: tuple[str]) -> dict:
    '''
    CALCULATES OPTIMAL CHUNK SIZE FOR IMAGE STACK (TARGET IS ~25MB EACH)
    N.B. CHUNK DIMENSION ORDER (XYZ) SHOULD CORRESPOND TO DASK DIMENSION ORDER (XYZ)
    
    ref: https://forum.image.sc/t/deciding-on-optimal-chunk-size/63023/7

    :param axis_names: tuple[str]
    :return: dict
    '''

    z_section_chunk = 20
    byte_per_pixel = 2
    target_chunk_size_mb = 25
    chunk_dim = (target_chunk_size_mb*10**6 / byte_per_pixel / z_section_chunk)**(1/2) #1MB / BYTES PER PIXEL / kui_constant, SPLIT (SQUARE ROOT) BETWEEN LAST 2 DIMENSIONS
    
    if len(axis_names) > 2:
        new_chunks = (int(chunk_dim), int(chunk_dim), z_section_chunk)
        print(F'EACH CHUNK MEM SIZE: {sizeof_fmt(new_chunks[0] * new_chunks[1] * new_chunks[2] * byte_per_pixel)}')
    else:
        new_chunks = (int(chunk_dim), int(chunk_dim))    
        print(F'EACH CHUNK MEM SIZE: {sizeof_fmt(new_chunks[0] * new_chunks[1] * byte_per_pixel)}')

    return {"chunks": new_chunks}

def create_omezarr(animal, downsample, debug):
    sqlController = SqlController(animal)
    fileLocationManager = FileLocationManager(animal)
    xy_resolution = sqlController.scan_run.resolution
    z_resolution = sqlController.scan_run.zresolution
    if downsample:
        storefile = 'C1T.zarr'
        scaling_factor = SCALING_FACTOR
        chunk_factor = 2
        INPUT = os.path.join(fileLocationManager.prep, 'C1', 'thumbnail_aligned')
        mips = 4
    else:
        storefile = 'C1.zarr'
        scaling_factor = 1
        chunk_factor = 1
        INPUT = os.path.join(fileLocationManager.prep, 'C1', 'full_aligned')
        mips = 4
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
    """
    scaling_factor = 10
    z_resolution *= scaling_factor
    zarrfile = f'C1.scale.10.zarr'
    zarrpath = os.path.join(fileLocationManager.www, 'neuroglancer_data', zarrfile)
    if not os.path.exists(zarrpath):
        print(f'Missing: {zarrpath}')
        sys.exit()
    
    stacked = da.from_zarr(url=zarrpath)
    stacked = np.swapaxes(stacked, 0,2)
    """


    print(f'Shape of stacked is now: {stacked.shape} type={type(stacked)}')


    chunk_dict = {
        1: [128//chunk_factor, 128//chunk_factor, 64//chunk_factor],
        2: [64//chunk_factor, 64//chunk_factor, 64//chunk_factor],
        3: [64//chunk_factor, 64//chunk_factor, 64//chunk_factor],
        4: [64//chunk_factor, 64//chunk_factor, 64//chunk_factor],
        5: [32//chunk_factor, 32//chunk_factor, 32//chunk_factor],
        6: [32//chunk_factor, 32//chunk_factor, 32//chunk_factor],
        7: [32//chunk_factor, 32//chunk_factor, 32//chunk_factor],
    }
    start_time = timer()
    #stacked = stacked.rechunk((128//chunk_factor, 128//chunk_factor, 64))
    stacked = stacked.rechunk('auto')
    if debug:
        print(f'Shape of downsampled stacked is now: {stacked.shape}')
        print('stacked.chunksize')
        print(stacked.chunksize)

    downsampled_stack = [stacked]
    rechunks = [] # better chunk size to serve via neuroglancer
    for mip in (range(1, mips)):
        axis_dict = {0:2, 1:2, 2:2}
        chunks = chunk_dict[mip]
        scaled = da.coarsen(mean_dtype, downsampled_stack[-1], axis_dict, trim_excess=True).rechunk('auto')
        chunks = scaled.chunksize
        rechunk = chunk_dict[mip]
        rechunks.append(rechunk)
        if debug:
            print(f'scaled {mip} chunks={chunks} axis_dict={axis_dict}')
        downsampled_stack.append(scaled)

    n_levels = len(downsampled_stack)
    resolution = {'x': xy_resolution*scaling_factor, 'y': xy_resolution*scaling_factor, 'z': z_resolution}
    transformations = get_transformations(axis_names, resolution, n_levels)
    storage_opts = get_storage_opts(axis_names=axis_names)
    #storage_opts = {'chunks': [512, 512, 20]}
    meta_data = get_meta_data(
        transformations=transformations, 
        axes=axes, 
        transformation_method='mean',
        description='Image stack to OME Zarr',
        perf_lab='UCSD')

    if debug:
        print(f'transformations={transformations}')
        print(f'storage opts={storage_opts}')
        print(f'meta data={meta_data}')

    # Open the zarr group manually
    storepath = os.path.join(fileLocationManager.www, 'neuroglancer_data', storefile)
    store = zarr.NestedDirectoryStore(storepath)
    root = zarr.group(store=store, overwrite=True)
    root.attrs['omero'] = {}
    if debug:
        print(root.info)

    # Use OME write multiscale; this actually computes the dask arrays but does so
    # in a memory-efficient way.
    write_start_time = timer()
    write_multiscale(pyramid=downsampled_stack, 
                     group=root, 
                     storage_options=storage_opts, 
                     axes=axes, 
                     coordinate_transformations=transformations,
                     metadata=meta_data)
    write_end_time = timer()
    write_elapsed_time = round((write_end_time - write_start_time), 2)
    print(f'Writing {len(downsampled_stack)} stacks took {write_elapsed_time} seconds')

    total_elapsed_time = round((write_end_time - start_time), 2)
    print(f'Total time took {total_elapsed_time} seconds')


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
    #create_pyramid(animal, downsample, debug)