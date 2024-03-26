import argparse
import os
import sys
from pathlib import Path
from timeit import default_timer as timer
from datetime import datetime

import zarr
from ome_zarr.writer import write_multiscale
import zarr
import dask.array as da

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.utilities.utilities_process import SCALING_FACTOR
from library.utilities.dask_utilities import get_transformations, imreads, mean_dtype
from library.controller.sql_controller import SqlController
from library.image_manipulation.filelocation_manager import FileLocationManager


def get_meta_data(axes, transformations, transformation_method: str, description, perf_lab):
    # META-DATA COMPILATION ('provenance' FROM precomputed FORMAT)

    #TODO: write_multiscales_metadata() from ome_zarr?
    meta_datasets = {'coordinateTransformations': transformations}

    now = datetime.now()  # datetime object containing current date and time
    dt_string = now.strftime("%m-%d-%Y %H:%M:%S")
    processing_meta = {'description': description, 'method': transformation_method, 'owners': perf_lab,
            'processing': {'date': dt_string}, 'kwargs': {'multichannel': False}}

    meta_multiscales = {'axes': axes, 'datasets': meta_datasets, 'metadata': processing_meta}
    return meta_multiscales

def get_xy_chunk() -> int:
    '''
    CALCULATES OPTIMAL CHUNK SIZE FOR IMAGE STACK (TARGET IS ~25MB EACH)
    N.B. CHUNK DIMENSION ORDER (XYZ) SHOULD CORRESPOND TO DASK DIMENSION ORDER (XYZ)
    
    ref: https://forum.image.sc/t/deciding-on-optimal-chunk-size/63023/7

    :return: int: xy chunk
    '''

    z_section_chunk = 20
    byte_per_pixel = 2
    target_chunk_size_mb = 25
    xy_chunk = (target_chunk_size_mb*10**6 / byte_per_pixel / z_section_chunk)**(1/2) #1MB / BYTES PER PIXEL / kui_constant, SPLIT (SQUARE ROOT) BETWEEN LAST 2 DIMENSIONS

    return int(xy_chunk)

def create_omezarr(animal, downsample, debug):
    sqlController = SqlController(animal)
    fileLocationManager = FileLocationManager(animal)
    xy_resolution = sqlController.scan_run.resolution
    z_resolution = sqlController.scan_run.zresolution
    if downsample:
        storefile = 'C1T.zarr'
        scaling_factor = SCALING_FACTOR
        chunks = [64, 64, 64]
        INPUT = os.path.join(fileLocationManager.prep, 'C1', 'thumbnail_aligned')
        mips = 4
    else:
        storefile = 'C1.zarr'
        scaling_factor = 1
        chunks = [128, 128, 64]
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

    
    start_time = timer()
    #stacked = stacked.rechunk((128//chunk_factor, 128//chunk_factor, 64))
    stacked = stacked.rechunk('auto')
    if debug:
        print(f'Shape of downsampled stacked is now: {stacked.shape}')
        print('stacked.chunksize')
        print(stacked.chunksize)

    downsampled_stack = [stacked]
    for mip in (range(1, mips)):
        axis_dict = {0:2, 1:2, 2:2}
        scaled = da.coarsen(mean_dtype, downsampled_stack[-1], axis_dict, trim_excess=True).rechunk('auto')
        if debug:
            print(f'scaled {mip} chunks={scaled.chunksize} axis_dict={axis_dict}')
        downsampled_stack.append(scaled)

    n_levels = len(downsampled_stack)
    resolution = {'x': xy_resolution*scaling_factor, 'y': xy_resolution*scaling_factor, 'z': z_resolution}
    transformations = get_transformations(axis_names, resolution, n_levels)
    storage_opts = {'chunks': chunks}
    meta_data = get_meta_data(
        transformations=transformations, 
        axes=axes, 
        transformation_method='mean',
        description=f'Image stack for {animal} to OME Zarr',
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
