import argparse
import os
import sys
from pathlib import Path
from timeit import default_timer as timer
from datetime import datetime

import zarr
import ome_zarr
from ome_zarr.writer import write_multiscale
import zarr
import dask.array as da
import numpy as np

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.utilities.utilities_process import SCALING_FACTOR
from library.utilities.dask_utilities import aligned_coarse_chunks, get_transformations, mean_dtype, write_mip_series
from library.controller.sql_controller import SqlController
from library.image_manipulation.filelocation_manager import FileLocationManager


def load_existing_zarr():
    fileLocationManager = FileLocationManager(animal)
    scaling_factor = 10
    z_resolution *= scaling_factor
    zarrfile = f'C1.scale.10.zarr'
    zarrpath = os.path.join(fileLocationManager.www, 'neuroglancer_data', zarrfile)
    if not os.path.exists(zarrpath):
        print(f'Missing: {zarrpath}')
        sys.exit()
    
    return da.from_zarr(url=zarrpath)

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

def create_omezarr(animal, downsample, debug):
    sqlController = SqlController(animal)
    fileLocationManager = FileLocationManager(animal)
    xy_resolution = sqlController.scan_run.resolution
    z_resolution = sqlController.scan_run.zresolution
    if downsample:
        storefile = 'C1T.zarr'
        scaling_factor = SCALING_FACTOR
        INPUT = os.path.join(fileLocationManager.prep, 'C1', 'thumbnail_aligned')
        mips = [0,1,2,3]
    else:
        storefile = 'C1.zarr'
        scaling_factor = 1
        INPUT = os.path.join(fileLocationManager.prep, 'C1', 'full_aligned')
        mips = [0,1,2,3,4]
    if not os.path.exists(INPUT):
        print(f'Missing: {INPUT}')
        sys.exit()
    files = os.listdir(INPUT)
    if len(files) < 5:
        print(f'Not enough files in: {INPUT}')
        sys.exit()
    # Open the zarr group manually
    storepath = os.path.join(fileLocationManager.www, 'neuroglancer_data', storefile)

    axes = [
        {
            "name": "x",
            "type": "space",
            "unit": "micrometer",
            "coarsen": 2,
            "resolution": xy_resolution * scaling_factor,
        },
        {
            "name": "y",
            "type": "space",
            "unit": "micrometer",
            "coarsen": 2,
            "resolution": xy_resolution * scaling_factor,
        },
        {
            "name": "z",
            "type": "space",
            "unit": "micrometer",
            "coarsen": 1,
            "resolution": z_resolution,
        }
    ]
    axis_scales = [a["coarsen"] for a in axes]
    #stacked = imreads(INPUT)
    #stacked = load_stack(INPUT)
    write_mip_series(INPUT, storepath)
    #stacked = np.swapaxes(stacked, 0,2)
    #print(f'Shape of stacked: {stacked.shape} type={type(stacked)} chunk size={stacked.chunksize}')
    print('Fini')
    return
    
    start_time = timer()
    downscale_start_time = timer()
    old_shape = stacked.shape
    trimto = 8
    new_shape = aligned_coarse_chunks(old_shape, trimto)
    print('new shape', new_shape)
    stacked = stacked[0:new_shape[0], 0:new_shape[1], :]
    oldchunk_size = stacked.chunksize
    #stacked = stacked.rechunk('auto')
    axis_dict = {0:axis_scales[0], 1:axis_scales[1], 2:axis_scales[2]}
    if debug:
        print(f'stacked original chunksize={oldchunk_size}')
        print(f'Shape of downsampled stacked is now: {stacked.shape}')
        print('stacked.chunksize', stacked.chunksize, type(stacked))
        print(f'axis dict={axis_dict}')
    downsampled_stack = [stacked]
    for mip in mips:
        scaled = da.coarsen(mean_dtype, downsampled_stack[-1], axis_dict, trim_excess=True)
        print(f'scaled {mip} chunks={scaled.chunksize} shape={scaled.shape} type={type(stacked)}')
        downsampled_stack.append(scaled)

    downscale_end_time = timer()
    downscale_elapsed_time = round((downscale_end_time - downscale_start_time), 2)
    print(f'Downsampling {len(downsampled_stack)} stacks took {downscale_elapsed_time} seconds')

    n_levels = len(downsampled_stack)
    transformations = get_transformations(axes, n_levels)
    storage_opts = {'chunks': [64,64,64]}
    
    meta_data = get_meta_data(
        transformations=transformations, 
        axes=axes,
        transformation_method='mean',
        description=f'Image stack for {animal} to OME Zarr',
        perf_lab='UCSD')

    if debug:
        for t in transformations:
            print(f'transformation={t}')
        print(f'storage opts={storage_opts}')
        print(f'metadata={meta_data}')


    root = zarr.group(store=store, overwrite=True)
    root.attrs['omero'] = {}
        
    write_start_time = timer()
    write_multiscale(pyramid=downsampled_stack, 
                     group=root, 
                     axes=axes, 
                     storage_options=storage_opts,
                     coordinate_transformations=transformations,
                     metadata=meta_data, 
                     compute=True)
    write_end_time = timer()
    write_elapsed_time = round((write_end_time - write_start_time), 2)
    print(f'Writing {len(downsampled_stack)} stacks took {write_elapsed_time} seconds')

    total_elapsed_time = round((write_end_time - start_time), 2)
    print(f'Total time using coarsen took {total_elapsed_time} seconds')

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
