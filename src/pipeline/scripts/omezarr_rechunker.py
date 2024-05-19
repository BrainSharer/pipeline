import argparse
import os
import sys
from pathlib import Path
from timeit import default_timer as timer

import zarr
import dask
import dask.array as da
import numpy as np
from timeit import default_timer as timer
from distributed import Client, progress
from dask.diagnostics import ProgressBar

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.image_manipulation.image_manager import ImageManager
from library.utilities.utilities_process import get_scratch_dir

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.image_manipulation.filelocation_manager import FileLocationManager
from library.utilities.dask_utilities import get_store


class OmezarrManager():

    def __init__(self, animal, channel, downsample, debug):
        """Initiates the brain object

        Args:
            animal (string): Animal ID
        """
        self.animal = animal
        self.channel = channel
        self.downsample = downsample
        self.debug = debug
        self.fileLocationManager = FileLocationManager(animal)
        if self.downsample:
            self.storefile = 'C1T.zarr'
            self.rechunkmefile = 'C1T_rechunk.zarr'
            self.input = self.fileLocationManager.get_thumbnail_aligned(self.channel)
        else:
            self.storefile = 'C1.zarr'
            self.rechunkmefile = 'C1_rechunk.zarr'
            self.input = self.fileLocationManager.get_full_aligned(self.channel)

        self.storepath = os.path.join(
            self.fileLocationManager.www, "neuroglancer_data", self.storefile
        )
        self.rechunkmepath = os.path.join(
            self.fileLocationManager.www, "neuroglancer_data", self.rechunkmefile
        )

        image_manager = ImageManager(self.input)
        self.dtype = image_manager.dtype

    def info(self):

        if os.path.exists(os.path.join(self.storepath, 'scale0')):
            print(f'Using existing {self.storepath}')
        else:
            print(f'No zarr: {self.storepath}')
            return
        
        store = get_store(self.storepath, 0, 'r')
        volume = zarr.open(store, 'r')
        print(volume.info)
        print(f'volume.shape={volume.shape}')

    def rechunkme(self):
        with Client(n_workers=2, threads_per_worker=4) as client:
    
            if os.path.exists(os.path.join(self.storepath, 'scale0')):
                print('Rechunked store exists, no need to rechunk full resolution.\n')
                return

            read_storepath = os.path.join(self.rechunkmepath, 'scale0')
            write_storepath = os.path.join(self.storepath, 'scale0')
            print(f'Loading data at: {read_storepath}', end=" ")
            if os.path.exists(read_storepath):
                print(': Success!')
            else:
                print('\nError: exiting ...')
                print(f'Missing {read_storepath}')            
                sys.exit()

            if os.path.exists(write_storepath):
                print(f'Already exists: {write_storepath}')            
                return

            rechunkme_stack = da.from_zarr(url=read_storepath)
            print(f'Using rechunking store with shape={rechunkme_stack.shape} chunks={rechunkme_stack.chunksize}')
            leading_chunk = rechunkme_stack.shape[2]
            target_chunks = (1, 1, 1, rechunkme_stack.shape[1]//4, rechunkme_stack.shape[2]//4)
            if leading_chunk < target_chunks[2]:
                target_chunks = (1, 1, leading_chunk, rechunkme_stack.shape[1]//4, rechunkme_stack.shape[2]//4)
            start_time = timer()
            rechunked = rechunkme_stack.rechunk(target_chunks)
            end_time = timer()
            total_elapsed_time = round((end_time - start_time), 2)
            print(f'Rechunking to chunk={rechunked.chunksize} took {total_elapsed_time} seconds.')
            del rechunkme_stack 
            store = get_store(self.storepath, 0)
            z = zarr.zeros(rechunked.shape, chunks=target_chunks, store=store, overwrite=True, dtype=self.dtype)
            to_store = da.store(rechunked, z, lock=False, compute=False)

            start_time = timer()
            print(f'Storing type={type(to_store)} rechunked data to: {write_storepath}')
            to_store = progress(client.compute(to_store))
            end_time = timer()
            total_elapsed_time = round((end_time - start_time), 2)
            print(f'Wrote rechunked data to: {write_storepath} took {total_elapsed_time} seconds.')
            print()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=True)
    parser.add_argument("--debug", help="debug", required=False, default=False)
    parser.add_argument("--info", help="info on zarr store", required=False, default=False)
    parser.add_argument("--channel", help="Enter a channel number", required=False, default=1, type=int)
    parser.add_argument("--downsample", help="downsample", required=False, default=True)
    parser.add_argument(
        "--task",
        help="Enter the task you want to perform: extract|stitch|move|status",
        required=False,
        default="status",
        type=str,
    )
    args = parser.parse_args()
    animal = args.animal
    channel = args.channel
    info = bool({"true": True, "false": False}[str(args.info).lower()])
    downsample = bool({"true": True, "false": False}[str(args.downsample).lower()])
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    task = str(args.task).strip().lower()
    
    pipeline = OmezarrManager(animal, channel, downsample, debug)


    function_mapping = {
        "info": pipeline.info,
        "rechunkme": pipeline.rechunkme,
    }

    if task in function_mapping:
        function_mapping[task]()
    else:
        print(f"{task} is not a correct task. Choose one of these:")
        for key in function_mapping.keys():
            print(f"\t{key}")
