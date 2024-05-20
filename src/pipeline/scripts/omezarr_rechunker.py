import argparse
import os
import sys
from pathlib import Path
from timeit import default_timer as timer

import psutil
import zarr
import dask
import dask.array as da
import numpy as np
from timeit import default_timer as timer
from distributed import Client, progress
from dask.diagnostics import ProgressBar
from pathlib import Path
import sys
import os
import zarr
from dask.distributed import Client
import argparse
import numpy as np
import dask.array as da
from timeit import default_timer as timer
from rechunker import rechunk

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
        tmp_dir = get_scratch_dir()
        self.tmp_dir = os.path.join(tmp_dir, f'{self.animal}')
        os.makedirs(self.tmp_dir, exist_ok=True)

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

        paths = [self.storepath, self.rechunkmepath]
        for path in paths:
            if os.path.exists(path):
                print(f'Using existing {path}')   
                store = get_store(path, 0, 'r')
                volume = zarr.open(store, 'r')
                print(volume.info)
                print(f'volume.shape={volume.shape}')

    @staticmethod
    def create_target_chunks(shape):
        """Create target chunks based on the shape of the array.

        Args:
            shape (tuple): Shape of the array.

        Returns:
            tuple: Target chunks.
        """
        # Define your logic to create target chunks here
        rows = shape[-2] // 4
        columns = shape[-1] // 4

        target_chunks = (1, 1, 1, rows, columns)
        return target_chunks


    def rechunkme(self):
    
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
        #target_chunks = self.create_target_chunks(rechunkme_stack.shape)
        print(f'Using existing store with old shape={rechunkme_stack.shape} chunks={rechunkme_stack.chunksize}', end=" ")
        GB = (psutil.virtual_memory().free // 1024**3) * 0.8
        max_mem = f"{GB}GB"        
        start_time = timer()
        rechunkme_stack = rechunkme_stack.rechunk('auto')
        rechunkme_stack = rechunkme_stack.reshape(1, 1, *rechunkme_stack.shape)
        rechunkme_stack = rechunkme_stack.rechunk('auto')
        print(f'New shape={rechunkme_stack.shape} new chunks={rechunkme_stack.chunksize}')
        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        print(f'Rechunking to chunk={rechunkme_stack.chunksize} took {total_elapsed_time} seconds.')
        
        store = get_store(self.storepath, 0)
        temp_store = os.path.join(self.tmp_dir, "rechunked-tmp.zarr")
        array_plan = rechunk(
            rechunkme_stack, rechunkme_stack.chunksize, max_mem, store, temp_store=temp_store
        )
        print(f'Executing plan with mem={max_mem}')
        start_time = timer()
        with ProgressBar():
            rechunked = array_plan.execute()        
        print('Plan executed')
        #rechunked = da.from_zarr(rechunked)
        store = get_store(self.storepath, 0)
        target_chunks = (1, 1, 1, 2048, 2048)
        workers = 2
        jobs = 4
        z = zarr.zeros(rechunked.shape, chunks=target_chunks, store=store, overwrite=True, dtype=self.dtype)
        with Client(n_workers=workers, threads_per_worker=jobs) as client:
            print(f'Writing to zarr with workers={workers} jobs={jobs} target_chunks={target_chunks} dtype={self.dtype}')
            to_store = da.store(rechunked, z, lock=False, compute=False)
            to_store = progress(client.compute(to_store))
            to_store = client.gather(to_store)

        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        print(f'Wrote rechunked data to: {write_storepath} took {total_elapsed_time} seconds.')




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
