import argparse
import os
import sys
from pathlib import Path
from timeit import default_timer as timer

import zarr

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

    def info(self):
        if self.downsample:
            filename = f'C{self.channel}T.zarr'
        else:
            filename = f'C{self.channel}.zarr'

        zarrpath = os.path.join(self.fileLocationManager.neuroglancer_data, filename)
        if os.path.exists(zarrpath):
            print(f'Using existing {zarrpath}')
        else:
            print(f'No zarr: {zarrpath}')
            return
        
        store = get_store(zarrpath, 0, 'r')
        volume = zarr.open(store, 'r')
        print(volume.info)
        print(f'volume.shape={volume.shape}')




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
    }

    if task in function_mapping:
        function_mapping[task]()
    else:
        print(f"{task} is not a correct task. Choose one of these:")
        for key in function_mapping.keys():
            print(f"\t{key}")
