import argparse
from pathlib import Path
import os
import sys
from scipy.ndimage import zoom
from tifffile import imread


PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.image_manipulation.filelocation_manager import FileLocationManager
from library.utilities.utilities_process import read_image, write_image


def resize_images(animal, channel):
    fileLocationManager = FileLocationManager(animal)
    input_path = fileLocationManager.get_full(channel=channel)
    output_path = fileLocationManager.get_thumbnail(channel=channel)
    files = sorted(os.listdir(input_path))
    change = 0.03125
    for file in files:
        inpath = os.path.join(input_path, file)
        outpath = os.path.join(output_path, file)
        if not os.path.exists(outpath):
            scaleme = imread(inpath)

            print(f'Processing {file} with shape {scaleme.shape} and dtype {scaleme.dtype}.', end="")
            scaled = zoom(scaleme, (change))
            write_image(outpath, scaled)
            print(f'Processed {file} with shape {scaleme.shape} and dtype {scaleme.dtype}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)
    parser.add_argument("--channel", help="Enter 1, 2, 3, 4", required=False, default=1, type=int)
    args = parser.parse_args()
    animal = args.animal
    channel = args.channel
    resize_images(animal, channel)

