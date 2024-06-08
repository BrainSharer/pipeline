import argparse
from pathlib import Path
import os
import sys
from scipy.ndimage import zoom


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
        if os.path.exists(outpath):
            print(f"Skipping {file}")
            continue
        else:
            scaleme = read_image(inpath)
            scaled = zoom(scaleme, (change))
            write_image(scaled, outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)
    parser.add_argument("--channel", help="Enter 1, 2, 3, 4", required=False, default=1, type=int)
    args = parser.parse_args()
    animal = args.animal
    channel = args.channel
    resize_images(animal, channel)

