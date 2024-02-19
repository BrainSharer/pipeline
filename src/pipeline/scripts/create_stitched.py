import argparse
from pathlib import Path
import sys, socket
from timeit import default_timer as timer

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.utilities.brain_stitcher import BrainStitcher


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)
    args = parser.parse_args()
    animal = args.animal
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])

    pipeline = BrainStitcher(animal,debug=debug)
    pipeline.create_channel_volume_from_h5()
