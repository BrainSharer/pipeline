import argparse
from pathlib import Path
import sys

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.utilities.brain_stitcher import BrainStitcher


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=False, default="DK20230126-003", type=str)
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)
    parser.add_argument("--channel", help="Enter 1, 2, or 4", required=False, default=1, type=int)
    parser.add_argument("--layer", help="Enter layer", required=False, default=1, type=int)
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
    layer = args.layer
    task = str(args.task).strip().lower()
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])

    pipeline = BrainStitcher(animal, layer, channel, debug=False)

    function_mapping = {
        "extract": pipeline.extract,
        "stitch": pipeline.stitch_tile,
        "status": pipeline.check_status,
        "move": pipeline.move_data
    }

    if task in function_mapping:
        function_mapping[task]()
    else:
        print(f"{task} is not a correct task. Choose one of these:")
        for key in function_mapping.keys():
            print(f"\t{key}")
