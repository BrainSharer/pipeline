"""This program deals with mask training and prediction.

- python src/pipeline/scripts/create_masks.py --animal DKXX --task 

Explanation for the tasks:

"""

import argparse
from pathlib import Path
import sys

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())
from masking.scripts.structure_predictor import MaskPrediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on masking")
    parser.add_argument('--abbreviation', help='Enter structure abbreviation', required=False, default=None, type=str)
    parser.add_argument('--animal', help='animal', required=False, default=None, type=str)
    parser.add_argument('--epochs', help='# of epochs', required=False, default=2, type=int)
    parser.add_argument("--task", help="Enter the task you want to perform", required=True, type=str)
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)

    args = parser.parse_args()
    abbreviation = args.abbreviation
    debug = bool({'true': True, 'false': False}[args.debug.lower()])
    animal = args.animal
    epochs = args.epochs
    task = str(args.task).strip().lower()
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    mask_predictor = MaskPrediction(animal, abbreviation, epochs, debug)

    function_mapping = {
        "train": mask_predictor.mask_trainer,
        "predict": mask_predictor.predict_masks,
        "update_session": mask_predictor.update_session
    }

    if task in function_mapping:
        function_mapping[task]()

    else:
        print(f"{task} is not a correct task. Choose one of these:")
        for key in function_mapping.keys():
            print(f"\t{key}")
