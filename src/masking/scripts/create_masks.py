import argparse
from pathlib import Path
import sys

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())
from masking.scripts.structure_predictor import MaskPrediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on masking")
    parser.add_argument('--structures', help='true for structures', required=False, default='true', type=str)
    parser.add_argument('--animal', help='animal', required=False, default='MD589', type=str)
    parser.add_argument('--epochs', help='# of epochs', required=False, default=2, type=int)
    parser.add_argument('--num_classes', help='# of structures', required=False, default=2, type=int)
    parser.add_argument("--task", help="Enter the task you want to perform", required=True, type=str)
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)

    args = parser.parse_args()
    structures = bool({'true': True, 'false': False}[args.structures.lower()])
    debug = bool({'true': True, 'false': False}[args.debug.lower()])
    animal = args.animal
    epochs = args.epochs
    num_classes = args.num_classes
    task = str(args.task).strip().lower()
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    mask_predictor = MaskPrediction(animal, structures, num_classes, epochs, debug)

    function_mapping = {
        "train": mask_predictor.mask_trainer,
        "predict": mask_predictor.predict_masks,
        "json": mask_predictor.create_json_masks
    }

    if task in function_mapping:
        function_mapping[task]()

    else:
        print(f"{task} is not a correct task. Choose one of these:")
        for key in function_mapping.keys():
            print(f"\t{key}")
