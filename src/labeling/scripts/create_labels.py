"""This program works with the cell detection process.
Currently, the program works with one animal and one task at a time.
All models are stored in /net/birdstore/Active_Atlas_Data/cell_segmentation/models/
The models are named by step: models_step_X_threshold_2000.pkl
where 'X' is the step number.
The program can be run with the following commands:

- python sr/labeling/scripts/create_labels.py --animal DKXX --task detect
- python sr/labeling/scripts/create_labels.py --animal DKXX --task extract
- python sr/labeling/scripts/create_labels.py --animal DKXX --task train
- python sr/labeling/scripts/create_labels.py --animal DKXX --task fix

Explanation for the tasks:

- detect - This is the 1st task to run and will create the cell_labels/detection_XXX.csv files. \
    This task will run the cell detection model on the images and create the detections.
- extract - This task will extract the predictions from the detection files and create the \
    cell_labels/all_predictions.csv file.
- train - This task creates the detection_XXX.csv files created above and trains the model. \
    The features are taken from the detection_XXX.csv files. \
    The model is saved in the cell_segmentation/models dir. \
    This new model can then be used to rerun the detection process. Repeat as necessary.
- fix - This is only needed when the images have the extra tissue and skull present. \
    You will need to create the rotated and aligned masks for the images.

Plan for implementation:

- Detect cells on available brains.
- Some of the brains have too many points to easily display in Neuroglancer. DK59 has about 75MB of points. \
    This won't display and will crash the browser. We can take the points and display them as a precomputed \
    data format, similar to the way we display large images.
- Once we have the display of the predicted points along with the image stacks of the dye and the virus channels, \
    we can create two more layers. A 'bad' layer where the user marks as 'bad' the predictions that are bad. \
    And another layer 'sure' where the user creates annotations that the prediction process has missed.
- These 'bad' and 'sure' new annotations are then saved to the database.
- We then create features from these 'bad' and 'sure' coordinates.
- These features are then fed back into the training process and a new model is created which we then use \
    to repeat the process.
"""

import argparse
from pathlib import Path
import sys
from timeit import default_timer as timer


PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

#from library.cell_labeling.cell_pipeline import CellPipeline
from library.cell_labeling.cell_manager import CellMaker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)
    parser.add_argument("--step", help="Enter step", required=False, default=4, type=int)
    parser.add_argument(
        "--task",
        help="Enter the task you want to perform: detect|extract_predictions|train",
        required=True,
        type=str,
    )

    args = parser.parse_args()

    animal = args.animal
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    task = str(args.task).strip().lower()
    step = args.step

    pipeline = CellMaker(animal=animal, task=task, step=step, debug=debug)
    
    function_mapping = {
        "create_features": pipeline.create_features,
        "detect": pipeline.create_detections,
        "extract": pipeline.extract_predictions,
        "train": pipeline.train,
        "fix": pipeline.fix_coordinates,
        "precomputed": pipeline.create_precomputed_annotations
    }
    

    if task in function_mapping:
        start_time = timer()
        function_mapping[task]()
        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        if total_elapsed_time >= 3600:
            hours = total_elapsed_time // 3600
            minutes = (total_elapsed_time % 3600) // 60
            time_out_msg = f'took {int(hours)} hour(s) and {int(minutes)} minute(s).'
        else:
            time_out_msg = f'took {total_elapsed_time} seconds.'

        print(f"{task} {time_out_msg}")
    else:
        print(f"{task} is not a correct task. Choose one of these:")
        for key in function_mapping.keys():
            print(f"\t{key}")
