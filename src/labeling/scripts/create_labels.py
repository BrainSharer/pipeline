"""This program works with the cell detection process.
Currently, the program works with one animal and one task at a time.
All models are stored in /net/birdstore/Active_Atlas_Data/cell_segmentation/models/
The models are named by step: models_step_X_threshold_2000.pkl
where 'X' is the step number.
The program can be run with the following commands:

- python srs/labeling/scripts/create_labels.py --animal DKXX --task create_features
- python srs/labeling/scripts/create_labels.py --animal DKXX --task detect
- python srs/labeling/scripts/create_labels.py --animal DKXX --task extract
- python srs/labeling/scripts/create_labels.py --animal DKXX --task neuroglancer
- python srs/labeling/scripts/create_labels.py --animal DKXX --task train
- python srs/labeling/scripts/create_labels.py --animal DKXX --task fix


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

Training workflow::
- The supervised models used require manual steps for processing (see )



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

from library.cell_labeling.cell_manager import CellMaker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)
    parser.add_argument("--model", help="Enter the model", required=False, type=str)
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)
    parser.add_argument("--annotation", help="Create features with specific annotation id", required=False, default="", type=str)
    parser.add_argument("--step", help="Enter step", required=False, type=int)
    parser.add_argument("--x", help="Enter x", required=False, default=0, type=int)
    parser.add_argument("--y", help="Enter y", required=False, default=0, type=int)
    parser.add_argument("--sampling", help="Random sampling qty", required=False, default=0, type=int)

    #SEGMENTATION ARGUMENTS [USED WITH 'detect' TASK]
    parser.add_argument("--kernel", help="Kernel size for Gaussian blurring [SEGMENTATION]", required=False, default=401, type=int)
    parser.add_argument("--sigma", help="Sigma for Gaussian blurring [SEGMENTATION]", required=False, default=350, type=int)
    parser.add_argument("--min-segment", help="Minimum segmentation size (pixels)", required=False, default=100, type=int)
    parser.add_argument("--max-segment", help="Maximum segmentation size (pixels)", required=False, default=100000, type=int)
    parser.add_argument("--segment-threshold", help="Intensity threshold (0 to 65535) [SEGMENTATION]", required=False, default=2000, type=int)
    parser.add_argument("--cell-radius", help="cell radius (pixels) [SEGMENTATION]", required=False, default=40, type=int)

    #NEUROGLANCER ARGUMENTS [USED WITH 'neuroglancer' TASK]
    #Note: ng-id should be id of full-resolution (all channels); this task will add CH3_DIFF and ML_POSITIVE point detections
    parser.add_argument("--ng-id", help="Brainsharer id", required=False, default="", type=str)

    parser.add_argument(
        "--task",
        help="Enter the task you want to perform: create_features|detect|extract|neuroglancer|train",
        required=True,
        type=str,
    )

    args = parser.parse_args()

    animal = args.animal
    model = args.model
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    task = str(args.task).strip().lower()
    step = args.step
    x = args.x
    y = args.y
    annotation_id = args.annotation
    sampling = args.sampling #Note: only used in conjunction with 'extract' task
    segment_size_min = args.min_segment
    segment_size_max = args.max_segment
    segment_gaussian_sigma = args.sigma
    segment_gaussian_kernel = args.kernel
    segment_threshold = args.segment_threshold
    cell_radius = args.cell_radius
    ng_id = args.ng_id

    pipeline = CellMaker(animal=animal, task=task, step=step, model=model, channel=1, x=x, y=y, annotation_id=annotation_id, sampling=sampling, segment_size_min=segment_size_min, segment_size_max=segment_size_max, segment_gaussian_sigma=segment_gaussian_sigma, segment_gaussian_kernel=segment_gaussian_kernel, segment_threshold=segment_threshold, cell_radius=cell_radius, ng_id=ng_id, debug=debug)

    function_mapping = {
        "create_features": pipeline.create_features,
        "detect": pipeline.create_detections,
        "segment": pipeline.segment,
        "extract": pipeline.extract_predictions_precomputed,        
        "train": pipeline.train,
        "fix": pipeline.fix_coordinates
        # "neuroglancer": pipeline.neuroglancer, #TODO: remove
        # "omezarr": pipeline.omezarr, #TODO: remove
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
