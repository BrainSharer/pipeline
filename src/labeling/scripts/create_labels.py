"""This program works with the cell detection process.
Currently, the program works with one animal and one task at a time.
All models are stored in /net/birdstore/Active_Atlas_Data/cell_segmentation/models/
The models are named by step: models_step_X_threshold_2000.pkl
where 'X' is the step number.
The program can be run with the following commands:

- python srs/labeling/scripts/create_labels.py --animal DKXX --task segment
- python srs/labeling/scripts/create_labels.py --animal DKXX --task detect
- python srs/labeling/scripts/create_labels.py --animal DKXX --task extract
- python srs/labeling/scripts/create_labels.py --animal DKXX --task train
- python srs/labeling/scripts/create_labels.py --animal DKXX --task ng_preview

Explanation for the tasks:

- segment - Uses parameters and gaussian blurring to create the initial cell segmentation.
  If channel argument is provided, that channel will be used for the segmentation
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
- Some of the brains have too many points to easily display in Neuroglancer as annotation layers
    and will crash the browser. We create them as a precomputed \
    data format, which can be loaded in Neuroglancer as segmentation layer
    [this is included when running task:detect]

@end of 'detect' task, artifacts:
[on birdstore]:
preps/DIFF_{uuid} #not removed
www/histogram/DIFF_{uuid} #not removed
www/neuroglancer/DIFF_{uuid} #not removed
www/neuroglancer_data/progress/DIFF_{uuid} #removed

[on compute server] #all removed
/scratch/pipeline_tmp/{animal}/cell_candidates_{uuid}
/scratch/pipeline_tmp/{animal}/cell_features_{uuid}
/scratch/pipeline_tmp/{animal}/DIFF_{uuid}
/scratch/pipeline_tmp/{animal}/DIFF_candidates_{uuid}_py
"""

import argparse
from pathlib import Path
import sys
from timeit import default_timer as timer


PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.cell_labeling.cell_process import CellMaker

class RangeOrIntAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        s = values.strip()
        
        # Case 1: Range (e.g., "8081:8087" or "[8081:8087]")
        if ":" in s:
            if s.startswith('[') and s.endswith(']'):
                s = s[1:-1]
            parts = s.split(':')
            if len(parts) != 2:
                raise argparse.ArgumentError(self, f"Range must be two numbers separated by ':', got '{s}'")
            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError:
                raise argparse.ArgumentError(self, "Start and end must be integers.")
            if start > end:
                raise argparse.ArgumentError(self, f"Start ({start}) cannot exceed end ({end}).")
            setattr(namespace, self.dest, list(range(start, end + 1)))
        
        # Case 2: List of IDs (e.g., "8081,8084,8087" or "[8081,8084,8087]")
        else:
            if s.startswith('[') and s.endswith(']'):
                s = s[1:-1]
            try:
                ids = [int(id.strip()) for id in s.split(',')]
                setattr(namespace, self.dest, ids)
            except ValueError:
                raise argparse.ArgumentError(self, f"Invalid ID list: '{s}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)
    parser.add_argument("--channel", help="Enter channel", required=False, default=None, type=int)
    parser.add_argument("--model", help="Enter the model", required=False, type=bool)
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)
    parser.add_argument("--uuid", help="Force prev. uuid; used for debug", required=False, default=None, type=str)
    parser.add_argument("--annotation", help="Create features with specific annotation id", required=False, default="", type=str)
    parser.add_argument("--step", help="Enter step", required=False, type=int)
    parser.add_argument("--sampling", help="Random sampling qty", required=False, default=0, type=int)

    #SEGMENTATION ARGUMENTS [USED WITH 'segment' & 'detect' TASK]
    parser.add_argument("--kernel", help="Kernel size for Gaussian blurring [SEGMENTATION]", required=False, default=401, type=int)
    parser.add_argument("--sigma", help="Sigma for Gaussian blurring [SEGMENTATION]", required=False, default=350, type=int)
    parser.add_argument("--min-segment", help="Minimum segmentation size (pixels)", required=False, default=100, type=int)
    parser.add_argument("--max-segment", help="Maximum segmentation size (pixels)", required=False, default=100000, type=int)
    parser.add_argument("--segment-threshold", help="Intensity threshold (0 to 65535) [SEGMENTATION]", required=False, default=2000, type=int)
    parser.add_argument("--cell-radius", help="cell radius (pixels) [SEGMENTATION]", required=False, default=40, type=int)
    parser.add_argument("--section-range", dest='section_range', action=RangeOrIntAction, help="Section processing range (e.g. [70:79] or 70:79). If omitted, all sections are processed.", required=False, default=None)

    #PRUNING ARGUMENTS [USED WITH 'prune' & 'segment' TASK]
    parser.add_argument("--pruning", help="Enter true or false to perform pruning", required=False, default="false", type=str)
    parser.add_argument("--prune-x-range", dest='prune_x_range', action=RangeOrIntAction, help="Pruning x axis filter range (e.g. [1500:7000] or 1500:7000).", required=False, default=None)
    parser.add_argument("--prune-y-range", dest='prune_y_range', action=RangeOrIntAction, help="Pruning y axis filter range (e.g. [1500:7000] or 1500:7000).", required=False, default=None)
    parser.add_argument("--prune-amin", help="The minimum area (pixel^2) [PRUNING]", required=False, default=100, type=int)
    parser.add_argument("--prune-amax", help="The maximum area (pixel^2) [PRUNING]", required=False, default=10000, type=int)
    parser.add_argument("--prune-annotation-ids", dest='prune_annotation_ids', action=RangeOrIntAction, help="Polygon annotation volume ids -NO SPACES- [PRUNING] (e.g. [81,97,8083] or int)", required=False, default=None)
    parser.add_argument("--prune-combine-method", help="Pruning combine method: union|intersection", required=False, default="union", type=str)

    #NEUROGLANCER ARGUMENTS [USED WITH 'neuroglancer' TASK]
    #Note: ng-id should be id of full-resolution (all channels); this task will add CH3_DIFF and ML_POSITIVE point detections
    parser.add_argument("--ng-id", help="Brainsharer id", required=False, default="", type=str)

    parser.add_argument(
        "--task",
        help="Enter the task you want to perform: segment|detect|extract|train|ng_preview|create_features|neuroglancer|",
        required=True,
        type=str,
    )

    args = parser.parse_args()

    animal = args.animal
    channel = args.channel
    model = args.model
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    arg_uuid = args.uuid
    pruning = bool({"true": True, "false": False}[str(args.pruning).lower()])
    task = str(args.task).strip().lower()
    step = args.step
    
    annotation_id = args.annotation
    sampling = args.sampling #Note: only used in conjunction with 'extract' task
    segment_size_min = args.min_segment
    segment_size_max = args.max_segment
    segment_gaussian_sigma = args.sigma
    segment_gaussian_kernel = args.kernel
    segment_threshold = args.segment_threshold
    cell_radius = args.cell_radius
    section_range = args.section_range
    prune_x_range = args.prune_x_range
    prune_y_range = args.prune_y_range
    prune_amin = args.prune_amin
    prune_amax = args.prune_amax
    prune_annotation_ids = args.prune_annotation_ids
    prune_combine_method = args.prune_combine_method

    pipeline = CellMaker(animal, 
                         channel=channel,
                         task=task, 
                         step=step, 
                         model=model,
                         run_pruning=pruning,
                         prune_x_range=prune_x_range, 
                         prune_y_range=prune_y_range,
                         prune_amin=prune_amin,
                         prune_amax=prune_amax,
                         annotation_id=annotation_id, 
                         sampling=sampling, 
                         segment_size_min=segment_size_min, 
                         segment_size_max=segment_size_max, 
                         segment_gaussian_sigma=segment_gaussian_sigma, 
                         segment_gaussian_kernel=segment_gaussian_kernel, 
                         segment_threshold=segment_threshold, 
                         cell_radius=cell_radius, 
                         process_range=section_range, 
                         prune_annotation_ids=prune_annotation_ids,
                         prune_combine_method=prune_combine_method,
                         arg_uuid=arg_uuid,
                         debug=debug)

    function_mapping = {
        "segment": pipeline.segment,
        "detect": pipeline.create_detections,
        "extract": pipeline.create_annotations,        
        "train": pipeline.train,
        "ng_preview": pipeline.ng_preview,
        #"create_features": pipeline.create_features, #TODO: remove
        # "fix": pipeline.fix_coordinates #TODO: remove
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
