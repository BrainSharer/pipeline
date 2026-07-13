import argparse
import sys
from pathlib import Path

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())


from library.annotation_utilities.annotation_helper import AnnotationHelper



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Annotation with ID")
    parser.add_argument("--session_id", help="Enter the session ID", required=False, default=0, type=int)
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)
    parser.add_argument("--channel", help="Enter the channel", required=False, type=int)
    parser.add_argument("--shifts", help="Enter xshift", required=False, default=(0,0,0), type=tuple)
    parser.add_argument("--task", help="Enter the task you want to perform: ",
        required=False,
        default="status",
        type=str,
    )
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)
    args = parser.parse_args()
    session_id = args.session_id
    animal = args.animal
    channel = args.channel
    shifts = args.shifts
    task = str(args.task).strip().lower()
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])

    pipeline = AnnotationHelper(session_id, animal, channel, shifts, debug)


    function_mapping = {
        "write_polygons": pipeline.write_polygons,
        "shift_cloud": pipeline.shift_cloud_annotations,
        "shift_volume": pipeline.shift_volume_annotations,
        "list_coms": pipeline.list_coms,
    }

    if task in function_mapping:
        function_mapping[task]()
        print(f"Running {task}")
    else:
        print(f"{task} is not a correct task. Choose one of these:")
        for key in function_mapping.keys():
            print(f"\t{key}")
