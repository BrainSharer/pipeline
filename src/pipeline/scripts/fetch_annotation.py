import argparse
from pathlib import Path
import sys, socket
from timeit import default_timer as timer


PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.image_manipulation.pipeline_process import Pipeline



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Annotation with ID")
    parser.add_argument("--session_id", help="Enter the session ID", required=True, type=int)
    args = parser.parse_args()
    session_id = args.session_id
    animal = 'MD589'
    pipeline = Pipeline(animal)

    pipeline.sqlController.get_annotation(session_id)
