import argparse
import sys
from pathlib import Path

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.controller.sql_controller import SqlController


def load_coms(animal, annotator_id):
    controller = SqlController(animal)
    coms = controller.get_com_dictionary(animal, annotator_id)
    for k,v in coms.items():
        print(k, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)
    parser.add_argument("--annotator_id", help="Enter the annotator_id", required=True, type=int)
    args = parser.parse_args()

    animal = args.animal
    annotator_id = args.annotator_id
    load_coms(animal, annotator_id)