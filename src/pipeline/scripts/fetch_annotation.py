import argparse
from pathlib import Path
import shutil
import sys, os
import cv2
import numpy as np

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.image_manipulation.pipeline_process import Pipeline
from library.utilities.utilities_process import read_image, write_image



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Annotation with ID")
    parser.add_argument("--session_id", help="Enter the session ID", required=True, type=int)
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)
    parser.add_argument("--channel", help="Enter channel", required=False, default=1, type=int)
    args = parser.parse_args()
    session_id = args.session_id
    animal = args.animal
    channel = args.channel
    pipeline = Pipeline(animal)
    pipeline.input = pipeline.fileLocationManager.get_directory(channel, downsample=True, inpath="aligned")
    pipeline.output = pipeline.fileLocationManager.get_directory(channel, downsample=True, inpath="aligned_shell")
    if os.path.exists(pipeline.output):
        print(f"Output directory {pipeline.output} already exists")
        shutil.rmtree(pipeline.output)
    os.makedirs(pipeline.output, exist_ok=True)
    files = sorted(os.listdir(pipeline.input))

    polygons = pipeline.sqlController.get_annotation(session_id)
    color = 65000
    for file in files:
        filepath = os.path.join(pipeline.input, file)
        outpath = os.path.join(pipeline.output, file)
        volume_slice = read_image(filepath)
        section = int(file.split(".")[0])        
        try:
            contour_points = polygons[section]
        except KeyError:
            print(f"No data for section {section}")
            continue
        vertices = np.array(contour_points)
        contour_points = (vertices).astype(np.int32)
        if len(contour_points) < 3:
            print(f"Skipping section {section} with less than 3 points")
            continue
        else:
            print(f"{section} {contour_points[0]}")
        try:
            volume_slice = cv2.polylines(volume_slice, [contour_points], isClosed=True, color=color, thickness=10)
        except Exception as e:
            print(f"Error in section {section} with {e}")
            continue
        write_image(outpath, volume_slice)
    
    #for section in sorted(polygons):
    #    print(section, polygons[section][0])

