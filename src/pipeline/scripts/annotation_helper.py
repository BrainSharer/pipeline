import argparse
from collections import OrderedDict
from pathlib import Path
import shutil
import sys, os
import cv2
import numpy as np

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.controller.sql_controller import SqlController
from library.image_manipulation.filelocation_manager import FileLocationManager
from library.utilities.utilities_process import M_UM_SCALE, SCALING_FACTOR, random_string, read_image, write_image


class AnnotationHelper:
    def __init__(self, session_id, animal, channel=1, xshift=0, yshift=0, debug=False):
        self.session_id = session_id
        self.animal = animal
        self.channel = channel
        self.color = 65000
        self.xshift = xshift
        self.yshift = yshift
        self.fileLocationManager = FileLocationManager(animal)
        self.sqlController = SqlController(animal)
        self.debug = debug

    def write_polygons(self):
        input = self.fileLocationManager.get_directory(self.channel, downsample=True, inpath="aligned")
        output = self.fileLocationManager.get_directory(self.channel, downsample=True, inpath="aligned_shell")
        polygons = self.sqlController.get_volume(self.session_id)
        if os.path.exists(output):
            print(f"Output directory {output} already exists")
            shutil.rmtree(output)
        os.makedirs(output, exist_ok=True)
        for file in self.files:
            filepath = os.path.join(input, file)
            outpath = os.path.join(output, file)
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
                volume_slice = cv2.polylines(volume_slice, [contour_points], isClosed=True, color=self.color, thickness=10)
            except Exception as e:
                print(f"Error in section {section} with {e}")
                continue
            write_image(outpath, volume_slice)

    def shift_annotations(self):
        """
        Shifts the annotations based on the x and y shift values and updates the annotation session.
        This method updates the points so they are red and a bit bigger (size 10 instead of 5). It calculates the new 
        positions of the points based on the x and y shift values, converts them to pixel coordinates, and updates 
        the annotation session with the new points.
        Attributes:
            self.sqlController (SQLController): Controller to interact with the SQL database.
            self.session_id (int): The ID of the current annotation session.
            self.xshift (float): The shift value in the x direction.
            self.yshift (float): The shift value in the y direction.
        Raises:
            KeyError: If the 'childJsons' key is not found in the annotation data.
        Returns:
            None
        """

        xy_resolution = self.sqlController.scan_run.resolution * SCALING_FACTOR
        z_resolution = self.sqlController.scan_run.zresolution
        default_props = ["#ff0000", 1, 1, 10, 3, 1]
        cloud_points = {}
        points = []
        childJsons = []
        parent_id = f"{random_string()}"
        annotation_session = self.sqlController.get_annotation_by_id(self.session_id)
        annotation = annotation_session.annotation
        description = annotation["description"]
        try:
            data = annotation["childJsons"]
        except KeyError as ke:
            print("No childJsons key in data")
            print(f"Error: {ke}")

        xshift = self.xshift / M_UM_SCALE * xy_resolution
        yshift = self.yshift * M_UM_SCALE
        for row in data:
            x, y, section = row["point"]
            if self.debug:
                pixel_point = [x * M_UM_SCALE / xy_resolution, y * M_UM_SCALE / xy_resolution, section * M_UM_SCALE / z_resolution]
                pixel_point = [round(x) for x in pixel_point]
                print(f"Original = {pixel_point}", end="\t")
            x += xshift
            y += yshift
            point = [x, y, section]
            if self.debug:
                pixel_point = [x * M_UM_SCALE / xy_resolution, y * M_UM_SCALE / xy_resolution, section * M_UM_SCALE / z_resolution]
                pixel_point = [round(x) for x in pixel_point]
                print(f"shifted point = {pixel_point}")
            childJson = {
                "point": point,
                "type": "point",
                "parentAnnotationId": row["parentAnnotationId"],
                "props": default_props
            }
            childJsons.append(childJson)
            points.append(point)

        cloud_points["source"] = points[0]
        cloud_points["centroid"] = np.mean(points, axis=0).tolist()
        cloud_points["childrenVisible"] = True
        cloud_points["type"] = "cloud"
        cloud_points["description"] = f"{description}"
        cloud_points["sessionID"] = f"{parent_id}"
        cloud_points["props"] = default_props
        cloud_points["childJsons"] = childJsons

        if self.debug:
            x,y,section = cloud_points["centroid"]
            pixel_point = [x * M_UM_SCALE / xy_resolution, y * M_UM_SCALE / xy_resolution, section * M_UM_SCALE / z_resolution]
            pixel_point = [round(x) for x in pixel_point]
            print(f"Shifted centroid={pixel_point}")
        else:
            update_dict = {'annotation': cloud_points}
            print(f'Updating session {self.session_id} with length {len(childJsons)}')
            self.sqlController.update_session(session_id, update_dict=update_dict)

    def convert_to_allen(self, com):
        affine_transformation = np.array(
            [
                [9.36873602e-01, 6.25910930e-02, 3.41078823e-03, 4.07945327e02],
                [5.68396089e-04, 1.18742192e00, 6.28369930e-03, 4.01267566e01],
                [-1.27831427e-02, 8.42516452e-03, 1.11913658e00, -6.42895756e01],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )

    def list_coms(self):
        """
        Lists the COMs from the annotation session table. The data
        is stored in meters so you will want to convert it to micrometers
        and then by the resolution of the scan run.
        """
        xy_resolution = self.sqlController.scan_run.resolution
        z_resolution = self.sqlController.scan_run.zresolution

        annotator_id = 1 # Hardcoded to edward
        com_dictionaries = self.sqlController.get_com_dictionary(prep_id=self.animal, annotator_id=annotator_id)
        com_dictionaries = OrderedDict(sorted(com_dictionaries.items()))
        for k, v in com_dictionaries.items():
            x = round(v[0] * M_UM_SCALE / xy_resolution, 2)
            y = round(v[1] * M_UM_SCALE / xy_resolution, 2)
            z = round(v[2] * M_UM_SCALE / z_resolution, 2)
            print(k, x,y,z)
            
        return com_dictionaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Annotation with ID")
    parser.add_argument("--session_id", help="Enter the session ID", required=False, default=0, type=int)
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)
    parser.add_argument("--channel", help="Enter the channel", required=False, type=int)
    parser.add_argument("--xshift", help="Enter xshift", required=False, default=0, type=float)
    parser.add_argument("--yshift", help="Enter yshift", required=False, default=0, type=float)
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
    xshift = args.xshift
    yshift = args.yshift
    task = str(args.task).strip().lower()
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])

    pipeline = AnnotationHelper(session_id, animal, channel, xshift, yshift, debug)


    function_mapping = {
        "write_polygons": pipeline.write_polygons,
        "shift_annotations": pipeline.shift_annotations,
        "list_coms": pipeline.list_coms,
    }

    if task in function_mapping:
        function_mapping[task]()
        print(f"Running {task}")
    else:
        print(f"{task} is not a correct task. Choose one of these:")
        for key in function_mapping.keys():
            print(f"\t{key}")
