from collections import OrderedDict, defaultdict
from pathlib import Path
import shutil
import sys, os
import cv2
import numpy as np
import sqlalchemy

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.controller.sql_controller import SqlController
from library.image_manipulation.filelocation_manager import FileLocationManager
from library.utilities.utilities_process import M_UM_SCALE, SCALING_FACTOR, random_string, read_image, write_image
from library.database_model.annotation_points import AnnotationLabel, AnnotationSession
from library.atlas.atlas_utilities import get_edge_coordinates


class AnnotationHelper:
    def __init__(self, animal, session_id=None, channel=1, shifts=(0,0,0), debug=False):
        self.session_id = session_id
        self.animal = animal
        self.channel = channel
        self.color = 65000
        self.shift = shifts
        self.fileLocationManager = FileLocationManager(animal)
        self.sqlController = SqlController(animal)
        self.debug = debug
        self.annotator_id = 1 # Hard code to edward
        self.xy_resolution = self.sqlController.scan_run.resolution
        self.z_resolution = self.sqlController.scan_run.zresolution

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

    def shift_cloud_annotations(self):
        """
        Shifts the annotations based on the x and y shift values and updates the annotation session.
        This method updates the points so they are red and a bit bigger (size 10 instead of 5). It calculates the new 
        positions of the points based on the x and y shift values, converts them to pixel coordinates, and updates 
        the annotation session with the new points.
        Attributes:
            self.sqlController (SQLController): Controller to interact with the SQL database.
            self.session_id (int): The ID of the current annotation session.
            self.shifts (float): The shift value in the (x,y,z) direction.
        Raises:
            KeyError: If the 'childJsons' key is not found in the annotation data.
        Returns:
            None
        """

        xy_resolution = self.sqlController.scan_run.resolution * SCALING_FACTOR
        z_resolution = self.sqlController.scan_run.zresolution
        default_props = ["#ff0000", 1, 1, 10, 3, 1]
        annotation_points = {}
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

        xshift = self.shifts[0] / M_UM_SCALE * xy_resolution
        yshift = self.shifts[1] * M_UM_SCALE
        zshift = self.shifts[2] * M_UM_SCALE
        for row in data:
            try:
                x, y, section = row["point"]
            except KeyError as ke:
                print(f'No point key in row={row}')
                return
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

        annotation_points["source"] = points[0]
        annotation_points["centroid"] = np.mean(points, axis=0).tolist()
        annotation_points["childrenVisible"] = True
        annotation_points["type"] = "cloud"
        annotation_points["description"] = f"{description}"
        annotation_points["sessionID"] = f"{parent_id}"
        annotation_points["props"] = default_props
        annotation_points["childJsons"] = childJsons

        if self.debug:
            x,y,section = annotation_points["centroid"]
            pixel_point = [x * M_UM_SCALE / xy_resolution, y * M_UM_SCALE / xy_resolution, section * M_UM_SCALE / z_resolution]
            pixel_point = [round(x) for x in pixel_point]
            print(f"Shifted centroid={pixel_point}")
        else:
            update_dict = {'annotation': annotation_points}
            print(f'Updating session {self.session_id} with length {len(childJsons)}')
            self.sqlController.update_session(self.session_id, update_dict=update_dict)


    def shift_volume_annotations(self):
        """
        {
            'pointA': [0.010121309198439121, 0.008618032559752464, 0.005150000099092722], 
            'pointB': [0.009859367273747921, 0.008647968992590904, 0.005150000099092722], 
            'type': 'line', 
            'parentAnnotationId': 'fdb486526ebb517fd6b7d19f0da63b96fe0acc1b', 
            'props': ['#00ff59', 1, 1, 10, 3, 1]}
        
        """
        xy_resolution = self.sqlController.scan_run.resolution
        z_resolution = self.sqlController.scan_run.zresolution
        default_props = ["#ff0000", 1, 1, 5, 3, 1]
        points = []
        childJsons = []
        annotation_session = self.sqlController.get_annotation_by_id(self.session_id)
        volume = annotation_session.annotation
        description = volume["description"]
        try:
            polygons = volume["childJsons"]
        except KeyError as ke:
            print("No childJsons key in volume")
            print(f"Error: {ke}")

        xshift = self.shifts[0] / M_UM_SCALE * xy_resolution
        yshift = self.shifts[1] / M_UM_SCALE * xy_resolution
        zshift = self.shifts[2] / M_UM_SCALE * z_resolution

        reformatted_polygons = []
        for polygon in polygons:
            if 'childJsons' not in polygon:
                print('No childJsons key in row')
                return
            new_lines = []
            new_polygon = {}
            for line in polygon['childJsons']:
                xa,ya,za = line['pointA']
                xb,yb,zb = line['pointB']
                if self.debug:
                    pixel_point = [xa * M_UM_SCALE / xy_resolution, ya * M_UM_SCALE / xy_resolution, za * M_UM_SCALE / z_resolution]
                    pixel_point = [round(x) for x in pixel_point]
                    print(f"Original = {pixel_point}", end="\t")
                xa += xshift
                ya += yshift
                za += zshift
                xb += xshift
                yb += yshift
                zb += zshift
                pointA = [xa, ya, za]
                pointB = [xb, yb, zb]
                new_line = {
                    "pointA": pointA,
                    "pointB": pointB,
                    "type": "line",
                    "parentAnnotationId": line["parentAnnotationId"],
                    "props": default_props
                }
                if self.debug:
                    pixel_point = [xa * M_UM_SCALE / xy_resolution, ya * M_UM_SCALE / xy_resolution, za * M_UM_SCALE / z_resolution]
                    pixel_point = [round(x) for x in pixel_point]
                    print(f"shifted point = {pixel_point}")
                new_lines.append(new_line)
                points.append(pointA)

            # polygon keys
            new_polygon["source"] = points[0]
            new_polygon["centroid"] = np.mean(points, axis=0).tolist()
            new_polygon["childrenVisible"] = True
            new_polygon["type"] = "polygon"
            new_polygon["parentAnnotationId"] = polygon["parentAnnotationId"]
            new_polygon["description"] = f"{description}"
            new_polygon["props"] = default_props
            new_polygon["childJsons"] = new_lines

            reformatted_polygons.append(new_polygon)

        # Create the annotation dictionary
        # volume keys=['type', 'props', 'source', 'centroid', 'childJsons', 'description']
        # create the childJsons dictionary
        new_annotation = {}
        new_annotation["type"] = "volume"
        new_annotation["props"] = default_props
        new_annotation["source"] = points[0]
        new_annotation["centroid"] = np.mean(points, axis=0).tolist()
        new_annotation["childJsons"] = reformatted_polygons
        new_annotation["description"] = f"{description}"   

        if self.debug:
            x,y,section = new_annotation["centroid"]
            pixel_point = [x * M_UM_SCALE / xy_resolution, y * M_UM_SCALE / xy_resolution, section * M_UM_SCALE / z_resolution]
            pixel_point = [round(x) for x in pixel_point]
            print(f"Shifted centroid={pixel_point}")
        else:
            update_dict = {'annotation': new_annotation}
            print(f'Updating session {self.session_id} with length {len(childJsons)}')
            self.sqlController.update_session(self.session_id, update_dict=update_dict)



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
    
    def upsert_annotationXXX(self, volume):
        default_props = ["#00FF00", 1, 1, 5, 3, 1]
        volumeID = random_string()
        polygons = []
        description = "Predicted values"

        for z, polygon in volume.items():
            polygonID = random_string()
            lines = []
            len_polygon = polygon.shape[0]
            # neuroglancer uses another 0.5 for the z axis
            zm =  (z + 0.5) * self.z_resolution / M_UM_SCALE
            for i in range(len_polygon - 1):
                xa, ya = polygon[i]
                xam = xa / M_UM_SCALE * self.xy_resolution * SCALING_FACTOR
                yam = ya / M_UM_SCALE * self.xy_resolution * SCALING_FACTOR

                xb, yb = polygon[i+1]
                xbm = xb / M_UM_SCALE * self.xy_resolution * SCALING_FACTOR
                ybm = yb / M_UM_SCALE * self.xy_resolution * SCALING_FACTOR

                lines.append({
                    "type": "line",
                    "props": default_props,
                    "pointA": [xam, yam, zm],
                    "pointB": [xbm, ybm, zm],
                    "parentAnnotationId": polygonID
                })

            # close polygon
            xa, ya = polygon[-1]
            xam = xa / M_UM_SCALE * self.xy_resolution * SCALING_FACTOR
            yam = ya / M_UM_SCALE * self.xy_resolution * SCALING_FACTOR

            xb, yb = polygon[0]
            xbm = xb / M_UM_SCALE * self.xy_resolution * SCALING_FACTOR
            ybm = yb / M_UM_SCALE * self.xy_resolution * SCALING_FACTOR
            lines.append({
                "type": "line",
                "props": default_props,
                "pointA": [xam, yam, zm],
                "pointB": [xbm, ybm, zm],
                "parentAnnotationId": polygonID
            })
            polygon_centroid = [np.mean(d["pointA"]) for d in lines]
            print('polygon centroid', polygon_centroid)
            polygons.append({
                "type": "polygon",
                "props": default_props,
                "source": lines[0]["pointA"],
                "centroid": polygon_centroid,
                "childJsons": lines,
                "parentAnnotationId": volumeID
            })
            #print(f'z {z} polygon z centroid {polygon_centroid}' )
            
        if len(polygons) > 0:
            json_entry = {
                "id": volumeID,
                "source": polygons[0]["source"],
                "centroid": polygons[len(polygons) // 2]["centroid"],
                "childrenVisible": True,
                "type": "volume",
                "description": description,
                "props": default_props,
                "childJsons": polygons
            }

        volume_centroid = polygons[len(polygons) // 2]["centroid"]
        print(f'volume centroid {volume_centroid}')
        
        try:
            annotation_session = (
                self.sqlController.session.query(AnnotationSession)
                .filter(AnnotationSession.active == True)
                .filter(AnnotationSession.FK_user_id == self.annotator_id)
                .filter(AnnotationSession.FK_prep_id == self.animal)
                .filter(AnnotationSession.annotation["description"] == description)
                .one_or_none()
            )
        except Exception as e:
            print(f"Found more than one structure for {self.animal} {description}. Exiting program, please fix")
            print(e)
            exit(1)

        labels = ['TG_L', 'TG_R']
        
        if annotation_session is None:
            print(f'Inserting {self.animal} with {description}')
            
            try:
                self.sqlController.insert_annotation_with_labels(
                    FK_user_id=self.annotator_id,
                    FK_prep_id=self.animal,
                    annotation=json_entry,
                    labels=labels)
            except sqlalchemy.exc.OperationalError as e:
                print(f"Operational error inserting annotation: {e}")
                self.sqlController.session.rollback()
            
        else:                
            update_dict = {'annotation': json_entry}
            print(f'Updating {self.animal} session {annotation_session.id} with {description}')
            self.sqlController.update_session(annotation_session.id, update_dict=update_dict)

        print('\nfinished processing points')


    def upsert_annotation(self, polygons, structure, get_even=False):
        """Creates a volume from a dictionary of polygons
        The polygons are in the form of {section: [x,y]}  in the downsampled pixel space
        """

        
        default_props = ["#ff0000", 1, 1, 5, 3, 1]

        reformatted_polygons = []
        centroids = []
        counter = 0
        for (section, points) in sorted(polygons.items()):
            section = int(section)
            if get_even:
                points = get_evenly_spaced_vertices(points)

            if points is None or points.shape[0] == 0: 
                continue
            new_lines = []
            new_polygon = {}
            parentAnnotationId = random_string()
            point_summary = []
            for i in range(len(points)):
                try:
                    xa,ya = points[i]
                except ValueError:
                    continue
                try:
                    xb,yb = points[i+1]
                except IndexError:
                    xb,yb = points[0]
                except ValueError as ve:
                    print(f"Value Error B with {structure} {ve}")
                    continue

                xa = float(xa * self.xy_resolution * SCALING_FACTOR / M_UM_SCALE)
                ya = float(ya * self.xy_resolution * SCALING_FACTOR / M_UM_SCALE)
                # neuroglancer uses another 0.5 for the z axis
                z =  float((section + 0.5) * self.z_resolution / M_UM_SCALE)
                xb = float(xb * self.xy_resolution * SCALING_FACTOR / M_UM_SCALE)
                yb = float(yb * self.xy_resolution * SCALING_FACTOR / M_UM_SCALE)

                pointA = [xa, ya, z]
                pointB = [xb, yb, z]
                
                new_line = {
                    "pointA": pointA,
                    "pointB": pointB,
                    "type": "line",
                    "parentAnnotationId": parentAnnotationId,
                    "props": default_props
                }
                new_lines.append(new_line)
                point_summary.append(pointA)
                counter += 1

            sx, sy = points[0]
            parentAnnotationId = random_string()
            new_polygon["source"] = [float(sx), float(sy), z]
            new_polygon["centroid"] = np.mean(point_summary, axis=0).tolist()
            new_polygon["childrenVisible"] = True
            new_polygon["type"] = "polygon"
            new_polygon["parentAnnotationId"] = parentAnnotationId
            new_polygon["description"] = f"{structure}"
            new_polygon["props"] = default_props
            new_polygon["childJsons"] = new_lines

            centroids.append(new_polygon["centroid"])
            reformatted_polygons.append(new_polygon)

        # Create the annotation dictionary
        json_entry = {}
        json_entry["type"] = "volume"
        json_entry["props"] = default_props
        json_entry["source"] = centroids[0]
        json_entry["centroid"] = np.mean(centroids, axis=0).tolist()
        json_entry["childJsons"] = reformatted_polygons
        json_entry["description"] = f"{structure}"   

        if self.debug:
            centroid = json_entry["centroid"]
            print(f'total points={counter}', end=" ")
            print(f"len of centroids={len(centroids)} len of polygons={len(polygons)} len of reformatted_polygons={len(reformatted_polygons)}")
            numpy_keys = [key for key, value in json_entry.items() if isinstance(value, np.ndarray)]
            if len(numpy_keys) > 0:
                print('Found some numpy values. Keys:')
                print(numpy_keys)
            else:
                print('No JSON values are numpy arrays')

            xp,yp,zp = centroid
            xp *= M_UM_SCALE
            yp *= M_UM_SCALE
            zp *= M_UM_SCALE
            print(f"Centroid of {structure} {centroid=} xp={xp/self.xy_resolution}, yp={yp/self.xy_resolution}, zp={zp/self.z_resolution}")
            return


        label_ids = self.get_label_ids(structure)
        
        try:
            annotation_session = (
                self.sqlController.session.query(AnnotationSession)
                .filter(AnnotationSession.active == True)
                .filter(AnnotationSession.FK_prep_id == self.animal)
                .filter(AnnotationSession.FK_user_id == self.annotator_id)
                .filter(AnnotationSession.labels.any(AnnotationLabel.id.in_(label_ids)))
                .filter(AnnotationSession.annotation["type"] == "volume")
                .one_or_none()
            )
        except Exception as e:
            print(f"Found more than one structure for {self.animal} {structure}. Exiting program, please fix")
            exit(1)

        if annotation_session is None:
            print(f"Inserting {structure} for {self.animal}")
            try:
                self.sqlController.insert_annotation_with_labels(
                    FK_user_id=self.annotator_id,
                    FK_prep_id=self.animal,
                    annotation=json_entry,
                    labels=[structure])
            except sqlalchemy.exc.OperationalError as e:
                print(f"Operational {e} for {structure}")
                self.sqlController.session.rollback()
        else:                
            update_dict = {'annotation': json_entry}
            print(f'Updating {self.animal} session {annotation_session.id} with {structure}')
            self.sqlController.update_session(annotation_session.id, update_dict=update_dict)

    def get_label_ids(self, structure) -> list:
        label = self.sqlController.get_annotation_label(structure)
        if label is not None:
            label_ids = [label.id]
        else:
            print(f"Could not find {structure} label in database")
            label_ids = [0]

        return label_ids


def get_evenly_spaced_vertices(vertices: list, num_points=20) -> np.ndarray:
    """
    Returns a specified number of evenly spaced points along the perimeter of a polygon.

    Args:
        vertices (list of tuple): List of (x, y) tuples representing the polygon vertices.
        num_points (int): The number of evenly spaced points to return.

    Returns:
        list of tuple: List of (x, y) tuples representing the evenly spaced points.
    """
    # Close the polygon if it's not already closed


    if not isinstance(vertices, list):
        if isinstance(vertices, np.ndarray):
            #non_zero_coords = np.argwhere(vertices != 0)
            non_zero_coords = get_edge_coordinates(vertices)
            vertices = [tuple(row) for row in non_zero_coords]
            #return vertices


    if not isinstance(vertices[0], tuple) and len(vertices[0]) != 2:
        print("Vertices[0] should be a list of tuples.")
        print(type(vertices[0]), len(vertices[0]))
        exit(1)
    
    if vertices[0] != vertices[-1]:
        vertices.append(vertices[0])

    # Calculate distances between consecutive vertices
    distances = [np.linalg.norm(np.subtract(vertices[i+1], vertices[i])) for i in range(len(vertices)-1)]
    perimeter = sum(distances)

    # Total length between each evenly spaced point
    try:
        step = perimeter / num_points
    except TypeError as te:
        print(f"Error in calculating step size: {te}")
        print(f"perimeter = {perimeter}")
        print(f"num_points = {num_points}")
        exit(1)

    # Generate points
    result = []
    current_distance = 0
    i = 0
    while len(result) < num_points:
        start = np.array(vertices[i])
        end = np.array(vertices[i+1])
        segment_length = distances[i]

        while current_distance + segment_length >= len(result) * step:
            t = ((len(result) * step) - current_distance) / segment_length
            point = start + t * (end - start)
            result.append(tuple(point))

            if len(result) == num_points:
                break

        current_distance += segment_length
        i += 1
        if i >= len(distances):  # Safety break in case of rounding issues
            break

    return np.array(result)