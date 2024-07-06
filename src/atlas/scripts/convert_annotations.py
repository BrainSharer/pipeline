import sys
from pathlib import Path
import numpy as np

from scipy.ndimage import center_of_mass
from collections import defaultdict
from tqdm import tqdm


PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.controller.marked_cell_controller import MarkedCellController
from library.controller.structure_com_controller import StructureCOMController
from library.controller.annotation_session_controller import AnnotationSessionController
from library.database_model.annotation_points import AnnotationType
from library.controller.polygon_sequence_controller import PolygonSequenceController

default_props = ["#ffff00", 1, 1, 5, 3, 1]
m_um_scale = 1000000

def load_annotation_sessions(debug):
    """x,y,z data fetched here is in micrometers
    """
    annotationSessionController = AnnotationSessionController('Allen')
    comController = StructureCOMController('Allen')
    markedCellController = MarkedCellController('Allen')
    polygonController = PolygonSequenceController('Allen')

    com_sessions = annotationSessionController.get_existing_session(annotation_type=AnnotationType.STRUCTURE_COM)
    cell_sessions = annotationSessionController.get_existing_session(annotation_type=AnnotationType.MARKED_CELL)
    polygon_sessions = annotationSessionController.get_existing_session(annotation_type=AnnotationType.POLYGON_SEQUENCE)

    # coms and cells have no centroid or source and are of type com/cell
    # coms
    for annotation_session in tqdm(com_sessions):
        continue
        # coms below is always just one for each session 
        com = comController.get_data_per_session(annotation_session.id)
        annotation = {}
        animal = annotation_session.FK_prep_id
        brain_region = annotation_session.brain_region.abbreviation
        user = annotation_session.annotator.first_name
        x = com.x / m_um_scale
        y = com.y / m_um_scale
        z = com.z / m_um_scale
        annotation['type'] = 'point'
        annotation['point'] = [x, y, z]
        annotation['description'] = brain_region
        annotation['centroid'] = [x, y, z]
        annotation['props'] =  default_props
        # update the code below for the new JSON format
        update_dict = {'annotation': annotation }
        #print(f'{annotation_session.id} {animal} {brain_region} {user} and com {x} {y} {z}')
        annotationSessionController.update_session(annotation_session.id, update_dict)
    # cells
    for annotation_session in tqdm(cell_sessions):
        continue
        # coms below is always just one for each session 
        points = markedCellController.get_data_per_session(annotation_session.id)
        annotation = {}
        animal = annotation_session.FK_prep_id
        brain_region = annotation_session.brain_region.abbreviation
        user = annotation_session.annotator.first_name
        #print(f'{annotation_session.id} {animal} {brain_region} {user} and len points {len(points)}')
        
        annotation['type'] = 'cell'
        annotation['props'] =  default_props
        annotation['description'] = brain_region
        point_list = []
        for point in points:
            x = point.x / 1000000
            y = point.y / 1000000
            z = point.z / 1000000
            if point.cell_type is not None:
                category = point.cell_type.cell_type
            else:
                category = 'UNMARKED'
            point_list.append([x, y, z])
            # update the code below for the new JSON format
        annotation['childJsons'] = point_list        
        update_dict = {'annotation': annotation }
        annotationSessionController.update_session(annotation_session.id, update_dict)
    # polygons
    for annotation_session in polygon_sessions:
        animal = annotation_session.FK_prep_id
        brain_region = annotation_session.brain_region.abbreviation
        user = annotation_session.annotator.first_name
        points = polygonController.get_data_per_session(annotation_session.id)
        index_points = defaultdict(list)
        index_orders = defaultdict(list)
        for point in points:
            try:
                index = int(point.polygon_index)
            except ValueError:
                index = int(point.z)
            index_points[index].append([point.x, point.y, point.z])
            index_orders[index].append(point.point_order)
        index_points_sorted = {}
        for index, points in index_points.items():
            points = np.array(points)
            point_indices = np.array(index_orders[index])
            point_indices = point_indices - point_indices.min()

            sorted_points = np.array(points)[point_indices, :] / m_um_scale
            index_points_sorted[index] = sorted_points
            
        polygons = []
        for index in sorted(list(index_points_sorted.keys())):
            if index not in index_points_sorted: 
                continue
            points = index_points_sorted[index]

            lines = []
            for i in range(len(points) - 1):
                lines.append({
                    "type": "line",
                    "props": default_props,
                    "pointA": points[i].tolist(),
                    "pointB": points[i + 1].tolist(),
                })
            lines.append({
                "type": "line",
                "props": default_props,
                "pointA": points[-1].tolist(),
                "pointB": points[0].tolist(),
            })

            polygons.append({
                "type": "polygon",
                "props": default_props,
                "source": points[0].tolist(),
                "centroid": np.mean(points, axis=0).tolist(),
                "childJsons": lines
            })

        if len(polygons) > 0:
            volume = {
                "type": "volume",
                "props": default_props,
                "source": polygons[0]["source"],
                "centroid": polygons[len(polygons) // 2]["centroid"],
                "childJsons": polygons,
                "description": brain_region
            }

        update_dict = {'annotation': volume }
        annotationSessionController.update_session(annotation_session.id, update_dict)

    
    if debug:
        animal = annotation_session.FK_prep_id
        brain_region = annotation_session.brain_region.abbreviation
        user = annotation_session.annotator.first_name
        points = polygonController.get_data_per_session(7947)
        index_points = defaultdict(list)
        index_orders = defaultdict(list)
        for point in points:
            try:
                index = int(point.polygon_index)
            except ValueError:
                index = int(point.z)
            index_points[index].append([point.x, point.y, point.z])
            index_orders[index].append(point.point_order)
        index_points_sorted = {}
        for index, points in index_points.items():
            points = np.array(points)
            point_indices = np.array(index_orders[index])
            point_indices = point_indices - point_indices.min()

            sorted_points = np.array(points)[point_indices, :] / m_um_scale
            print(f'sorted_points {sorted_points}')
            index_points_sorted[index] = sorted_points



if __name__ == '__main__':
    debug = False
    load_annotation_sessions(debug)
