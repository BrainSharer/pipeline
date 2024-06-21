import sys
from pathlib import Path

from scipy.ndimage import center_of_mass
from collections import defaultdict



PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.controller.marked_cell_controller import MarkedCellController
from library.controller.structure_com_controller import StructureCOMController
from library.controller.annotation_session_controller import AnnotationSessionController
from library.database_model.annotation_points import AnnotationType
from library.controller.polygon_sequence_controller import PolygonSequenceController


def load_annotation_sessions():
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
    for annotation_session in com_sessions:
        continue
        # coms below is always just one for each session 
        com = comController.get_data_per_session(annotation_session.id)
        annotation = {}
        animal = annotation_session.FK_prep_id
        brain_region = annotation_session.brain_region.abbreviation
        user = annotation_session.annotator.first_name
        #print(f'{annotation_session.id} {animal} {brain_region} {user}')
        x = com.x / 1000000
        y = com.y / 1000000
        z = com.z / 1000000
        annotation['type'] = 'com'
        annotation['point'] = [x, y, z]
        annotation['description'] = brain_region
        annotation['props'] =  ["#ffff00",1]
        # update the code below for the new JSON format
        update_dict = {'annotation': annotation }
        annotationSessionController.update_session(annotation_session.id, update_dict)
    # cells
    for annotation_session in cell_sessions:
        continue
        # coms below is always just one for each session 
        points = markedCellController.get_data_per_session(annotation_session.id)
        annotation = {}
        animal = annotation_session.FK_prep_id
        brain_region = annotation_session.brain_region.abbreviation
        user = annotation_session.annotator.first_name
        print(f'{annotation_session.id} {animal} {brain_region} {user} and len points {len(points)}')
        
        annotation['type'] = 'cell'
        annotation['props'] =  ["#ffff00", 1]
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
        # coms below is always just one for each session 
        points = polygonController.get_data_per_session(annotation_session.id)
        annotation = {}
        animal = annotation_session.FK_prep_id
        brain_region = annotation_session.brain_region.abbreviation
        user = annotation_session.annotator.first_name
        print(f'{annotation_session.id} {animal} {brain_region} {user} and len points {len(points)}')
        
        annotation['type'] = 'volume'
        annotation['props'] =  ["#ffff00", 1]
        point_list = []
        for point in points:
            polygon_index = point.polygon_index
            point_order = point.point_order
            x = point.x / 1000000
            y = point.y / 1000000
            z = point.z / 1000000
            point_list.append([x, y, z])
            # update the code below for the new JSON format
        annotation['childJsons'] = point_list        
        update_dict = {'annotation': annotation }
        annotationSessionController.update_session(annotation_session.id, update_dict)



"""
    polygon_json = []
    npoints = len(polygon_points)
    parent_annotation, child_ids = create_parent_annotation_json(npoints, polygon_id, polygon_points[0], _type='polygon',parent_id=parent_id)
    polygon_json.append(parent_annotation)
    for point in range(npoints - 1):
        line = {}
        line["pointA"] = polygon_points[point]
        line["pointB"] = polygon_points[point + 1]
        line["type"] = "line"
        line["id"] = child_ids[point]
        line["parentAnnotationId"] = polygon_id
        line["props"] = [hexcolor]
        polygon_json.append(line)
    line = {}
    line["pointA"] = polygon_points[-1]
    line["pointB"] = polygon_points[0]
    line["type"] = "line"
    line["id"] = child_ids[-1]
    line["parentAnnotationId"] = polygon_id
    line["props"] = [hexcolor]
    polygon_json.append(line)
    return polygon_json



def create_parent_annotation_json(npoints, id, source, _type, child_ids=None,parent_id = None, description = None):
    '''create the json entry for a parent annotation.  The parent annotation need to have a 
    specific id and the list of id for all the children

    Args:
        npoints (int): number of points in this parent annotation
        parent_id (int): id of parent annotation
        source (list of x,y,z): the source coordinate
        _type (string): annotation type: this could be polygon or volumes
        child_ids (list, optional): list of id of child annotations that belong to the parent annotation. Defaults to None.

    Returns:
        _type_: _description_
    '''

    parent_annotation = {}
    if child_ids is None:
        child_ids = [random_string() for _ in range(npoints)]
    parent_annotation["source"] = source
    parent_annotation["childAnnotationIds"] = child_ids
    if parent_id is not None:
        parent_annotation["parentAnnotationId"] = parent_id
    if description is not None:
        parent_annotation["description"] = description
    parent_annotation["type"] = _type
    parent_annotation["id"] = id
    parent_annotation["props"] = [hexcolor]
    return parent_annotation, child_ids
    


"""

if __name__ == '__main__':
    load_annotation_sessions()
