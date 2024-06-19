import sys
from pathlib import Path

from scipy.ndimage import center_of_mass
from collections import defaultdict


PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.controller.structure_com_controller import StructureCOMController
from library.controller.annotation_session_controller import AnnotationSessionController
from library.database_model.annotation_points import AnnotationType


def load_annotation_sessions():
    """x,y,z data fetched here is in micrometers
    """
    annotationSessionController = AnnotationSessionController('Allen')
    comController = StructureCOMController('Allen')

    com_sessions = annotationSessionController.get_existing_session(annotation_type=AnnotationType.STRUCTURE_COM)
    cell_sessions = annotationSessionController.get_existing_session(annotation_type=AnnotationType.MARKED_CELL)
    polygon_sessions = annotationSessionController.get_existing_session(annotation_type=AnnotationType.POLYGON_SEQUENCE)

    for session in com_sessions:
        coms = comController.get_data_per_session(session.id)
        annotation = {}
        
        for com in coms:
            x = com.x / 1000000
            y = com.y / 1000000
            z = com.z / 1000000
            print(f'{session.id} {x} {y} {z}')
            annotation[session.id] = (x, y, z)
            # update the code below for the new JSON format
            update_dict = {'annotation': annotation }

        annotationSessionController.update_session(session.id, update_dict)


if __name__ == '__main__':
    load_annotation_sessions()
