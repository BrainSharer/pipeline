import sys
from pathlib import Path

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.registration.brain_structure_manager import BrainStructureManager
from library.controller.structure_com_controller import StructureCOMController
from library.controller.annotation_session_controller import AnnotationSessionController
from library.database_model.annotation_points import PolygonSequence, AnnotationType, StructureCOM

def load_com():
    animal = 'Allen'
    brainManager = BrainStructureManager(animal)
    source = 'MANUAL'
    annotationSessionController = AnnotationSessionController(animal)
    structureController = StructureCOMController(animal)
    # Pn looks like one mass in Allen    
    structures = {
        'SC': (373, 72, 227.68),
        'IC': (416.1, 93.29, 227.68),
        'AP': (506.13, 200.42, 227.69),
        'SNR_L': (337.62106378683893, 206.69036132418248, 170),
        'SNR_R': (337.62106378683893, 206.69036132418248, 289),
        'PBG_L': (376.07, 153.81, 143),
        'PBG_R': (376.07, 153.81, 313),
        '3N_L': (364.12, 151.77, 220),
        '3N_R': (364.12, 151.77, 235),
        '4N_L': (383.51, 151.05, 218),
        '4N_R': (383.51, 151.05, 238),
        'SNC_L': (334.2, 204.95, 168),
        'SNC_R': (334.2, 204.95, 287),
        'VLL_L': (378.59, 206.99, 168),
        'VLL_R': (378.59, 206.99, 296),
        '5N_L': (407.73, 211.44, 168),
        '5N_R': (407.73, 211.44, 292),
        'LC_L': (428.39, 171.13, 189),
        'LC_R': (428.39, 171.13, 266),
        'DC_L': (450.82, 200.08, 127),
        'DC_R': (450.82, 200.08, 326),
        'Sp5C_L': (513.3, 235.84, 162),
        'Sp5C_R': (513.3, 235.84, 295),
        'SpV_L': (513.3, 235.84, 167),
        'SpV_R': (513.3, 235.84, 292),
        'Sp5I_L': (484.26, 234.88, 156),
        'Sp5I_R': (484.26, 234.88, 300),
        'Sp5O_L': (445.55, 236.5, 155),
        'Sp5O_R':(445.55, 236.5, 300),
        '6N_L': (430.85, 208.59, 212),
        '6N_R': (430.85, 208.59, 245),
        '7N_L': (434.11, 271.02, 176),
        '7N_R': (434.11, 271.02, 283),
        '7n_L': (427.5, 219.3, 180),
        '7n_R': (427.5, 219.3, 277),

        }
    for abbreviation, points in structures.items():
        FK_brain_region_id = structureController.structure_abbreviation_to_id(abbreviation=abbreviation)
        FK_session_id = annotationSessionController.create_annotation_session(annotation_type=AnnotationType.STRUCTURE_COM, 
                                                                                FK_user_id=1, FK_prep_id=animal, FK_brain_region_id=FK_brain_region_id)
        x,y,z = (p*25 for p in points)
        print(source, FK_brain_region_id, FK_session_id, abbreviation, points, x,y,z)
        com = StructureCOM(source=source, x=x, y=y, z=z, FK_session_id=FK_session_id)
        brainManager.sqlController.add_row(com)




def load_foundation_brain_polygon_sequences(animal):
    brainManager = BrainStructureManager(animal)
    brainManager.load_aligned_contours()
    contours = brainManager.aligned_contours
    annotationSessionController = AnnotationSessionController(animal)
    structureController = StructureCOMController(animal)
    xy_resolution = brainManager.sqlController.scan_run.resolution
    zresolution = brainManager.sqlController.scan_run.zresolution
    source = 'NA'
    for abbreviation, v in contours.items():
        FK_brain_region_id = structureController.structure_abbreviation_to_id(abbreviation=abbreviation)
        FK_session_id = annotationSessionController.create_annotation_session(annotation_type=AnnotationType.POLYGON_SEQUENCE, 
                                                                                FK_user_id=1, FK_prep_id=animal, FK_brain_region_id=FK_brain_region_id)
        for section, vertices in v.items():
            polygon_index = int(section)
            point_order = 1
            z = float(int(section) * zresolution)
            vlist = []
            for x,y in vertices:
                x = x * 32 * xy_resolution
                y = y * 32 * xy_resolution
                #print(source, x, y, z, polygon_index, point_order, FK_session_id)
                polygon_sequence = PolygonSequence(x=x, y=y, z=z, source=source, 
                                                polygon_index=polygon_index, point_order=point_order, FK_session_id=FK_session_id)
                point_order += 1
                vlist.append(polygon_sequence)
                #brainManager.sqlController.add_row(polygon_sequence)
            brainManager.sqlController.session.bulk_save_objects(vlist)
            brainManager.sqlController.session.commit()


if __name__ == '__main__':
    animals = ['MD585', 'MD589', 'MD594']
    for animal in animals:
        continue
        load_foundation_brain_polygon_sequences(animal)
    load_com()
