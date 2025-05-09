import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.registration.brain_structure_manager import BrainStructureManager
from library.controller.annotation_session_controller import AnnotationSessionController
from library.utilities.atlas import allen_structures, singular_structures

def load_com():
    um = 10
    animal = 'Allen'
    mcc = MouseConnectivityCache(resolution=um)
    rsp = mcc.get_reference_space()
    print('Shape of entire brain', rsp.annotation.shape)
    midpoint = int(rsp.annotation.shape[2] / 2)
    print('Mid z', midpoint)
    brainManager = BrainStructureManager(animal)
    #brainManager.inactivate_coms(animal)
    source = 'MANUAL'
    #annotationSessionController = AnnotationSessionController(animal)
    #structureController = StructureCOMController(animal)
    # Pn looks like one mass in Allen
    for abbreviation, structure_id in allen_structures.items():
        if type(structure_id) == list:
            sid = structure_id
        else:
            sid = [structure_id]
        structure_mask = rsp.make_structure_mask(sid, direct_only=False)
        #FK_brain_region_id = structureController.structure_abbreviation_to_id(abbreviation=abbreviation)
        #FK_session_id = annotationSessionController.create_annotation_session(annotation_type=AnnotationType.STRUCTURE_COM, 
        #                                                                        FK_user_id=1, FK_prep_id=animal, FK_brain_region_id=FK_brain_region_id)
        if abbreviation in singular_structures:
            x,y,z = center_of_mass(structure_mask)
            x *= um
            y *= um
            z *= um
            print('Singular',end="\t")
        else:

            if abbreviation.endswith('L'):
                print('Left', end="\t")
                left_side = structure_mask[:,:,0:midpoint]
                right_side = structure_mask[:,:,midpoint:]
                x,y,z = center_of_mass(left_side)
                x *= um
                y *= um
                z *= um
            elif abbreviation.endswith('R'):
                print('Right', end="\t")
                x,y,z = center_of_mass(right_side)
                x *= um
                y *= um
                z = (z + midpoint) * 25
            else:
                print(f'We should not be here abbreviation={abbreviation}')

        print(f'{abbreviation} {x} {y} {z}')
        #com = StructureCOM(source=source, x=x, y=y, z=z, FK_session_id=FK_session_id)
        #brainManager.sqlController.add_row(com)




    """
    for abbreviation, points in structures.items():
        FK_brain_region_id = structureController.structure_abbreviation_to_id(abbreviation=abbreviation)
        #FK_session_id = annotationSessionController.create_annotation_session(annotation_type=AnnotationType.STRUCTURE_COM, 
        #                                                                        FK_user_id=1, FK_prep_id=animal, FK_brain_region_id=FK_brain_region_id)
        FK_session_id = 0
        x,y,z = (p*25 for p in points)
        print(source, FK_brain_region_id, FK_session_id, abbreviation, points, x,y,z)
        #com = StructureCOM(source=source, x=x, y=y, z=z, FK_session_id=FK_session_id)
        #brainManager.sqlController.add_row(com)
    """



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
