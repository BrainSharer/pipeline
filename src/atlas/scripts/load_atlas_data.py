import sys
from pathlib import Path
import numpy as np
import pymysql
from sqlalchemy import exc

from scipy.ndimage import center_of_mass
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from collections import defaultdict
import cv2


PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.registration.brain_structure_manager import BrainStructureManager
from library.controller.structure_com_controller import StructureCOMController
from library.controller.annotation_session_controller import AnnotationSessionController
from library.database_model.annotation_points import PolygonSequence, AnnotationType, StructureCOM
from library.utilities.atlas import allen_structures, singular_structures
from library.controller.polygon_sequence_controller import PolygonSequenceController
from library.utilities.utilities_process import SCALING_FACTOR


def update_coms(debug=False):
    
    animals = ['MD585', 'MD589', 'MD594']
    for animal in animals:
        brainManager = BrainStructureManager(animal)
        polygon = PolygonSequenceController(animal=animal)
        controller = StructureCOMController(animal)
        structures = controller.get_structures()
        # get transformation at um 
        allen_um = 25
                
        for structure in structures:
            #if structure.id not in (666,33):
            #    continue
            polygons = defaultdict(list)
            df = polygon.get_volume(animal, 1, structure.id)
            if df.empty:
                continue;

            for _, row in df.iterrows():
                x = row['coordinate'][0] 
                y = row['coordinate'][1] 
                z = row['coordinate'][2]
                # scale transformed points to 25um
                x /= allen_um
                y /= allen_um
                z /= allen_um
                xy = (x, y)
                section = int(np.round(z))
                polygons[section].append(xy)

            color = 1 # on/off
            origin, section_size = brainManager.get_origin_and_section_size(polygons)

            volume = []
            for _, contour_points in polygons.items():
                vertices = np.array(contour_points)
                # subtract origin so the array starts drawing in the upper top left
                vertices = np.array(contour_points) - origin[:2]
                contour_points = (vertices).astype(np.int32)
                volume_slice = np.zeros(section_size, dtype=np.uint8)
                volume_slice = cv2.polylines(volume_slice, [contour_points], isClosed=True, color=color, thickness=1)
                volume_slice = cv2.fillPoly(volume_slice, pts=[contour_points], color=color)
                volume.append(volume_slice)
            volume = np.array(volume).astype(np.bool8)
            volume = np.swapaxes(volume,0,2)
            com = center_of_mass(volume)
            com += origin
            brainManager.update_com(com, structure.id)
            print(animal, structure.abbreviation, origin, com, section_size, end="\t")
            print(volume.dtype, volume.shape, end="\t")
            ids, counts = np.unique(volume, return_counts=True)
            print(ids, counts)


def load_com():
    animal = 'Allen'
    mcc = MouseConnectivityCache(resolution=25)
    rsp = mcc.get_reference_space()
    print('Shape of entire brain', rsp.annotation.shape)
    midpoint = int(rsp.annotation.shape[2] / 2)
    print('Mid z', midpoint)
    brainManager = BrainStructureManager(animal)
    source = 'MANUAL'
    annotationSessionController = AnnotationSessionController(animal)
    structureController = StructureCOMController(animal)
    # Pn looks like one mass in Allen
    for abbreviation, structure_id in allen_structures.items():
        if type(structure_id) == list:
            sid = structure_id
        else:
            sid = [structure_id]
        structure_mask = rsp.make_structure_mask(sid, direct_only=False)
        FK_brain_region_id = structureController.structure_abbreviation_to_id(abbreviation=abbreviation)
        FK_session_id = annotationSessionController.create_annotation_session(annotation_type=AnnotationType.STRUCTURE_COM, 
                                                                                FK_user_id=1, FK_prep_id=animal, FK_brain_region_id=FK_brain_region_id)
        if abbreviation in singular_structures:
            x,y,z = center_of_mass(structure_mask)
            mins = np.where(structure_mask>0)
            minx = min(mins[0])
            miny = min(mins[1])
            minz = min(mins[2])            
            x *= 25
            y *= 25
            z *= 25
            print('Singular',end="\t")
        else:

            if abbreviation.endswith('L'):
                print('Left', end="\t")
                left_side = structure_mask[:,:,0:midpoint]
                right_side = structure_mask[:,:,midpoint:]
                x,y,z = center_of_mass(left_side)
                mins = np.where(structure_mask>0)
                minx = min(mins[0])
                miny = min(mins[1])
                minz = min(mins[2])            

                x *= 25
                y *= 25
                z *= 25
            elif abbreviation.endswith('R'):
                print('Right', end="\t")
                x,y,z = center_of_mass(right_side)
                x *= 25
                y *= 25
                z = (z + midpoint) * 25
                mins = np.where(structure_mask>0)
                minx = min(mins[0])
                miny = min(mins[1])
                minz = min(mins[2]) + midpoint           
            else:
                print(f'We should not be here abbreviation={abbreviation}')

        print(f'{abbreviation} {FK_brain_region_id} {x} {y} {z}')
        com = StructureCOM(source=source, x=x, y=y, z=z, FK_session_id=FK_session_id, minx=minx, miny=miny, minz=minz)
        brainManager.sqlController.add_row(com)




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
    annotationSessionController = AnnotationSessionController(animal)
    structureController = StructureCOMController(animal)
    xy_resolution = brainManager.sqlController.scan_run.resolution
    zresolution = brainManager.sqlController.scan_run.zresolution
    source = 'NA'
    for abbreviation, v in brainManager.aligned_contours.items():
        FK_brain_region_id = structureController.structure_abbreviation_to_id(abbreviation=abbreviation)
        annotation_session = annotationSessionController.get_annotation_session(prep_id=animal, brain_region_id=FK_brain_region_id, 
                                                                            annotator_id=1, annotation_type=AnnotationType.POLYGON_SEQUENCE)
        for section, vertices in v.items():
            polygon_index = int(section)
            point_order = 1
            z = float(int(section) * zresolution)
            vlist = []
            step = int(len(vertices)/ 100)
            #print(f'abbrev={abbreviation} section={section} len vertices={len(vertices)} step={step}')
            #continue
            for vertex in range(0, len(vertices), step):
                x = vertices[vertex][0]
                y = vertices[vertex][1]
                
                x = x * SCALING_FACTOR * xy_resolution
                y = y * SCALING_FACTOR * xy_resolution
                #print(source, x, y, z, polygon_index, point_order, FK_session_id)
                polygon_sequence = PolygonSequence(x=x, y=y, z=z, source=source, 
                                                polygon_index=polygon_index, point_order=point_order, FK_session_id=annotation_session.id)
                
                point_order += 1
                vlist.append(polygon_sequence)

            try:
                brainManager.sqlController.session.bulk_save_objects(vlist)
                brainManager.sqlController.session.commit()
            except pymysql.err.IntegrityError as e:
                brainManager.sqlController.session.rollback()
            except exc.IntegrityError as e:
                brainManager.sqlController.session.rollback()
            except Exception as e:
                brainManager.sqlController.session.rollback()



if __name__ == '__main__':
    animals = ['MD585', 'MD589', 'MD594']
    for animal in animals:
        load_foundation_brain_polygon_sequences(animal)
    #load_com()
    #update_coms()
