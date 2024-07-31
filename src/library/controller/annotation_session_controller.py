from collections import defaultdict
import datetime
import numpy as np

from library.database_model.brain_region import BrainRegion
from library.database_model.annotation_points import AnnotationLabel, StructureCOM
from library.database_model.annotation_points import AnnotationSession
from library.utilities.utilities_process import M_UM_SCALE, SCALING_FACTOR

FIDUCIAL = 8 # label ID in table annotation_label


class AnnotationSessionController():
    """The class that queries and addes entry to the annotation_session table
    """

    def update_session(self, id, update_dict):
        """
        Update the table with the given ID using the provided update dictionary.

        Args:
            id (int): The ID of the scan run to update.
            update_dict (dict): A dictionary containing the fields to update and their new values.

        Returns:
            None
        """

        try:
            self.session.query(AnnotationSession).filter(AnnotationSession.id == id).update(update_dict)
            self.session.commit()

        except Exception as e:
            print(f'No merge for  {e}')
            self.session.rollback()


    
    def get_existing_sessionXXXXXXXXX(self):
        """retruns a list of available session objects that is currently active in the database

        Returns:
            list: list of volume sessions
        """ 
        active_sessions = self.session.query(AnnotationSession) \
                .filter(AnnotationSession.active==True).all()
        return active_sessions
    
    def get_brain_region(self, abbreviation):
        brain_region = self.session.query(BrainRegion) \
                .filter(BrainRegion.abbreviation==abbreviation)\
                .filter(BrainRegion.active==True).one_or_none()
        return brain_region
    

    def get_annotation_session(self, prep_id, label_id, annotator_id):

        annotation_session = self.session.query(AnnotationSession)\
            .filter(AnnotationSession.active==True)\
            .filter(AnnotationSession.FK_prep_id==prep_id)\
            .filter(AnnotationSession.FK_user_id==annotator_id)\
            .filter(AnnotationSession.labels.any(AnnotationLabel.id.in_([label_id])))\
            .order_by(AnnotationSession.updated.desc()).first()
            
        return annotation_session


    def get_annotation_label(self, label):

        annotation_label = self.session.query(AnnotationLabel)\
            .filter(AnnotationLabel.label==label).first()
            
        return annotation_label


    def upsert_structure_com(self, entry):
        """Method to do update/insert. It first checks if there is already an entry. If not,
        it does insert otherwise it updates.
        """
        FK_session_id = entry['FK_session_id']
        annotation_session = self.session.query(AnnotationSession).get(FK_session_id)
        annotation_session.updated = datetime.datetime.now()
        structure_com = self.session.query(StructureCOM).filter(StructureCOM.FK_session_id==FK_session_id).first()
        if structure_com is None:
            data = StructureCOM(
                source=entry['source'],
                FK_session_id = FK_session_id,
                x = entry['x'],
                y = entry['y'],
                z = entry['z']
            )
            self.add_row(data)
        else:
            self.session.query(StructureCOM)\
                .filter(StructureCOM.FK_session_id == FK_session_id).update(entry)
            self.session.commit()

    def create_annotation_session(self, annotation_type, FK_user_id, FK_prep_id, FK_brain_region_id):
        data = AnnotationSession(
            annotation_type=annotation_type,
            FK_user_id=FK_user_id,
            FK_prep_id=FK_prep_id,
            FK_brain_region_id=FK_brain_region_id,
            created=datetime.datetime.now(),
            active=True
        )
        self.add_row(data)
        self.session.commit()
        return data.id


    def get_fiducials(self, prep_id):
        """Fiducials will be marked on downsampled images. You will need the resolution
        to convert from micrometers back to pixels of the downsampled images.
        """
        
        fiducials = defaultdict(list)
        annotation_session = self.session.query(AnnotationSession)\
            .filter(AnnotationSession.active==True)\
            .filter(AnnotationSession.FK_prep_id==prep_id)\
            .filter(AnnotationSession.labels.any(AnnotationLabel.id.in_([FIDUCIAL]))).first()

        if not annotation_session:
            print('No fiducial data for this animal was found.')
            return fiducials
        
        xy_resolution = self.scan_run.resolution
        z_resolution = self.scan_run.zresolution


        #data = annotation_session.annotation['childJsons']
        #print(f'data: {data}  type: {type(data)}')
        #return {}

        # first test data to make sure it has the right keys    
        try:
            data = annotation_session.annotation['childJsons']
        except KeyError:
            print("No childJsons key in data")
            return fiducials
        
        for point in data:
            x,y,z = point['point']
            x = x * M_UM_SCALE / xy_resolution / SCALING_FACTOR
            y = y * M_UM_SCALE / xy_resolution / SCALING_FACTOR
            section = int( np.round((z * M_UM_SCALE / z_resolution),2) )
            print(x,y,section)
            fiducials[section].append((x,y))

        return fiducials


    def get_annotation(self, session_id):

        #annotation_session = self.session.query(AnnotationSession).get(session_id)

        annotation_session = self.session.get(AnnotationSession, session_id)

        if not annotation_session:
            print('No fiducial data for this animal was found.')
            return
        
        xy_resolution = self.scan_run.resolution
        z_resolution = self.scan_run.zresolution



        # first test data to make sure it has the right keys    
        try:
            data = annotation_session.annotation['childJsons']
        except KeyError:
            print("No childJsons key in data")
        
        for point in data:
            for k, v in point.items():
                print(f'{k}')
            #print(f'type: {type(point)}')
            continue
            x,y,z = point['point']
            x = x * M_UM_SCALE / xy_resolution / SCALING_FACTOR
            y = y * M_UM_SCALE / xy_resolution / SCALING_FACTOR
            section = int( np.round((z * M_UM_SCALE / z_resolution),2) )
            print(x,y,section)



