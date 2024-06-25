import datetime

from library.controller.sql_controller import SqlController
from library.database_model.brain_region import BrainRegion
from library.database_model.annotation_points import StructureCOM
from library.database_model.annotation_points import AnnotationSession, AnnotationType



class AnnotationSessionController(SqlController):
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

    
    def get_existing_session(self, annotation_type=AnnotationType.STRUCTURE_COM):
        """retruns a list of available session objects that is currently active in the database

        Returns:
            list: list of volume sessions
        """ 
        active_sessions = self.session.query(AnnotationSession) \
                .filter(AnnotationSession.annotation_type==annotation_type)\
                .filter(AnnotationSession.active==True).all()
        return active_sessions
    
    def get_brain_region(self, abbreviation):
        brain_region = self.session.query(BrainRegion) \
                .filter(BrainRegion.abbreviation==abbreviation)\
                .filter(BrainRegion.active==True).one_or_none()
        return brain_region
    

    def get_annotation_session(self, prep_id, brain_region_id, annotator_id, annotation_type):
        annotation_session = self.session.query(AnnotationSession).filter(AnnotationSession.active==True)\
            .filter(AnnotationSession.annotation_type==annotation_type)\
            .filter(AnnotationSession.FK_prep_id==prep_id)\
            .filter(AnnotationSession.FK_brain_region_id==brain_region_id)\
            .filter(AnnotationSession.FK_user_id==annotator_id)\
            .order_by(AnnotationSession.created.desc()).first()

        if annotation_session is None:
            annotation_session = AnnotationSession(
                FK_prep_id=prep_id,
                FK_user_id=annotator_id,
                FK_brain_region_id=brain_region_id,
                annotation_type=annotation_type,
                active=True,
                created=datetime.datetime.now())

            self.session.add(annotation_session)
            self.session.commit()
            self.session.refresh(annotation_session)
        else:
            annotation_session.updated=datetime.datetime.now()
            self.session.refresh(annotation_session)

        return annotation_session

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

