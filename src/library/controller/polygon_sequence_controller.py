import pandas as pd

from library.controller.sql_controller import SqlController
from library.database_model.annotation_points import PolygonSequence
from library.database_model.annotation_points import AnnotationSession



class PolygonSequenceController(SqlController):
    """The class that queries and addes entry to the PolygonSequence table
    """
        
    def get_available_volumes(self):
        active_sessions = self.get_available_volumes_sessions()
        information = [[i.FK_prep_id,i.user.first_name,i.brain_region.abbreviation] for i in active_sessions]
        return information
    
    def get_volume(self,prep_id, annotator_id, structure_id):
        """Returns the points in a brain region volume

        Args:
            prep_id (str): Animal ID
            annotator_id (int): Annotator ID
            structure_id (int): Structure ID

        Returns:
            dictionary: points in a volume grouped by polygon.
        """        
        #Polygons must be ordered by section(z), then point ordering
        annotation_session = self.session.query(AnnotationSession)\
            .filter(AnnotationSession.FK_prep_id==prep_id)\
            .filter(AnnotationSession.FK_user_id==annotator_id)\
            .filter(AnnotationSession.FK_brain_region_id==structure_id)\
            .filter(AnnotationSession.active==1).first()
        if annotation_session is None:
            df = pd.DataFrame()
            return df
        
        volume_points = self.session.query(PolygonSequence)\
            .filter(PolygonSequence.FK_session_id==annotation_session.id)\
                .order_by(PolygonSequence.z)\
                .order_by(PolygonSequence.point_order)\
                    .all()
        volume = {}
        volume['coordinate']=[[i.x,i.y,i.z] for i in volume_points]
        volume['point_ordering']=[i.point_order for i in volume_points]
        volume['polygon_ordering']=[i.polygon_index for i in volume_points]
        volume = pd.DataFrame(volume)
        return volume
    
    def get_available_volumes_sessions(self):
        """returns a list of available session objects that is currently active in the database
        ID=54 is the ID for polygon in the brain_region table

        Returns:
            list: list of volume sessions
        """        
        active_sessions = self.session.query(AnnotationSession)\
            .filter(AnnotationSession.annotation_type==AnnotationType.POLYGON_SEQUENCE)\
            .filter(AnnotationSession.active==1)\
            .filter(AnnotationSession.FK_brain_region_id != 54)\
            .all()
        return active_sessions
    
    def get_data_per_session(self, session_id):
        """returns the data for a session

        Args:
            session_id (int): session id

        Returns:
            list: list of StructureCOM objects
        """
        return self.session.query(PolygonSequence).filter(PolygonSequence.FK_session_id == session_id)\
                .order_by(PolygonSequence.z)\
                .order_by(PolygonSequence.point_order)\
            .all()
