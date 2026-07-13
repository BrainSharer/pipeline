import pandas as pd

from library.controller.sql_controller import SqlController
from library.database_model.annotation_points import AnnotationSession



class PolygonSequenceController(SqlController):
    """The class that queries and addes entry to the PolygonSequence table
    """
        
    def get_available_volumes(self):
        active_sessions = self.get_available_volumes_sessions()
        information = [[i.FK_prep_id,i.user.first_name,i.brain_region.abbreviation] for i in active_sessions]
        return information
    
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
