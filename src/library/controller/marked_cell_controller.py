from collections import defaultdict

from library.controller.sql_controller import SqlController
from library.database_model.annotation_points import AnnotationSession, AnnotationType, MarkedCell


class MarkedCellController(SqlController):

    
    def get_session(self, prep_id):
        session = self.session.query(AnnotationSession)\
            .filter(AnnotationSession.active==True)\
            .filter(AnnotationSession.annotation_type==AnnotationType.MARKED_CELL)\
            .filter(AnnotationSession.FK_prep_id==prep_id)\
            .join(MarkedCell)\
            .filter(MarkedCell.FK_cell_type_id==FIDUCIAL).first()
        return session

    def get_data_per_session(self, session_id):
        """returns the data for a session

        Args:
            session_id (int): session id

        Returns:
            list: list of StructureCOM objects
        """
        return self.session.query(MarkedCell).filter(MarkedCell.FK_session_id == session_id).all()
