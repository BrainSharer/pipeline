
from library.database_model.annotation_points import AnnotationSession, AnnotationType, MarkedCell

FIDUCIAL = 33 # cell type ID in table cell type

class MarkedCellController():

    def get_fiducials(self, prep_id):
        """Fiducials will be marked on downsampled images. You will need the resolution
        and the scaling factor to convert from micrometers back to pixels of
        the downsampled images.
        """

        row_dict = {}

        annotation_session = self.get_session(prep_id)
        if not annotation_session:
            print('No data for this animal')
            return row_dict
        
        print(f'Annotation session ID: {annotation_session.id}')

        
        rows = self.session.query(MarkedCell).filter(MarkedCell.FK_session_id==annotation_session.id).order_by(MarkedCell.id).all()

        for row in rows:
            row_dict[row.id] = [row.x, row.y, row.z]
        return row_dict

    
    def get_session(self, prep_id):
        session = self.session.query(AnnotationSession)\
            .filter(AnnotationSession.active==True)\
            .filter(AnnotationSession.annotation_type==AnnotationType.MARKED_CELL)\
            .filter(AnnotationSession.FK_prep_id==prep_id)\
            .join(MarkedCell)\
            .filter(MarkedCell.FK_cell_type_id==FIDUCIAL).first()
        return session
