
from collections import defaultdict
from library.database_model.annotation_points import AnnotationSession, AnnotationType, MarkedCell

FIDUCIAL = 33 # cell type ID in table cell type

class MarkedCellController():

    def get_fiducials(self, prep_id):
        """Fiducials will be marked on downsampled images. You will need the resolution
        to convert from micrometers back to pixels of the downsampled images.
        """

        fiducials = defaultdict(list)
        annotation_session = self.get_session(prep_id)
        if not annotation_session:
            print('No data for this animal')
            return fiducials
        

        xy_resolution = self.scan_run.resolution
        z_resolution = self.scan_run.zresolution
        
        rows = self.session.query(MarkedCell).filter(MarkedCell.FK_session_id==annotation_session.id)\
            .order_by(MarkedCell.z, MarkedCell.x, MarkedCell.y)\
            .all()

        for row in rows:
            x = row.x / xy_resolution
            y = row.y / xy_resolution
            section = row.z / z_resolution
            fiducials[section].append((x,y))

        return fiducials

    
    def get_session(self, prep_id):
        session = self.session.query(AnnotationSession)\
            .filter(AnnotationSession.active==True)\
            .filter(AnnotationSession.annotation_type==AnnotationType.MARKED_CELL)\
            .filter(AnnotationSession.FK_prep_id==prep_id)\
            .join(MarkedCell)\
            .filter(MarkedCell.FK_cell_type_id==FIDUCIAL).first()
        return session
