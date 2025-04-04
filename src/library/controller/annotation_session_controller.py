from collections import defaultdict
import datetime
import numpy as np
from collections import defaultdict

from library.database_model.brain_region import BrainRegion
from library.database_model.annotation_points import AnnotationLabel, StructureCOM
from library.database_model.annotation_points import AnnotationSession
from library.utilities.utilities_process import M_UM_SCALE, SCALING_FACTOR

FIDUCIAL = 8  # label ID in table annotation_label


class AnnotationSessionController:
    """The class that queries and addes entry to the annotation_session table"""

    def update_session(self, id: int, update_dict: dict):
        """
        Update the table with the given ID using the provided update dictionary.

        Args:
            id (int): The ID of the scan run to update.
            update_dict (dict): A dictionary containing the fields to update and their new values.

        Returns:
            None
        """

        try:
            self.session.query(AnnotationSession).filter(
                AnnotationSession.id == id
            ).update(update_dict)
            self.session.commit()

        except Exception as e:
            print(f"No merge for  {e}")
            self.session.rollback()
    
    def get_labels(self, labels):
        annotation_labels = (
            self.session.query(AnnotationLabel)
            .filter(AnnotationLabel.label.in_(labels))
            .all()
        )
        return annotation_labels


    def get_annotation_session(self, prep_id: str, label_ids: list, annotator_id: int, debug: bool = False) -> AnnotationSession:
        if isinstance(label_ids, int):
            label_ids = [label_ids]
            
        query = (
            self.session.query(AnnotationSession)
            .filter(AnnotationSession.active == True)
            .filter(AnnotationSession.FK_prep_id == prep_id)
            .filter(AnnotationSession.FK_user_id == annotator_id)
            .filter(AnnotationSession.labels.any(AnnotationLabel.id.in_(label_ids)))
            .order_by(AnnotationSession.updated.desc())
        )

        if debug:  # Print the raw SQL query
            print(f'RAW SQL: {query.statement.compile(compile_kwargs={"literal_binds": True})}')
        
        return query.first()


    def get_annotation_label(self, label):

        annotation_label = (
            self.session.query(AnnotationLabel)
            .filter(AnnotationLabel.label == label)
            .first()
        )

        return annotation_label
    
    def create_annotation_session(self, FK_user_id, FK_prep_id, annotation):
        data = AnnotationSession(
            FK_user_id=FK_user_id,
            FK_prep_id=FK_prep_id,
            annotation=annotation,
            created=datetime.datetime.now(),
            active=True,
        )
        self.add_row(data)
        self.session.commit()
        return data.id

    def insert_annotation_with_labels(self, FK_user_id: int, FK_prep_id: int, annotation: dict, labels: list):

        annotation_session = AnnotationSession(
            FK_user_id=FK_user_id,
            FK_prep_id=FK_prep_id,
            annotation=annotation,
            created=datetime.datetime.now(),
            active=True,
        )
        
        # Check if books exist, create them if they don't
        for label in labels:
            annotation_label = self.session.query(AnnotationLabel).filter_by(label=label).first()
            if not annotation_label:
                print("No label found, please fix")
                return
                
            annotation_session.labels.append(annotation_label)
        
        self.session.add(annotation_session)
        self.session.commit()
        return annotation_session.id

    def get_fiducials(self, prep_id, debug: bool = False):
        """Fiducials will be marked on downsampled images. You will need the resolution
        to convert from micrometers back to pixels of the downsampled images.

        :param debug: whether to print the raw SQL query
        """
        fiducials = defaultdict(list)

        # Define query
        query = (
            self.session.query(AnnotationSession)
            .filter(AnnotationSession.active == True)
            .filter(AnnotationSession.FK_prep_id == prep_id)
            .filter(AnnotationSession.labels.any(AnnotationLabel.id.in_([FIDUCIAL])))
            .order_by(AnnotationSession.updated.desc())
        )
        
        annotation_session = query.first()

        if not annotation_session:
            print("No annotation session for this animal was found.")
            return fiducials

        xy_resolution = self.scan_run.resolution
        z_resolution = self.scan_run.zresolution

        try:
            data = annotation_session.annotation["childJsons"]
        except KeyError:
            print("No childJsons key in data")
            return fiducials


        for point in data:
            x, y, z = point["point"]
            x = x * M_UM_SCALE / xy_resolution / SCALING_FACTOR
            y = y * M_UM_SCALE / xy_resolution / SCALING_FACTOR
            section = int(np.round((z * M_UM_SCALE / z_resolution), 2))
            fiducials[section].append((x, y))

        if debug:  # Print the raw SQL query
            print(f'RAW SQL: {str(query.statement.compile(compile_kwargs={"literal_binds": True}))}')

        return fiducials


    def get_com_dictionary(self, prep_id):
        """This returns data in meters, not pixels or micrometers.
        This is the way neuroglancer uses data. If you need
        to convert to pixels, you will need to use the resolution
        of the downsampled images. If you need to convert to micrometers,
        just multiply by M_UM_SCALE.
        The data is also not sorted by key. If you need it sorted,
        use an ordered dictionary.
        annotator ID is being hard coded to 2 for Beth
        """
        annotator_id = 2

        coms = {}
        annotation_sessions = (
            self.session.query(AnnotationSession)
            .filter(AnnotationSession.active == True)
            .filter(AnnotationSession.FK_prep_id == prep_id)
            .filter(AnnotationSession.FK_user_id == annotator_id)
            .filter(AnnotationSession.annotation['type'] == 'point')
            .all()
        )

        if not annotation_sessions:
            return coms

        # first test data to make sure it has the right keys
        for annotation_session in annotation_sessions:
            try:
                data = annotation_session.annotation["point"]
            except KeyError:
                print(f'No data for {annotation_session.FK_prep_id} was found.')
                continue
            
            try:
                label = annotation_session.labels[0].label
            except IndexError:
                continue
            
            x, y, z = data
            coms[label] = [x,y,z]

        return coms


    def get_annotation_volume(self, session_id, scaling_factor=1):
        """
        This returns data in micrometers divided by the scaling_factor provided.
        If you need x,y offsets, you'll need to convert to scale a
        and then do an offset. (e.g., fixing DK78)
        """

        annotation_session = self.session.query(AnnotationSession).get(session_id)
        polygons = defaultdict(list)

        if not annotation_session:
            return polygons
        
        annotation = annotation_session.annotation

        # first test data to make sure it has the right keys
        try:
            data = annotation["childJsons"]
        except KeyError as ke:
            return polygons
        
        for points in data:
            if 'childJsons' not in points:
                return polygons
            for child in points['childJsons']:
                x,y,z = child['pointA']
                x = int(np.round(x * M_UM_SCALE / scaling_factor))
                y = int(np.round(y * M_UM_SCALE / scaling_factor))
                section = int(np.round(z * M_UM_SCALE / scaling_factor))
                polygons[section].append((x,y))

        return polygons
    

    def get_annotation_features(self, prep_id: str, annotator_id: int, debug: bool = False):

        features = defaultdict(list)

        # Define query
        query = (
            self.session.query(AnnotationSession)
            .filter(AnnotationSession.active == True)
            .filter(AnnotationSession.FK_prep_id == prep_id)
            .filter(AnnotationSession.FK_user_id == annotator_id)
            .filter(AnnotationSession.labels.any(AnnotationLabel.id.in_([FIDUCIAL])))
            .order_by(AnnotationSession.updated.desc())
        )
        
        annotation_session = query.first()

        if not annotation_session:
            print("No annotation session for this animal was found.")
            return features

        xy_resolution = self.scan_run.resolution
        z_resolution = self.scan_run.zresolution

        try:
            data = annotation_session.annotation["childJsons"]
        except KeyError:
            return features


        for point in data:
            x, y, z = point["point"]
            x = x * M_UM_SCALE / xy_resolution
            y = y * M_UM_SCALE / xy_resolution
            section = int(np.round((z * M_UM_SCALE / z_resolution), 2))
            if debug:
                print(x, y, section)
            features[section].append((x, y))


        return features




    def get_annotation_by_id(self, session_id):

        annotation_session = self.session.query(AnnotationSession).get(session_id)

        if annotation_session is not None:
            return annotation_session
        else:
            print("No data for this animal was found.")
            return None
