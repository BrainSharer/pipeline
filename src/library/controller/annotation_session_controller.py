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
            self.session.query(AnnotationSession).filter(
                AnnotationSession.id == id
            ).update(update_dict)
            self.session.commit()

        except Exception as e:
            print(f"No merge for  {e}")
            self.session.rollback()

    def get_brain_region(self, abbreviation):
        brain_region = (
            self.session.query(BrainRegion)
            .filter(BrainRegion.abbreviation == abbreviation)
            .filter(BrainRegion.active == True)
            .one_or_none()
        )
        return brain_region
    
    def get_labels(self, labels):
        annotation_labels = (
            self.session.query(AnnotationLabel)
            .filter(AnnotationLabel.label.in_(labels))
            .all()
        )
        return annotation_labels


    def get_annotation_session(self, prep_id: str, label_ids: list, annotator_id: int):
        if isinstance(label_ids, int):
            label_ids = [label_ids]
        annotation_session = (
            self.session.query(AnnotationSession)
            .filter(AnnotationSession.active == True)
            .filter(AnnotationSession.FK_prep_id == prep_id)
            .filter(AnnotationSession.FK_user_id == annotator_id)
            .filter(AnnotationSession.labels.any(AnnotationLabel.id.in_(label_ids)))
            .order_by(AnnotationSession.updated.desc())
            .first()
        )

        return annotation_session

    def get_annotation_label(self, label):

        annotation_label = (
            self.session.query(AnnotationLabel)
            .filter(AnnotationLabel.label == label)
            .first()
        )

        return annotation_label

    def upsert_structure_com(self, entry):
        """Method to do update/insert. It first checks if there is already an entry. If not,
        it does insert otherwise it updates.
        """
        FK_session_id = entry["FK_session_id"]
        annotation_session = self.session.query(AnnotationSession).get(FK_session_id)
        annotation_session.updated = datetime.datetime.now()
        structure_com = (
            self.session.query(StructureCOM)
            .filter(StructureCOM.FK_session_id == FK_session_id)
            .first()
        )
        if structure_com is None:
            data = StructureCOM(
                source=entry["source"],
                FK_session_id=FK_session_id,
                x=entry["x"],
                y=entry["y"],
                z=entry["z"],
            )
            self.add_row(data)
        else:
            self.session.query(StructureCOM).filter(
                StructureCOM.FK_session_id == FK_session_id
            ).update(entry)
            self.session.commit()

    def create_annotation_sessionDEPRECATED(self, annotation_type, FK_user_id, FK_prep_id, FK_brain_region_id):
        data = AnnotationSession(
            annotation_type=annotation_type,
            FK_user_id=FK_user_id,
            FK_prep_id=FK_prep_id,
            FK_brain_region_id=FK_brain_region_id,
            created=datetime.datetime.now(),
            active=True,
        )
        self.add_row(data)
        self.session.commit()
        return data.id
    
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

    def insert_annotation_with_labels(self, FK_user_id, FK_prep_id, annotation, labels):

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


    def get_com_dictionary(self, prep_id, annotator_id):

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
            print("No data for this animal was found.")
            return coms

        xy_resolution = self.scan_run.resolution
        z_resolution = self.scan_run.zresolution


        # first test data to make sure it has the right keys
        for annotation_session in annotation_sessions:
            try:
                data = annotation_session.annotation["point"]
            except KeyError:
                print("No childJsons key in data")
                return coms

            label = annotation_session.labels[0].label

            x, y, z = data
            x = x * M_UM_SCALE / xy_resolution / SCALING_FACTOR
            y = y * M_UM_SCALE / xy_resolution / SCALING_FACTOR
            z = z * M_UM_SCALE / z_resolution
            section = int(np.round((z), 2))
            #print(label, x,y,z, section)
            coms[label] = [x,y,z]

        return coms


    def get_annotation_volume(self, session_id):

        annotation_session = self.session.query(AnnotationSession).get(session_id)
        polygons = defaultdict(list)

        if not annotation_session:
            print("No data for this animal was found.")
            return
        
        annotation = annotation_session.annotation

        xy_resolution = self.scan_run.resolution
        z_resolution = self.scan_run.zresolution

        # first test data to make sure it has the right keys
        try:
            data = annotation["childJsons"]
        except KeyError as ke:
            print("No childJsons key in data")
            print(f"Error: {ke}")
            return
        
        x_offset = 2200 // 32
        y_offset = 5070 // 32
        x_offset = 0
        y_offset = 65

        for points in data:
            if 'childJsons' not in points:
                print('No childJsons in points')
                return
            for child in points['childJsons']:
                x,y,z = child['pointA']
                x = int(np.round(x * M_UM_SCALE / xy_resolution / SCALING_FACTOR) - x_offset)
                y = int(np.round(y * M_UM_SCALE / xy_resolution / SCALING_FACTOR) - y_offset)
                section = int(np.round((z * M_UM_SCALE / z_resolution), 2))
                #print(x, y, section)
                polygons[section].append((x,y))

        return polygons


    def get_annotation_by_id(self, session_id):

        annotation_session = self.session.query(AnnotationSession).get(session_id)

        if annotation_session is not None:
            return annotation_session
        else:
            print("No data for this animal was found.")
            return None
