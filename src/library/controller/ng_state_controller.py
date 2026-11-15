from collections import defaultdict
from datetime import datetime
import json

import numpy as np
from library.database_model.ng_state import NeuroglancerState
from library.utilities.utilities_process import M_UM_SCALE

class NeuroglancerStateController():
    """Controller class for the neuroglancer_state table
    """
    def __init__(self, session):
        """initiates the controller class
        """
        self.session = session

    def fetch_polygons_from_state_id(self, state_id: int, layer_name: str = None):
        """
        Fetch polygons from the database using the provided state_id.
        """
        state = None
        try:
            state = self.session.query(NeuroglancerState).filter_by(id=state_id).first()
            if not state:
                print(f"No record found for state_id: {state_id}")
                return None
        except Exception as e:
            print(f"Error fetching polygons for state_id {state_id}: {e}")
            return None

        xy_resolution = self.scan_run.resolution
        z_resolution = self.scan_run.zresolution
        print(f"database resolutions: {xy_resolution}, {z_resolution}")
        neuroglancer_state = json.loads(state.neuroglancer_state)
        resolutions = neuroglancer_state.get('dimensions', [])
        print(f"Fetched neuroglancer_state for state_id {state_id}: dimensions: {resolutions}, layers: {len(neuroglancer_state.get('layers', []))}")
        if len(resolutions) > 0:
            xy_resolution = resolutions.get('x', 1)[0] * M_UM_SCALE
            z_resolution = resolutions.get('z', 1)[0] * M_UM_SCALE
        # Index all annotations in one pass.
        volume_layers = []
        polygons_by_id = {}
        polygon_ids_by_volume = defaultdict(list)
        lines_by_polygon = defaultdict(list)
        polygons_by_section = {}
        for layer in neuroglancer_state.get("layers", []):
            if layer.get("type") != "annotation":
                continue
            if layer_name and layer.get("name") != layer_name:
                print(f'Skipping layer {layer.get("name")} because it does not match the specified layer_name {layer_name}')
                continue
            print(f"Processing layer: {layer.get('name', 'Unnamed Layer')} with type: {layer.get('type')}")

            layer_name = layer.get("name", "")
            annotations = layer.get("annotations") or []
            description = "NA"
            for annotation in annotations:
                annotation_type = annotation.get("type")

                if annotation_type == "volume" and annotation.get("id") is not None:
                    volume_id = annotation["id"]
                    description = annotation.get("description", "NA")
                    volume_layers.append((layer_name, volume_id, description))

                elif annotation_type == "polygon":
                    polygon_id = annotation.get("id")
                    parent_id = annotation.get("parentAnnotationId")

                    if polygon_id is not None:
                        polygons_by_id[polygon_id] = annotation

                        if parent_id is not None:
                            polygon_ids_by_volume[parent_id].append(polygon_id)

                elif annotation_type == "line":
                    parent_id = annotation.get("parentAnnotationId")
                    point = annotation.get("pointA")

                    if (
                        parent_id is not None
                        and point is not None
                        and len(point) >= 3
                    ):
                        lines_by_polygon[parent_id].append(point)

                else:
                    print(f"Skipping annotation type '{annotation_type}' in layer '{layer_name}'")
                    continue

        # Process only polygons belonging to each volume.
        for layer_name, volume_id, description in volume_layers:
            for polygon_id in polygon_ids_by_volume.get(volume_id, []):
                lines = lines_by_polygon.get(polygon_id, [])

                if not lines:
                    continue

                line_array = np.asarray(lines, dtype=np.float64)

                polygon_xy = line_array[:, :2]
                section_value = line_array[0, 2]
                section_number = int(section_value - 0.5)
                polygons_by_section[section_number] = polygon_xy

        return polygons_by_section
        

    def insert_ng_state(self, combined_json: str, fk_prep_id: str, created='', updated='', fk_lab_id=2, comments: str = "", description=None, fk_user_id=None, readonly=False, public=False, active=True):
        """
        Inserts a new record into the neuroglancer_state table.  If a record with the same comments exists, updates it instead.

        Args:
            combined_json (str): The JSON data to insert (neuroglancer_state).
            created (datetime): Timestamp for when the record was created.
            updated (datetime): Timestamp for when the record was last updated.
            fk_lab_id (int): Foreign key for lab ID (defaults to UCSD)
            comments (str): Comments for the record.
            description (str): Optional description of the record.
            fk_user_id (int): Optional foreign key for user ID.
            readonly (bool): Whether the record is read-only.
            public (bool): Whether the record is public.
            active (bool): Whether the record is active.

        Returns:
            NeuroglancerState: The inserted record.
        """
        #TODO: create 'system' account for auto-generated preview states
        #TODO: WOULD BE BETTER TO SET FIELD DEFAULT IN DB
        created = datetime.now()
        updated = datetime.now()
        print(f"adding neuroglancer state for: {fk_prep_id}")
        if not fk_user_id:
            fk_user_id = 37

        try:
            # if ng_id: #ATTEMPT TO AUTO-DETECT EXISTING
            #     # Check if a record with the same comments already exists
            #     existing_record = (
            #         self.session.query(NeuroglancerState)
            #         .filter_by(id=ng_id)
            #         .first()
            #     )

            #     if existing_record:
            #         existing_record.neuroglancer_state = combined_json
            #         existing_record.updated = updated
            #         existing_record.description = description
            #         existing_record.FK_user_id = fk_user_id
            #         existing_record.readonly = readonly
            #         existing_record.public = public
            #         existing_record.active = active

            #         self.session.commit()
            #         print(f"Updated existing record with ID: {existing_record.id}")
            #         return existing_record
            # else:

            #Always create new state; user will need to manually delete unneeded states
            #5-AUG-2025
            new_record = NeuroglancerState(
                neuroglancer_state=combined_json,
                FK_lab_id=fk_lab_id,
                FK_prep_id=fk_prep_id,
                created=created,
                updated=updated,
                comments=comments,
                description=description,
                FK_user_id=fk_user_id,
                readonly=readonly,
                public=public,
                active=active
            )

            self.session.add(new_record)
            self.session.commit()

            print(f"New record inserted with ID: {new_record.id}")
            return new_record
        except Exception as e:
            # Rollback in case of an error
            self.session.rollback()
            print(f"Error inserting new neuroglancer_state: {e}")
            return None