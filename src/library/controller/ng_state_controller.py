from datetime import datetime
from library.database_model.ng_state import NeoroglancerState

class NeoroglancerStateController():
    """Controller class for the neuroglancer_state table
    """
    def __init__(self, session):
        """initiates the controller class
        """
        self.session = session
        

    def insert_ng_state(self, combined_json: str, created='', updated='', fk_lab_id=2, user_date=None, comments="", description=None, fk_user_id=None, readonly=False, public=False, active=True):
        """
        Inserts a new record into the neuroglancer_state table.  If a record with the same comments exists, updates it instead.

        Args:
            combined_json (str): The JSON data to insert (neuroglancer_state).
            created (datetime): Timestamp for when the record was created.
            updated (datetime): Timestamp for when the record was last updated.
            fk_lab_id (int): Foreign key for lab ID (defaults to UCSD)
            user_date (str): Optional user-provided date.
            comments (str): Comments for the record.
            description (str): Optional description of the record.
            fk_user_id (int): Optional foreign key for user ID.
            readonly (bool): Whether the record is read-only.
            public (bool): Whether the record is public.
            active (bool): Whether the record is active.

        Returns:
            NeoroglancerState: The inserted record.
        """
        #TODO: create 'system' account for auto-generated preview states
        #TODO: WOULD BE BETTER TO SET FIELD DEFAULT IN DB
        created = datetime.now()
        updated = datetime.now()
        if not fk_user_id:
            fk_user_id = 37

        try:
            # Check if a record with the same comments already exists
            existing_record = (
                self.session.query(NeoroglancerState)
                .filter_by(comments=comments)
                .first()
            )

            if existing_record:
                existing_record.neuroglancer_state = combined_json
                existing_record.updated = updated
                existing_record.user_date = user_date
                existing_record.description = description
                existing_record.FK_user_id = fk_user_id
                existing_record.readonly = readonly
                existing_record.public = public
                existing_record.active = active

                self.session.commit()
                print(f"Updated existing record with ID: {existing_record.id}")
                return existing_record

            new_record = NeoroglancerState(
                neuroglancer_state=combined_json,
                FK_lab_id=fk_lab_id,
                created=created,
                updated=updated,
                user_date=user_date,
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