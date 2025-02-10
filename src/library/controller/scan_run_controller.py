import sys

from library.database_model.scan_run import ScanRun
from library.database_model.slide import SlideCziTif
from library.database_model.slide import Slide
from library.utilities.utilities_process import SCALING_FACTOR

class ScanRunController():
    """Controller for the scan_run table"""

    def __init__(self, session):
        """initiates the controller class
        """
        self.session = session

    def get_scan_run(self, animal):
        """Check to see if there is a row for this animal in the
        scan run table

        :param animal: the animal (AKA primary key)
        :return scan run object: one object (row)
        """

        search_dictionary = dict(FK_prep_id=animal)
        return self.get_row(search_dictionary, ScanRun)


    def update_width_height(self, id, width, height, scaling_factor=SCALING_FACTOR):
        def roundtochunk(x):
            """I think this needs to be bumped up to 1024"""
            ROUNDUPTO = 64
            return ROUNDUPTO * round(x/ROUNDUPTO)        

        scan_run = self.session.query(ScanRun).filter(ScanRun.id == id).first()
        rotation = scan_run.rotation

        width *= scaling_factor
        height *= scaling_factor
        width_buffer = int(width * 0.005)
        height_buffer = int(height * 0.005)
        width = roundtochunk(width + width_buffer)
        height = roundtochunk(height + height_buffer)
        if (rotation % 2) == 0:
            update_dict = {'width': width, 'height': height}
        else:
            update_dict = {'width': height, 'height': width}

            
        try:
            self.session.query(ScanRun).filter(ScanRun.id == id).update(update_dict)
            self.session.commit()
        except Exception as e:
            print(f'No merge for  {e}')
            self.session.rollback()

        self.scan_run = self.session.query(ScanRun)\
            .filter(ScanRun.id == id).one()

    def update_scan_run(self, id, update_dict):
        """
        Update the scan run with the given ID using the provided update dictionary.

        Args:
            id (int): The ID of the scan run to update.
            update_dict (dict): A dictionary containing the fields to update and their new values.

        Returns:
            None
        """

        try:
            self.session.query(ScanRun).filter(ScanRun.id == id).update(update_dict)
            self.session.commit()

        except Exception as e:
            print(f'No merge for  {e}')
            self.session.rollback()
        self.scan_run = self.session.query(ScanRun)\
            .filter(ScanRun.id == id).one()


    def get_channels(self, FK_prep_id):
        """Update the scan run table with safe and good values for the width and height

        :param id: integer primary key of scan run table
        """
        scan_run = self.session.query(ScanRun).filter(ScanRun.FK_prep_id == FK_prep_id).first()
        channel_count = int(scan_run.channels_per_scene)
        channels = []
        channels.append(scan_run.channel1_name)
        if channel_count > 1:
            channels.append(scan_run.channel2_name)
        if channel_count > 2:
            channels.append(scan_run.channel3_name)
        if channel_count > 3:
            channels.append(scan_run.channel4_name)

        return channels


