from sqlalchemy import func

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
        self.rescan_number = 0

    def get_scan_run(self, animal):
        """Check to see if there is a row for this animal in the
        scan run table

        :param animal: the animal (AKA primary key)
        :return scan run object: one object (row)
        """

        search_dictionary = dict(FK_prep_id=animal, rescan_number=self.rescan_number)
        return self.get_row(search_dictionary, ScanRun)

    def update_scanrun(self, id):
        """Update the scan run table with safe and good values for the width and height

        :param id: integer primary key of scan run table
        """
        scan_run = self.session.query(ScanRun).filter(ScanRun.id == id).first()
        rotation = scan_run.rotation
        width = self.session.query(func.max(SlideCziTif.width)).join(Slide).join(ScanRun)\
            .filter(SlideCziTif.active == True) \
            .filter(ScanRun.id == id).scalar()
        height = self.session.query(func.max(SlideCziTif.height)).join(Slide).join(ScanRun)\
            .filter(SlideCziTif.active == True) \
            .filter(ScanRun.id == id).scalar()
        SAFEMAX = 10000
        LITTLE_BIT_MORE = 500
        # just to be safe, we don't want to update numbers that aren't realistic
        print(f'Found max file size with width={width} height: {height}')
        if height > SAFEMAX and width > SAFEMAX:
            height = round(height, -3)
            width = round(width, -3)
            height += LITTLE_BIT_MORE
            width += LITTLE_BIT_MORE
            # width and height get flipped when there is a rotation.
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


    def update_cropped_data(self, id, width, height):
        """Update the scan run table with safe and good values for the width and height

        :param id: integer primary key of scan run table
        """
        scan_run = self.session.query(ScanRun).filter(ScanRun.id == id).first()
        rotation = scan_run.rotation
        width *= SCALING_FACTOR
        height *= SCALING_FACTOR
        SAFEMAX = 10000
        LITTLE_BIT_MORE = 500
        # just to be safe, we don't want to update numbers that aren't realistic
        print(f'Found max file size with width={width} height: {height}')
        if height > SAFEMAX and width > SAFEMAX:
            height = round(height, -3)
            width = round(width, -3)
            height += LITTLE_BIT_MORE
            width += LITTLE_BIT_MORE
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