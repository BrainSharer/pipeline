from sqlalchemy import func
from sqlalchemy.orm.exc import NoResultFound

from library.database_model.slide import Slide, SlideCziTif


class SlideCZIToTifController():

    def update_tif(self, id, width, height):
        """Update a TIFF object (row)
        
        :param id: primary key
        :param width: int of width of TIFF  
        :param height: int of height of TIFF  
        """
        
        try:
            self.session.query(SlideCziTif).filter(
                SlideCziTif.id == id).update({'width': width, 'height': height})
            self.session.commit()
        except Exception as e:
            print(f'No merge for  {e}')
            self.session.rollback()


    def get_slide(self, id):
        return self.session.query(Slide).filter(Slide.id == id)
    

    def get_slide_by_physical_id(self, scan_run_id, slide_physical_id):
        slide = None
        try:
            slide = self.session.query(Slide).filter(Slide.scan_run_id == scan_run_id).filter(Slide.slide_physical_id == slide_physical_id).one()
        except NoResultFound as nrf:
            print(f'No slide found for scan_run ID={scan_run_id} and slide physical ID={slide_physical_id}')
            slide = None

        return slide

    def get_and_correct_multiples(self, scan_run_id, slide_physical_id):
        """
        Retrieves slides with the given scan_run_id and slide_physical_id,
        corrects their scene_index values, and sets inactive flag for empty slides.

        Args:
            scan_run_id (int): The ID of the scan run.
            slide_physical_id (int): The physical ID of the slide.

        Returns:
            None
        """
        slide_physical_ids = []
        slide_rows = self.session.query(Slide)\
            .filter(Slide.scan_run_id == scan_run_id)\
            .filter(Slide.slide_physical_id == slide_physical_id)
        for slide_row in slide_rows:
            slide_physical_ids.append(slide_row.id)
        print(f'Slide_physical_ids={slide_physical_ids}')
        master_slide = min(slide_physical_ids)
        print(f'Master slide={master_slide}')
        slide_physical_ids.remove(master_slide)
        print(f'Other slides = {slide_physical_ids}')
        for other_slide in slide_physical_ids:
            print(f'Updating slideczitiff set FK_slide_id={master_slide} where FK_slideid={other_slide}')
            
            try:
                self.session.query(SlideCziTif)\
                    .filter(SlideCziTif.FK_slide_id == other_slide).update({'FK_slide_id': master_slide})
                self.session.commit()
            except Exception as e:
                print(f'No merge for  {e}')
                self.session.rollback()

            # set empty slide to inactive
            try:
                self.session.query(Slide)\
                    .filter(Slide.id == other_slide).update({'active': False})
                self.session.commit()
            except Exception as e:
                print(f'No merge for  {e}')
                self.session.rollback()

