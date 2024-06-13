from sqlalchemy import func

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

    def get_and_correct_multiples(self, scan_run_id, slide_physical_id):
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
        max_index = self.session.query(func.max(SlideCziTif.scene_index)).filter(SlideCziTif.FK_slide_id == master_slide).one()[0] + 1
        if not isinstance(max_index, int):
            print('we should not get here')
            max_index = 3
        for other_slide in slide_physical_ids:
            print(f'Updating slideczitiff set FK_slide_id={master_slide} where FK_slideid={other_slide}')
            try:
                self.session.query(SlideCziTif)\
                    .filter(SlideCziTif.FK_slide_id == other_slide).update({'FK_slide_id': master_slide, 'scene_index': SlideCziTif.scene_index + max_index})
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

