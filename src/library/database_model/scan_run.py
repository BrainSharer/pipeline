from sqlalchemy import Column, Integer, Date, ForeignKey, Enum, String, Float, Boolean
from sqlalchemy.orm import relationship

from library.database_model.atlas_model import Base, AtlasModel
NO_MASK = 0
FULL_MASK = 1
BOTTOM_MASK = 2
FULL_MASK_NO_CROP = 3
IMAGE_MASK = {0: 'NO_MASK', 1: 'FULL_MASK', 2: 'BOTTOM_MASK', 3: 'FULL_MASK_NO_CROP'}


class ScanRun(Base, AtlasModel):  
    """This class describes the blueprint of a scan. Each animal will usually 
    have just one scan run, but they can have more than one. Information in 
    this table is used extensively throughout the pre-processing 
    """

    __tablename__ = 'scan_run'

    id =  Column(Integer, primary_key=True, nullable=False)
    FK_prep_id = Column("FK_prep_id", String, ForeignKey('animal.prep_id'), nullable=False)
    machine = Column(Enum("Axioscan I", "Axioscan II", "Nanozoomer"))
    objective = Column(Enum("60X", "40X", "20X", "10X"))
    resolution = Column(Float, default=0)
    zresolution = Column(Float, default=20)
    number_of_slides = Column(Integer, default=0)
    scan_date = Column(Date)
    file_type = Column(Enum("CZI", "JPEG2000", "NDPI", "NGR"))
    scenes_per_slide = Column(Enum("1", "2", "3", "4", "5", "6"))
    section_schema = Column(Enum("L to R", "R to L"))
    channels_per_scene = Column(Integer, default=3)
    slide_folder_path = Column(String)
    converted_status = Column(Enum("not started", "converted", "converting", "error"))
    ch_1_filter_set = Column(Enum("68", "47", "38", "46", "63", "64", "50"))
    ch_2_filter_set = Column(Enum("68", "47", "38", "46", "63", "64", "50"))
    ch_3_filter_set = Column(Enum("68", "47", "38", "46", "63", "64", "50"))
    ch_4_filter_set = Column(Enum("68", "47", "38", "46", "63", "64", "50"))
    width = Column(Integer, default=0, nullable=False)
    height = Column(Integer, default=0, nullable=False)
    rotation = Column(Integer, default=0, nullable=False)
    flip = Column(Enum("none", "flip", "flop"))
    mask = Column(Integer, default=FULL_MASK, nullable=False)
    comments = Column(String)

    slides = relationship('Slide', lazy=True)


    def __repr__(self):
        return "ScanRun(prep_id='%s', scan_id='%s'" % (self.prep_id, self.id)


