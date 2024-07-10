from sqlalchemy import Column, String, Integer, Boolean, Float, ForeignKey, Enum
from library.database_model.atlas_model import Base, AtlasModel


class Slide(Base, AtlasModel):
    """This class describes an individual slide. Each slide usually has 
    4 scenes (pieces of tissue). This is the parent class to the 
    TIFF (SlideCziToTif) class.
    """
    
    __tablename__ = 'slide'
    id =  Column(Integer, primary_key=True, nullable=False)
    scan_run_id = Column("FK_scan_run_id", Integer, ForeignKey('scan_run.id'))
    slide_physical_id = Column(Integer)
    slide_status = Column(Enum("Bad", "Good"), nullable=False)
    scenes = Column(Integer, nullable=False)
    insert_before_one = Column(Integer, default=0)
    scene_qc_1 =  Column(Integer, default=0)
    insert_between_one_two =  Column(Integer, default=0)
    scene_qc_2 =  Column(Integer, default=0)
    insert_between_two_three = Column(Integer, default=0)
    scene_qc_3 =  Column(Integer, default=0)
    insert_between_three_four =  Column(Integer, default=0)
    scene_qc_4 =  Column(Integer, default=0)
    insert_between_four_five =  Column(Integer, default=0)
    scene_qc_5 = Column(Integer, default=0)
    insert_between_five_six = Column(Integer, default=0)
    scene_qc_6 =  Column(Integer, default=0)
    processed = Column(Boolean(), default=False)
    file_size = Column(Float, nullable=False)
    file_name = Column(String, nullable=False)
    checksum = Column(String, nullable=True)
    comments = Column(String)

class SlideCziTif(Base, AtlasModel):
    """This is the child class of the Slide class. This model describes the 
    metadata associated with a TIFF file, or another way to think of it, 
    it describes one piece of brain tissue on a slide.
    """

    __tablename__ = 'slide_czi_to_tif'
    id =  Column(Integer, primary_key=True, nullable=False)
    FK_slide_id = Column(Integer, ForeignKey('slide.id'), nullable=False)
    file_name = Column(String, nullable=False)
    scene_number = Column(Integer, nullable=False)
    width = Column(Integer)
    height = Column(Integer)
    file_size = Column(Float)
    comments = Column(String)
    channel = Column(Integer)
    scene_index = Column(Integer)
    processing_duration = Column(Float, nullable=False)
    

class Section(Base, AtlasModel):
    """This class describes a view and not an actual database table.
    This table provides the names, locations and ordering of the 
    TIFF files.
    """

    __tablename__ = 'sections'
    id =  Column(Integer, primary_key=True, nullable=False)
    prep_id = Column(String, ForeignKey('animal.prep_id'), nullable=False)
    rescan_number = Column(Integer, nullable=False)
    czi_file = Column(String, nullable=False)
    slide_physical_id = Column(Integer, nullable=False)
    file_name = Column(String, nullable=False)
    tif_id = Column(Integer, ForeignKey('slide_czi_to_tif.id'), nullable=False)
    scene_number = Column(Integer, nullable=False)
    scene_index = Column(Integer, nullable=False)
    channel = Column(Integer, nullable=False)
    channel_index = Column(Integer, nullable=False)
    FK_slide_id = Column(Integer, ForeignKey('slide.id'), nullable=False)

