from sqlalchemy import Column, ForeignKey, String, Integer, DateTime, Double, DECIMAL

from library.database_model.atlas_model import Base, AtlasModel

class BrainRegion(Base, AtlasModel):
    __tablename__ = 'brain_region'
    __table_args__ = {'extend_existing': True}
    id =  Column(Integer, primary_key=True, nullable=False)
    abbreviation = Column(String, nullable=False)
    description = Column(String, nullable=False)
    active =  Column(Integer,default=1)
    created =  Column(DateTime)

class RegionOfInterest(Base, AtlasModel):
    __tablename__ = 'region_of_interest'
    __table_args__ = {'extend_existing': True}
    id =  Column(Integer, primary_key=True, nullable=False)
    FK_state_id = Column(Integer, nullable=False)
    FK_prep_id = Column(String(20), nullable=False)
    layer_name = Column(String(50), nullable=False)
    volume_name = Column(String(50), nullable=False)
    section = Column(Integer, nullable=False)
    com_x = Column(Double, default=0)
    com_y = Column(Double, default=0)
    area = Column(Double, default=0)
    threshold = Column(Integer, default=0)
    contour_count = Column(Integer, default=0)
    active =  Column(Integer,default=1)
    created =  Column(DateTime)


class CellData(Base):
    __tablename__ = 'cell_data'
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, nullable=False)
    FK_roi_id = Column(Integer, ForeignKey('region_of_interest.id'), nullable=False)
    com_x = Column(Double, default=0.00)
    com_y = Column(Double, default=0.00)
    area = Column(Double, default=0)
