from sqlalchemy import Column, Integer, String, Boolean, DateTime

from library.database_model.atlas_model import Base


class Lab(Base):
    __tablename__ = 'auth_lab' 

    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    lab_name = Column(String(100), nullable=False)
    lab_url = Column(String(250), nullable=False)
    active = Column(Boolean, nullable=False)
    created = Column(DateTime, nullable=False)
