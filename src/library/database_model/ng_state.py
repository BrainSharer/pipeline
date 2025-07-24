from sqlalchemy import Column, String, Date, Boolean, Enum, Integer, ForeignKey

from library.database_model.atlas_model import AtlasModel, Base
from library.database_model.lab import Lab

class NeoroglancerState(Base, AtlasModel):
    """This class provides access to neuroglancer_state table
    """
    
    __tablename__ = 'neuroglancer_state'

    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    neuroglancer_state = Column(String, nullable=False)  # Maps to 'longtext'
    created = Column(Date, nullable=False)
    updated = Column(Date, nullable=False)
    comments = Column(String(255), nullable=False)  # Maps to 'varchar(255)'
    description = Column(String(2048), nullable=True)  # Maps to 'varchar(2048)'
    readonly = Column(Boolean, nullable=False)  # Maps to 'tinyint(1)'
    public = Column(Boolean, default=False, nullable=False)  # Maps to 'tinyint(1)' with default value 0
    active = Column(Boolean, default=True, nullable=False)  # Maps to 'tinyint(1)' with default value 1
    FK_user_id = Column(String, ForeignKey('auth_user.id'), nullable=False)
    FK_lab_id = Column(Integer, ForeignKey('auth_lab.id'), nullable=False)

    def __repr__(self):
        return f"<NeuroglancerState(id={self.id}, comments={self.comments})>"
