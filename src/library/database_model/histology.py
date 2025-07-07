from sqlalchemy import Column, String, Date, Enum, Integer, ForeignKey

from library.database_model.atlas_model import AtlasModel, Base

class Histology(Base, AtlasModel):
    """This class provides the metadata associated with the histology of the animal
    """
    
    __tablename__ = 'histology'

    id =  Column(Integer, primary_key=True, nullable=False)
    FK_prep_id = Column("FK_prep_id", String, ForeignKey('animal.prep_id'), nullable=False, unique=True)
    virus_id = Column("FK_virus_id", Integer, ForeignKey('virus.id'), nullable=True)
    #performance_center = Column(Enum("CSHL", "Salk", "UCSD", "HHMI"))
    anesthesia = Column(Enum("ketamine", "isoflurane", "pentobarbital", "fatal plus"))
    perfusion_age_in_days = Column(Integer, nullable=False)
    perfusion_date = Column(Date)
    exsangination_method = Column(Enum("PBS", "aCSF", "Ringers"))
    fixative_method = Column(Enum("Para", "Glut", "Post fix") )
    special_perfusion_notes = Column(String)
    post_fixation_period = Column(Integer, default=0, nullable=False)
    whole_brain = Column(Enum("Y", "N"))
    block = Column(String)
    date_sectioned = Column(Date)
    side_sectioned_first = Column(Enum("Left","Right", "Dorsal", "Ventral", "Anterior", "Posterior"))
    scene_order = Column(Enum("ASC", "DESC"))
    sectioning_method = Column(Enum("cryoJane", "cryostat", "vibratome", "optical", "sliding microtiome"))
    section_thickness = Column(Integer, default=20, nullable=False)
    orientation = Column(Enum("coronal", "horizontal", "sagittal", "oblique"))
    oblique_notes = Column(String)
    mounting = Column(Enum("every section", "2nd", "3rd", "4th", "5ft", "6th"))
    counterstain = Column(Enum("thionin","NtB","NtFR","DAPI","Giemsa","Syto41","NTB/thionin", "NTB/PRV-eGFP", "NTB/PRV", "NTB/ChAT/ΔGRV", "NTB/ChAT/Ai14"))
    comments = Column(String)
    FK_lab_id = Column("FK_lab_id", Integer, ForeignKey('auth_lab.id'), nullable=False)

    #def __repr__(self):
    #    return "Histology(prep_id='%s')" % (self.prep_id)
