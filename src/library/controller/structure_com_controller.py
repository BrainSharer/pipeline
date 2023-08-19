import numpy as np
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy import func, not_

from library.controller.sql_controller import SqlController
from library.database_model.annotation_points import AnnotationSession, AnnotationType, StructureCOM
from library.database_model.brain_region import BrainRegion



class StructureCOMController(SqlController):
    """The class that queries and addes entry to the StructureCOM table
    """

    def get_annotation_dict(self, prep_id, annotator_id=2):
        """This method replaces get_centers_dict and get_layer_data_row.

        :param prep_id: string name of animal
        :param label: formerly layer, the string name of the layer
        """

        row_dict = {}
        """
        try:
            animal = Animal.objects.get(pk=prep_id)
        except Animal.DoesNotExist:
            logger.error(f'Error, {prep_id} does not exist in DB. Method: get_annotation_dict is returning an empty dictionary.')
            return row_dict
        base = AnnotationBase()
        base.set_animal_from_id(prep_id)
        base.set_annotator_from_id(annotator_id)
        """

        sessions = self.get_available_sessions(prep_id, annotator_id)

        sids = [session.id for session in sessions]
        
        rows = self.session.query(StructureCOM)\
            .filter(StructureCOM.FK_session_id.in_(sids))\
            .join(AnnotationSession)\
                .filter(AnnotationSession.FK_prep_id==prep_id)\
                .filter(AnnotationSession.FK_user_id==annotator_id)\
            .   all()
        brain_region_dict = {}
        brain_regions = self.session.query(BrainRegion).filter(BrainRegion.active==True).all()
        for brain_region in brain_regions:
            brain_region_dict[brain_region.id] = brain_region.abbreviation

        for row in rows:
            brain_region_id = row.session.brain_region.id
            abbreviation = brain_region_dict[brain_region_id]
            row_dict[abbreviation] = [row.x, row.y, row.z]
        return row_dict

    def get_coms(self, prep_id, annotator_id=2):
        """returns the Center Of Mass of structures for a Animal ID and annotator combination

        Args:
            prep_id (str): Animal ID
            annotator_id (int): Annotator Id

        Returns:
            list: a list of coms in order by abbreviation
        """

        sessions = self.get_available_sessions(prep_id, annotator_id)
        coms = []
        for session in sessions:
            com = self.session.query(StructureCOM).join(AnnotationSession).join(BrainRegion)\
                .filter(StructureCOM.FK_session_id == session.id).order_by(BrainRegion.abbreviation).first()
            coms.append(com)
        return coms

    def get_COM(self, prep_id, annotator_id=2):
        """returns the Center Of Mass of structures for a Animal ID and annotator combination

        Args:
            prep_id (str): Animal ID
            annotator_id (int): Annotator Id

        Returns:
            dict: dictionary of x,y,z coordinates indexed by structure name
        """

        sessions = self.get_available_sessions(prep_id, annotator_id)
        coms = []
        for session in sessions:
            com = self.session.query(StructureCOM)\
                .filter(StructureCOM.FK_session_id == session.id).first()
            coms.append(com)
        coordinate = [[i.x, i.y, i.z] for i in coms if i is not None]
        structure = [i.session.brain_region.abbreviation for i in coms if i is not None]
        return dict(zip(structure, coordinate))

    def get_all_manual_COM(self):
        coms = self.session.query(StructureCOM)\
            .filter(StructureCOM.source == 'MANUAL')\
            .join(AnnotationSession).filter(AnnotationSession.FK_brain_region_id != 52)\
            .all()
        coms = np.array(coms)
        animals = np.array([i.session.FK_prep_id for i in coms])
        unique_animals = np.unique(animals)
        all_coms = {}
        for animal in unique_animals:
            animal_com = coms[animals==animal]
            names = [self.structure_id_to_abbreviation(i.session.FK_brain_region_id) for i in animal_com]
            #coords = [np.floor([i.x,i.y,i.z]).astype(int) for i in animal_com]
            coords = [[i.x,i.y,i.z] for i in animal_com]
            all_coms[animal] = dict(zip(names,coords))
        return all_coms        
    
    def get_available_sessions(self, prep_id, annotator_id):
        sessions = self.session.query(AnnotationSession)\
            .filter(AnnotationSession.active==True)\
            .filter(AnnotationSession.annotation_type==AnnotationType.STRUCTURE_COM)\
            .filter(AnnotationSession.FK_prep_id==prep_id)\
            .filter(AnnotationSession.FK_user_id==annotator_id)\
            .join(BrainRegion)\
            .order_by(BrainRegion.abbreviation).all()
        return sessions
    
    def get_active_sessions(self):
        sessions = self.session.query(AnnotationSession)\
            .filter(AnnotationSession.active==True)\
            .filter(AnnotationSession.annotation_type==AnnotationType.STRUCTURE_COM)\
            .order_by(AnnotationSession.FK_prep_id).all()
        return sessions

    def get_active_animal_sessions(self, FK_prep_id):
        sessions = self.session.query(AnnotationSession)\
            .filter(AnnotationSession.FK_prep_id==FK_prep_id)\
            .filter(AnnotationSession.active==True)\
            .filter(AnnotationSession.annotation_type==AnnotationType.STRUCTURE_COM)\
            .order_by(AnnotationSession.FK_prep_id).all()
        return sessions

    def get_com_annotator(self, FK_prep_id):
        sessions = self.get_active_animal_sessions(FK_prep_id=FK_prep_id)
        users = []
        for session in sessions:
            users.append(session.FK_user_id)
        try:
            id = max(users, key=users.count)
        except:
            id = None
        return id
        

    def structure_abbreviation_to_id(self, abbreviation):
        try:
            structure = self.get_structure(str(abbreviation).strip())
        except NoResultFound as nrf:
            print(f'No structure found for {abbreviation} {nrf}')
            return
        return structure.id

    def get_structure(self, abbrv):
        """
        Returns a structure ORM object
        This search has to be case sensitive!
        :param abbrv: the abbreviation of the structure
        :return: structure object
        """
        return self.session.query(BrainRegion).filter(BrainRegion.abbreviation == func.binary(abbrv)).one()

    def structure_id_to_abbreviation(self, id):
        try:
            structure = self.get_row({'id': id}, BrainRegion)
        except NoResultFound as nrf:
            print(f'No structure found for {id} {nrf}')
            return
        return structure.abbreviation
    
    def get_structures(self):
        """return a list of active structures. We don't want line, point, polygon
        with ids 52, 53, 54

        Returns:
            list: list of structure ORM
        """        
        return self.session.query(BrainRegion).filter(BrainRegion.active.is_(True))\
            .filter(not_(BrainRegion.id.in_([52, 53, 54]))).all()

