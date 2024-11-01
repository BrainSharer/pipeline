from datetime import datetime
from sqlalchemy.orm.exc import NoResultFound
from library.database_model.elastix_transformation import ElastixTransformation
#from library.controller.sql_controller import SqlController

class ElastixController():
    """Controller class for the elastix table

    Args:
        Controller (Class): Parent class of sqalchemy session
    """

    def check_elastix_row(self, animal, section, iteration=0):
        """checks that a given elastix row exists in the database

        :param animal: (str): Animal ID
        :section (int): Section Number
        :return boolean: if the row in question exists
        """

        row_exists = bool(self.session.query(ElastixTransformation).filter(
            ElastixTransformation.FK_prep_id == animal,
            ElastixTransformation.iteration == iteration,
            ElastixTransformation.section == section).first())
        return row_exists

    def get_elastix_row(self, animal, section, iteration=0):
        """gets a given elastix row exists in the database

        :param animal: (str): Animal ID
        :section (int): Section Number
        :iteration (int): Iteration, which pass are we working on.
        :return boolean: if the row in question exists
        """
        row = None
        try:
            row = self.session.query(ElastixTransformation).filter(
                ElastixTransformation.FK_prep_id == animal,
                ElastixTransformation.iteration == iteration,
                ElastixTransformation.section == section).first()
        except NoResultFound as nrf:
            print(f'No row value for {animal} {section} error: {nrf}')

        return row

    def check_elastix_metric_row(self, animal, section, iteration=0):
        """checks that a given elastix row exists in the database

        :param animal (str): Animal ID
        :param section (int): Section Number

        :return bool: if the row in question exists
        """

        row_exists = bool(self.session.query(ElastixTransformation).filter(
            ElastixTransformation.FK_prep_id == animal,
            ElastixTransformation.section == section,
            ElastixTransformation.iteration == iteration,
            ElastixTransformation.metric != 0).first())
        return row_exists
    
    def add_elastix_row(self, animal, section, rotation, xshift, yshift, metric=0, iteration=0):
        """adding a row in the elastix table

        :param animal: (str) Animal ID
        :param section: (str) Section Number
        :param rotation: float
        :param xshift: float
        :param yshift: float
        """

        data = ElastixTransformation(
            FK_prep_id=animal, section=section, rotation=rotation, xshift=xshift, yshift=yshift, iteration=iteration,
            metric=metric, created=datetime.now(), active=True)
        self.add_row(data)


    def update_elastix_row(self, animal, section, updates):
        """Update a row
        
        :param animal: (str) Animal ID
        :param section: (str) Section Number
        :param updates: dictionary of column:values to update
        """
        self.session.query(ElastixTransformation)\
            .filter(ElastixTransformation.FK_prep_id == animal)\
            .filter(ElastixTransformation.iteration == 0)\
            .filter(ElastixTransformation.section == section).update(updates)
        self.session.commit()
