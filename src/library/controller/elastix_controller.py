from datetime import datetime
from library.database_model.elastix_transformation import ElastixTransformation

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

    def get_elastix_row(self, animal, section, iteration):
        """
        Retrieve a specific row from the ElastixTransformation table based on the given animal, section, and iteration.
        :param animal (str): The identifier for the animal.
        :param section (int): The section number.
        :param iteration (int, optional): The iteration number. Defaults to 0.
        :return ElastixTransformation: The first matching row from the ElastixTransformation table, or None if no match is found.
        :raises NoResultFound: If no matching row is found in the database.
        """

        row = self.session.query(ElastixTransformation).filter(
            ElastixTransformation.FK_prep_id == animal,
            ElastixTransformation.iteration == iteration,
            ElastixTransformation.section == section).first()

        if row is None:
            R = 0
            xshift = 0
            yshift = 0
        else:
            R = row.rotation
            xshift = row.xshift
            yshift = row.yshift

        return R, xshift, yshift

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
    
    def add_elastix_row(self, animal, section, rotation, xshift, yshift, metric, iteration):
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


    def update_elastix_row(self, animal, section, updates, iteration):
        """Update a row
        
        :param animal: (str) Animal ID
        :param section: (str) Section Number
        :param updates: dictionary of column:values to update
        """
        self.session.query(ElastixTransformation)\
            .filter(ElastixTransformation.FK_prep_id == animal)\
            .filter(ElastixTransformation.iteration == iteration)\
            .filter(ElastixTransformation.section == section).update(updates)
        self.session.commit()

    def delete_elastix_iteration(self, animal, iteration=1):
        """
        Deletes a specific iteration of Elastix transformation for a given animal.

        Args:
            animal (str): The identifier for the animal.
            iteration (int, optional): The iteration number of the Elastix transformation to delete. Defaults to 1.

        Returns:
            None
        """

        self.session.query(ElastixTransformation)\
            .filter(ElastixTransformation.FK_prep_id == animal)\
            .filter(ElastixTransformation.iteration == iteration).delete()
        self.session.commit()

    def get_elastix_count(self, animal, iteration):
        count = self.session.query(ElastixTransformation)\
            .filter(ElastixTransformation.FK_prep_id == animal)\
            .filter(ElastixTransformation.iteration == iteration)\
            .count() 
        return count
