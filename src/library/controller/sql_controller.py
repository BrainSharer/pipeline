"""
This is the base sql class. It is mostly used per animal, so the init function
needs an animal passed to the constructor
It also needs for the animal, histology and scan run tables to be
filled out for each animal to use
"""

import sys
import numpy as np
from collections import OrderedDict
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import scoped_session
from sqlalchemy.pool import NullPool
import urllib

from library.controller.annotation_session_controller import AnnotationSessionController
from library.controller.animal_controller import AnimalController
from library.controller.elastix_controller import ElastixController
from library.controller.histology_controller import HistologyController
from library.controller.scan_run_controller import ScanRunController
from library.controller.sections_controller import SectionsController
from library.controller.slide_tif_controller import SlideCZIToTifController
from library.database_model.scan_run import ScanRun
from library.database_model.histology import Histology

try:
    from settings import host, password, user, schema
    password = urllib.parse.quote_plus(str(password)) # escape special characters
except ImportError as fe:
    print('You must have a settings file in the pipeline directory.', fe)
    raise


class SqlController(AnnotationSessionController, AnimalController, ElastixController, HistologyController,
                     ScanRunController, SectionsController, SlideCZIToTifController):
    """ This is the base controller class for all things SQL.  
    Each parent class of SqlController would correspond to one table in the database, and include all the 
    methods to interact with that table
    """

    def __init__(self, animal):
        """ setup the attributes for the SlidesProcessor class
            Args:
                animal: object of animal to process
        """

        connection_string = f'mysql+pymysql://{user}:{password}@{host}/{schema}?charset=utf8'
        engine = create_engine(connection_string, poolclass=NullPool)
        self.session = scoped_session(sessionmaker(bind=engine)) 
        self.session.begin()

        if self.animal_exists(animal):
            self.animal = self.get_animal(animal)
        else:
            print(f'No animal/brain with the name {animal} was found in the database.')
            sys.exit()
        
        try:
            self.histology = self.session.query(Histology).filter(
                Histology.FK_prep_id == animal).first()
        except NoResultFound:
            print(f'No histology for {animal}')

        try:
            self.scan_run = self.session.query(ScanRun)\
                .filter(ScanRun.FK_prep_id == animal).one()
        except NoResultFound as nrf:
            print(f'No scan run for {animal}')
            sys.exit()
            
        
        self.slides = None
        self.tifs = None
        self.valid_sections = OrderedDict()
        self.session.close()

    def get_resolution(self, animal):
        """Returns the resolution for an animal
        
        :param animal: string primary key
        :return numpy array: of the resolutions
        """
        
        scan_run = self.get_scan_run(animal)
        histology = self.get_histology(animal)
        if histology.orientation == 'coronal':
            return np.array([scan_run.zresolution, scan_run.resolution, scan_run.resolution])
        elif histology.orientation == 'horizontal':
            return np.array([scan_run.resolution, scan_run.zresolution, scan_run.resolution])
        elif histology.orientation == 'sagittal':
            return np.array([scan_run.resolution, scan_run.resolution, scan_run.zresolution])

    def update_row(self, row):
        """update one row of a database

        :param row: a row of a database table.
        """

        try:
            self.session.merge(row)
            self.session.commit()
        except Exception as e:
            print(f'No merge for  {e}')
            self.session.rollback()
    
    def add_row(self, data):
        """adding a row to a table

        :param data: (data to be added ): instance of sqalchemy ORMs
        """

        try:
            self.session.add(data)
            self.session.commit()
        except Exception as e:
            print(f'No merge {e}')
            self.session.rollback()

    
    def get_row(self, search_dictionary, model):
        """look for a specific row in the database and return the result

        :param search_dictionary: (dict): field and value of the search
        :param model: (sqalchemy ORM): the sqalchemy ORM in question 

        :return:  sql alchemy query
        """ 

        query_start = self.session.query(model)
        exec(f'from {model.__module__} import {model.__name__}')
        for key, value in search_dictionary.items():
            query_start = eval(f'query_start.filter({model.__name__}.{key}=="{value}")')
        return query_start.one()
    
    def row_exists(self,search_dictionary,model):
        """check if a specific row exist in a table

        
        :param search_dictionary: (dict): field and value for the search
        :param model: (sqalchemy ORM): sqalchemy ORM

        :return boolean: whether the row exists
        """

        return self.get_row(search_dictionary,model) is not None
    
    def query_row(self, model, search_dictionary):
        """
        Queries a single row from the database based on the provided search criteria.

        Args:
            search_dictionary (dict): A dictionary where the keys are the column names and the values are the values to search for.
            model (Base): The SQLAlchemy model class to query.

        Returns:
            model: The first row that matches the search criteria, or None if no match is found.
        """

        query_start = self.session.query(model)
        exec(f'from {model.__module__} import {model.__name__}')
        for key, value in search_dictionary.items():
            query_start = eval(f'query_start.filter({model.__name__}.{key}=="{value}")')
        return query_start.first()
    
    def delete_row(self, model, search_dictionary):
        """
        Deletes a row from the database based on the given search criteria.

        Args:
            search_dictionary (dict): A dictionary containing the search criteria to find the row to delete.
            model (Base): The SQLAlchemy model class representing the table from which the row will be deleted.

        Returns:
            None
        """

        row = self.get_row(search_dictionary, model)
        self.session.delete(row)
        self.session.commit()

    def query_one_with_filters_and_sum(self, model, filters: dict, sum_columns: list, group_by_columns: list):
        """
        Query a single row with multiple filters, sum multiple columns, and group by multiple columns.
        
        :param model: SQLAlchemy model class
        :param filters: Dictionary of filters {column: value}
        :param sum_columns: List of columns to sum
        :param group_by_columns: List of columns to group by
        :return: Single row result or None
        """
        
        # Prepare sum expressions
        sum_expressions = [func.sum(getattr(model, col)).label(col) for col in sum_columns]
        
        # Prepare group by expressions
        group_by_expressions = [getattr(model, col) for col in group_by_columns]
        
        # Prepare filter conditions
        filter_conditions = [getattr(model, col) == value for col, value in filters.items()]
        
        # Build query
        query = (
            self.session.query(*group_by_expressions, *sum_expressions)
            .filter(*filter_conditions)
            .group_by(*group_by_expressions)
        )
        # Fetch one row
        return query.first()
        

