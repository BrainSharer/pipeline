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
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import scoped_session
from sqlalchemy.pool import NullPool
import urllib

from library.controller.animal_controller import AnimalController
from library.controller.elastix_controller import ElastixController
from library.controller.histology_controller import HistologyController
from library.controller.marked_cell_controller import MarkedCellController
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


class SqlController(AnimalController, ElastixController, HistologyController, MarkedCellController,
                     ScanRunController, SectionsController, SlideCZIToTifController):
    """ This is the base controller class for all things SQL.  
    Each parent class of SqlController would correspond to one table in the database, and include all the 
    methods to interact with that table
    """

    def __init__(self, animal, rescan_number=0):
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
                Histology.FK_prep_id == animal).one()
        except NoResultFound:
            print(f'No histology for {animal}')

        try:
            self.scan_run = self.session.query(ScanRun)\
                .filter(ScanRun.FK_prep_id == animal)\
                .filter(ScanRun.rescan_number == rescan_number).one()
        except NoResultFound:
            print(f'No scan run for {animal}')
        
        self.slides = None
        self.tifs = None
        self.rescan_number = rescan_number
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
    
    def query_table(self,search_dictionary,model):
        """query a sql table and return all the results fitting the search criterias

        :param search_dictionary: (dict): search field and value
        :param model: (sqalchemy ORM class): sqalchemy ORM

        returns list: the query result in a list of ORM objects 
        """

        query_start = self.session.query(model)
        exec(f'from {model.__module__} import {model.__name__}')
        for key, value in search_dictionary.items():
            query_start = eval(f'query_start.filter({model.__name__}.{key}=="{value}")')
        return query_start.all()
    
    def delete_row(self, search_dictionary, model):
        """Deletes one row of any table

        :param search_dictionary: (dict): search field and value
        :param model: (sqalchemy ORM class): sqalchemy ORM
        """

        row = self.get_row(search_dictionary,model)
        self.session.delete(row)
        self.session.commit()