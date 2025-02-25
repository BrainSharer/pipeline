

import glob
import os
import sys
import numpy as np
import pandas as pd
from library.cell_labeling.cell_detector_trainer import CellDetectorTrainer
from library.cell_labeling.cell_manager import CellMaker
from library.controller.sql_controller import SqlController
from library.image_manipulation.file_logger import FileLogger
from library.image_manipulation.filelocation_manager import FileLocationManager
from library.image_manipulation.parallel_manager import ParallelManager
from library.utilities.utilities_process import get_scratch_dir


class CellPipeline(
    CellMaker,
    ParallelManager
):
    
    def __init__(self, animal, task, debug=False):
        self.animal = animal
        self.task = task
        self.debug = debug
        self.channel = 1
        self.fileLocationManager = FileLocationManager(animal)
        self.sqlController = SqlController(animal)
        self.fileLogger = FileLogger(self.fileLocationManager.get_logdir(), self.debug)
        self.cell_label_path = os.path.join(self.fileLocationManager.prep, 'cell_labels')


    def create_detections(self):
        """
        USED FOR AUTOMATED CELL LABELING - FINAL OUTPUT FOR CELLS DETECTED
        """
        print("Starting cell detections")

        scratch_tmp = get_scratch_dir()
        self.check_prerequisites(scratch_tmp)

        # IF ANY ERROR FROM check_prerequisites(), PRINT ERROR AND EXIT

        # ASSERT STATEMENT COULD BE IN UNIT TEST (SEPARATE)

        self.start_labels()
        print(f'Finished cell detections')

        # ADD CLEANUP OF SCRATCH FOLDER

    def extract_predictions(self):
        print('Starting extraction')
        self.parse_cell_labels()
        print(f'Finished extraction.')

    def train(self):
        import warnings
        warnings.filterwarnings("ignore")

        detection_files = sorted(glob.glob( os.path.join(self.cell_label_path, f'detections_*.csv') ))
        if len(detection_files) == 0:
            print(f'ERROR: NO CSV FILES FOUND IN {self.cell_label_path}')
            sys.exit(1)

        dfs = []
        for csvfile in detection_files:
            df = pd.read_csv(csvfile)
            dfs.append(df)

        detection_features=pd.concat(dfs)

        detection_features['label'] = np.where(detection_features['predictions'] > 0, 1, 0)
        #mean_score, predictions, std_score are results, not features

        drops = ['animal', 'section', 'index', 'row', 'col', 'mean_score', 'std_score', 'predictions'] 
        detection_features=detection_features.drop(drops,axis=1)
        if self.debug:
            print(detection_features.head())
            
        #trainer = CellDetectorTrainer(self.animal, round=1)
        #new_models = trainer.train_classifier(detection_features, 676, 3)
        #trainer.save_models(new_models)

        round = 4
        trainer = CellDetectorTrainer('DK184', round=round) # Use Detector 4 as the basis
        new_models = trainer.train_classifier(detection_features, 676, 3, models = trainer.load_models()) # pass Detector 4 for training
        trainer = CellDetectorTrainer('DK184', round=round+1) # Be careful when saving the model. The model path is only relevant to 'round'. 
        #You need to use a new round to save the model, otherwise the previous models would be overwritten.
        trainer.save_models(new_models)
        