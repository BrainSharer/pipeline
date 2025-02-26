

import glob
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.cell_labeling.cell_detector_trainer import CellDetectorTrainer
from library.cell_labeling.cell_manager import CellMaker
from library.controller.sql_controller import SqlController
from library.image_manipulation.file_logger import FileLogger
from library.image_manipulation.filelocation_manager import FileLocationManager
from library.image_manipulation.parallel_manager import ParallelManager
from library.utilities.utilities_process import SCALING_FACTOR, get_scratch_dir, read_image


class CellPipeline(
    CellPipeline is a class that handles the automated cell labeling process, including creating detections,
    extracting predictions, fixing coordinates, and training models for cell detection.
    Attributes:
        animal (str): The animal identifier.
        task (str): The task identifier.
        round (int): The round number for the process. Default is 4.
        debug (bool): Flag to enable debug mode. Default is False.
        channel (int): The channel number. Default is 1.
        fileLocationManager (FileLocationManager): Manages file locations for the given animal.
        sqlController (SqlController): Manages SQL operations for the given animal.
        fileLogger (FileLogger): Logs file operations.
        cell_label_path (str): Path to the cell labels directory.
    Methods:
        create_detections():
            Starts the cell detection process and checks prerequisites.
        extract_predictions():
            Extracts cell label predictions.
        fix_coordinates():
            Fixes the coordinates of detected cells by checking against mask images.
        train():
            Trains the cell detection model using the detected features.
    CellMaker,
    ParallelManager
):
    
    def __init__(self, animal, task, round=4, debug=False):
        self.animal = animal
        self.task = task
        self.round = round
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

    def fix_coordinates(self):

        def check_df(csvfile, df):
            section = os.path.basename(csvfile).replace('detections_', '').replace('.csv', '')
            tif = str(section).zfill(3) + '.tif'
            maskpath = os.path.join(
                self.fileLocationManager.get_directory(channel=1, downsample=True, inpath='mask_placed_aligned'), tif )
            if os.path.exists(maskpath):
                mask = read_image(maskpath)
            else:
                print(f'ERROR: Mask not found {maskpath}')
                sys.exit(1)

            for index, df_row in df.iterrows():
                row = df_row['row']
                col = df_row['col']
                row = int(row // SCALING_FACTOR)
                col = int(col // SCALING_FACTOR)
                prediction = df_row['predictions']
                found = mask[row, col] > 0
                if prediction > 0 and not found:
                    #print(f'ERROR: Predicted cell {index=} not found at {row=}, {col=} {prediction=}')
                    df.loc[index, 'predictions'] = -2

            return df

        detection_files = sorted(glob.glob( os.path.join(self.cell_label_path, f'detections_*.csv') ))
        if len(detection_files) == 0:
            print(f'ERROR: NO CSV FILES FOUND IN {self.cell_label_path}')
            sys.exit(1)

        for csvfile in tqdm(detection_files):
            df = pd.read_csv(csvfile)
            df = check_df(csvfile, df)
            df.to_csv(csvfile, index=False)

    def train(self):
        import warnings
        warnings.filterwarnings("ignore")

        detection_files = sorted(glob.glob( os.path.join(self.cell_label_path, f'detections_00*.csv') ))
        if len(detection_files) == 0:
            print(f'ERROR: NO CSV FILES FOUND IN {self.cell_label_path}')
            sys.exit(1)

        dfs = []
        for csvfile in tqdm(detection_files, desc="Reading csv files"):
            df = pd.read_csv(csvfile)
            dfs.append(df)

        detection_features=pd.concat(dfs)
        detection_features_path = os.path.join(self.cell_label_path, 'detection_features.csv')
        #detection_features.to_csv(detection_features_path, index=False)
        print(detection_features.info())
        

        if self.debug:
            print(f'Found {len(dfs)} csv files in {self.cell_label_path}')
            print(f'Concatenated {len(detection_features)} rows from {len(dfs)} csv files')


        detection_features['label'] = np.where(detection_features['predictions'] > 0, 1, 0)
        #mean_score, predictions, std_score are results, not features

        drops = ['animal', 'section', 'index', 'row', 'col', 'mean_score', 'std_score', 'predictions'] 
        detection_features=detection_features.drop(drops,axis=1)
            
        #trainer = CellDetectorTrainer(self.animal, round=1)
        #new_models = trainer.train_classifier(detection_features, 676, 3)
        #trainer.save_models(new_models)

        print(f'Starting training on {self.animal} round={self.round} with {len(detection_features)} features')
        
        trainer = CellDetectorTrainer(self.animal, round=self.round) # Use Detector 4 as the basis
        new_models = trainer.train_classifier(detection_features, 676, 3, models = trainer.load_models()) # pass Detector 4 for training
        trainer = CellDetectorTrainer(self.animal, round=self.round + 1) # Be careful when saving the model. The model path is only relevant to 'round'. 
        #You need to use a new round to save the model, otherwise the previous models would be overwritten.
        trainer.save_models(new_models)
        