import os, sys
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.cell_extractor.cell_detector_trainer import CellDetectorTrainer


def trainer(animal):

    dfpath = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps/cell_labels/detections_100.csv'
    my_features = pd.read_csv(dfpath)
    my_features['label'] = np.where(my_features['predictions'] > 0, 1, 0)
    drops = ['animal', 'section', 'index', 'row', 'col'] 
    my_features=my_features.drop(drops,axis=1)
    #print(my_features.head())
    trainer = CellDetectorTrainer(animal, round=1)
    new_models = trainer.train_classifier(my_features, 676, 3)
    trainer.save_models(new_models)

    """
    trainer = CellDetectorTrainer('DK184', round=4) # Use Detector 4 as the basis
    new_models = trainer.train_classifier(my_features, 676, 3, models = trainer.load_models()) # pass Detector 4 for training
    trainer = CellDetectorTrainer('DK184', round=7) # Be careful when saving the model. The model path is only relevant to 'round'. 
    You need to use a new round to save the model, otherwise the previous models would be overwritten.
    trainer.save_models(new_models)
    """