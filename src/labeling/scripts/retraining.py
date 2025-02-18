import os, sys
import pickle as pkl
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.cell_extractor.cell_detector_base import CellDetectorBase
from library.cell_extractor.cell_detector_trainer import CellDetectorTrainerDK55, CellDetectorTrainer
from library.cell_extractor.cell_detector import CellDetector




generator = CellDetectorTrainerDK55('DK55',round=2,segmentation_threshold=2000)
train_features = generator.load_refined_original_feature()

print(train_features.shape)

trainer = CellDetectorTrainer('DK55',round=1)
new_models = trainer.train_classifier(train_features,676,3) # training iteration = 676, depth of XGBoost trees = 3
trainer.save_models(new_models)

detector = CellDetector('DK41',round=1)
# detector.detector.model = pkl.load(open('/scratch/k1qian/Cell_Detectors/detectors_new.pkl', 'rb'))
detector.calculate_and_save_detection_results()