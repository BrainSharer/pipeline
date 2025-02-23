import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import xgboost as xgb
import pandas as pd

from library.cell_extractor.cell_detector_base import CellDetectorBase
from library.cell_extractor.annotation_proximity_tool import AnnotationProximityTool

class CellDetector(CellDetectorBase):

    def __init__(self,animal, round = 2, *args, **kwargs):
        super().__init__(animal,round=round,*args, **kwargs)
        self.detector = self.load_detector()

    def print_version(self):
        print('version of xgboost is:',xgb.__version__,'should be at least 1.5.0')

    def get_detection_results(self):
        features = self.get_combined_features_for_detection()
        scores, labels, _mean, _std = self.detector.calculate_scores(features)
        import sys
        print('exitng')
        sys.exit()
        print(f'{scores=} {labels=} {_mean=} {_std=}')
        predictions = self.detector.get_prediction(_mean, _std)
        detection_df = self.get_combined_features()
        detection_df["mean_score"], detection_df["std_score"] = _mean, _std
        detection_df["label"] = labels
        detection_df["predictions"] = predictions
        detection_df = detection_df[
            [
                "animal",
                "section",
                "row",
                "col",
                "label",
                "mean_score",
                "std_score",
                "predictions",
            ]
        ]
        return detection_df

    def calculate_and_save_detection_results(self):
        print('calculating detection results line 43')

        detection_df = self.get_detection_results()
        detection_df.to_csv(self.DETECTION_RESULT_DIR,index=False)

class MultiThresholdDetector(CellDetector,AnnotationProximityTool):
    def __init__(self,animal,round,thresholds = [2000,2100,2200,2300,2700]):
        super().__init__(animal = animal,round = round)
        self.thresholds=thresholds
        self.MULTI_THRESHOLD_DETECTION_RESULT_DIR = os.path.join(self.DETECTION,f'multithreshold_detections_{self.thresholds}_round_{str(self.round)}.csv')

    def get_detections_and_scores_for_all_threshold(self):
        non_detections = []
        detections = []
        scores = []
        non_detection_scores = []
        for threshold in self.thresholds:
            print(f'loading threshold {threshold}')
            detector = CellDetector(self.animal,self.round,segmentation_threshold=threshold)
            detection = detector.load_detections()
            sure = detection[detection.predictions == 2]
            null = detection[detection.predictions == -2]
            null = null.sample(int(len(null)*0.01))
            unsure = detection[detection.predictions == 0]
            sure_score = pd.DataFrame({'mean':sure.mean_score,'std':sure.std_score,'label':sure.label})
            unsure_score = pd.DataFrame({'mean':unsure.mean_score,'std':unsure.std_score,'label':unsure.label})
            null_score = pd.DataFrame({'mean':null.mean_score,'std':null.std_score,'label':null.label})
            sure = pd.DataFrame({'x':sure.col,'y':sure.row,'section':sure.section,'name':[f'{threshold}_sure' for _ in range(len(sure))]})
            unsure = pd.DataFrame({'x':unsure.col,'y':unsure.row,'section':unsure.section,'name':[f'{threshold}_unsure' for _ in range(len(unsure))]})
            null = pd.DataFrame({'x':null.col,'y':null.row,'section':null.section,'name':[f'{threshold}_null' for _ in range(len(null))]})
            detections.append(sure)
            detections.append(unsure)
            scores.append(sure_score)
            scores.append(unsure_score)
            non_detections.append(null)
            non_detection_scores.append(null_score)
        detections = pd.concat(detections)
        non_detections = pd.concat(non_detections)
        non_detection_scores = pd.concat(non_detection_scores)
        scores = pd.concat(scores)
        return detections,scores,non_detections,non_detection_scores

    def check_cells(self,scores,check_function,determination_function):
        cell =[i for i in self.pair_categories.values() if check_function(i)]
        cell_pairs = [self.pairs[id] for id,i in self.pair_categories.items() if check_function(i)]
        final_cell_detection = []
        for id,categories in enumerate(cell):
            pair = cell_pairs[id]
            coords = self.annotations_to_compare.iloc[pair]
            score = scores.iloc[pair]
            row = determination_function(score,coords,categories)
            final_cell_detection=final_cell_detection+row
        final_cell_detection = pd.concat(final_cell_detection,axis=1).T
        return final_cell_detection

    def plot_detector_threshold(self):
        detections = self.load_detections()
        plt.figure(figsize=[30,15])
        alpha = 0.8
        size = 15
        plt.scatter(detections['mean_score'].to_numpy(),detections['std_score'].to_numpy(),color='slategrey',s=1,alpha=0.3)
        plt.axvline(-1.5)
        plt.axvline(1.5)

    def determine_pure_detection(self,scores,type_to_exclude='sure'):
        does_not_have_cell_type = lambda i: self.check(i,exclude=[f'{threshold}_{type_to_exclude}' for threshold in self.thresholds])
        def find_max_mean_score_of_the_group(score,coords,categories):
            max_id = np.argmax(score.to_numpy()[:,0])
            max_threshold = categories[max_id].split('_')[0]
            is_max_threshold = [i.split('_')[0]==max_threshold for i in categories]
            row = coords.iloc[is_max_threshold].join(score.iloc[is_max_threshold])
            return [i for _,i in row.iterrows()]
        final_cell_detection = self.check_cells(scores,check_function = does_not_have_cell_type,determination_function = find_max_mean_score_of_the_group)
        return final_cell_detection

    def determine_mixed_detection(self,scores):
        def mixed_cell_type(categories):
            has_sure = np.any([i.split('_')[1]=='sure' for i in categories])
            has_unsure = np.any([i.split('_')[1]=='unsure' for i in categories])
            return has_sure and has_unsure
        def find_max_mean_score_of_sure_detections_in_group(score,coords,categories):
            is_sure = np.array([i.split('_')[1]=='sure' for i in categories ])
            score = score[is_sure]
            coords = coords[is_sure]
            categories = np.array(categories)[is_sure]
            max_id = np.argmax(score.to_numpy()[:,0])
            max_threshold = categories[max_id].split('_')[0]
            is_max_threshold = [i.split('_')[0]==max_threshold for i in categories]
            row = coords.iloc[is_max_threshold].join(score.iloc[is_max_threshold])
            return [i for _,i in row.iterrows()]
        final_cell_detection = self.check_cells(scores,check_function = mixed_cell_type,determination_function = find_max_mean_score_of_sure_detections_in_group)
        return final_cell_detection

    def calculate_and_save_detection_results(self):
        print('calculating detection results line 136')
        detections,scores,non_detections,non_detection_scores = self.get_detections_and_scores_for_all_threshold()
        self.set_annotations_to_compare(detections)
        self.find_equivalent_points()
        final_unsure_detection = self.determine_pure_detection(scores,type_to_exclude = 'sure')
        final_sure_detection = self.determine_pure_detection(scores,type_to_exclude = 'unsure')
        final_mixed_detection = self.determine_mixed_detection(scores)
        self.set_annotations_to_compare(non_detections)
        self.find_equivalent_points()
        final_non_detection = self.determine_pure_detection(non_detection_scores,type_to_exclude = '')
        final_detection = pd.concat([final_sure_detection,final_unsure_detection,final_mixed_detection,final_non_detection])
        final_detection.to_csv(self.MULTI_THRESHOLD_DETECTION_RESULT_DIR,index=False)
    
    def load_detections(self):
        return pd.read_csv(self.MULTI_THRESHOLD_DETECTION_RESULT_DIR)
    
    def get_sures(self):
        detections = self.load_detections()
        return detections[[string_to_prediction(i) ==2 for i in detections.name]]
    
    def get_unsures(self):
        detections = self.load_detections()
        return detections[[string_to_prediction(i) ==0 for i in detections.name]]

def string_to_prediction(string):
    if string.split('_')[1] == 'sure':
        return 2
    elif string.split('_')[1] == 'unsure':
        return 0
    elif string.split('_')[1] == 'null':
        return -2

def detect_cell(animal,round,*args,**kwargs):
    print(f'detecting {animal}')
    detector = CellDetector(*args,animal = animal,round=round,**kwargs)
    detector.calculate_and_save_detection_results()

def detect_cell_multithreshold(animal,round,thresholds = [2000,2100,2200,2300,2700]):
    print(f'detecting {animal} multithreshold')
    detector = MultiThresholdDetector(animal,round,thresholds)
    detector.calculate_and_save_detection_results()
