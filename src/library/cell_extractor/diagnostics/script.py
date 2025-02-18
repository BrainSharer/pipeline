import sys
sys.path.append('/home/zhw272/programming/pipeline_utility/in_development/Will')
from cell_extractor.AnnotationProximityTool import AnnotationProximityTool
from cell_extractor.CellDetector import CellDetector,MultiThresholdDetector
from cell_extractor.CellDetectorBase import CellDetectorBase
from Controllers.MarkedCellController import MarkedCellController
import numpy as np

import pandas as pd
def parse_detections(detections):
    sure = detections[detections.predictions==2]
    unsure = detections[detections.predictions==0]
    sure = pd.DataFrame({'x':sure.col,'y':sure.row,'section':sure.section,'name':['machine_sure' for _ in range(len(sure))]})
    unsure = pd.DataFrame({'x':unsure.col,'y':unsure.row,'section':unsure.section,'name':['machine_unsure' for _ in range(len(unsure))]})
    return sure,unsure

def string_to_prediction(string):
    if string.split('_')[1] == 'sure':
        return 2
    elif string.split('_')[1] == 'unsure':
        return 0
    elif string.split('_')[1] == 'null':
        return -2

def get_cell_detections():
    Round1_SURE = 3581
    Round1_UNSURE = 3579
    # Round0_POSITIVE = 3575
    # Round0_NEGATIVE = 3577
    Round1_POSITIVE = 3574
    Round1_NEGATIVE = 3576
    Round2_UNSURE = 3578
    Round2_SURE = 3580
    Original_Positive = 3568
    controller = MarkedCellController()
    cell_detections = {}
    cell_detections['original_human_positive'] = controller.get_cells_from_sessioni(Original_Positive)
    # cell_detections['round1_machine_sure'] = controller.get_cells_from_sessioni(Round1_SURE)
    # cell_detections['round1_machine_unsure'] = controller.get_cells_from_sessioni(Round1_UNSURE)
    # cell_detections['round0_human_positive'] = controller.get_cells_from_sessioni(Round0_POSITIVE)
    # cell_detections['round0_human_negative'] = controller.get_cells_from_sessioni(Round0_NEGATIVE)
    cell_detections['round1_human_positive'] = controller.get_cells_from_sessioni(Round1_POSITIVE)
    cell_detections['round1_human_negative'] = controller.get_cells_from_sessioni(Round1_NEGATIVE)
    cell_detections['dbround2_machine_sure'] = controller.get_cells_from_sessioni(Round2_SURE)
    cell_detections['dbround2_machine_unsure'] = controller.get_cells_from_sessioni(Round2_UNSURE)
    for key in cell_detections:
        cell_detections[key] = pd.DataFrame(np.unique(cell_detections[key],axis = 0)/np.array([0.325,0.325,20]),columns= ['x','y','section']).join(pd.DataFrame({'name' : ['_'.join(key.split('_')[1:]) for _ in range(len(cell_detections[key]))]}))
    base = CellDetectorBase('DK55',round=1)
    detections = base.load_detections()
    cell_detections['round1_machine_sure'],cell_detections['round1_machine_unsure'] = parse_detections(detections)

    base = CellDetectorBase('DK55',round=2)
    detections = base.load_detections()
    cell_detections['round2_machine_sure'],cell_detections['round2_machine_unsure'] = parse_detections(detections)

    base = MultiThresholdDetector('DK55',round=2)
    detections = base.load_detections()
    predictions = np.array([string_to_prediction(i) for i in detections.name])
    sure = detections[predictions==2]
    unsure = detections[predictions==0]
    sure = pd.DataFrame({'x':sure.x,'y':sure.y,'section':sure.section,'name':['machine_sure' for _ in range(len(sure))]})
    unsure = pd.DataFrame({'x':unsure.x,'y':unsure.y,'section':unsure.section,'name':['machine_unsure' for _ in range(len(unsure))]})
    cell_detections['round2multi_machine_sure'], cell_detections['round2multi_machine_unsure'] = sure,unsure

    sorted_dict = {}
    for key in sorted(cell_detections):
        sorted_dict[key] = cell_detections[key]
    return sorted_dict