from Controllers.MarkedCellController import MarkedCellController
from model.annotation_points import CellSources
import pandas as pd
import numpy as np
animal = 'DK52'
detection  = pd.read_csv(f'/net/birdstore/Active_Atlas_Data/cell_segmentation/{animal}/detections_{animal}.2.csv')
sure = detection[detection.predictions==2]
controller = MarkedCellController()
coornidates = []
for _,row in sure.iterrows():
    pointi = np.array([row.col*0.325,row.row*0.325,row.section*20])
    coornidates.append(pointi)
controller.insert_marked_cells(coornidates,annotator_id = 34,prep_id = animal,cell_type_id = 6,type = CellSources.MACHINE_SURE)

