import matplotlib.pyplot as plt
import numpy as np
import os,sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/../../'))
from cell_extractor.diagnostics.ToolKit import get_DataFrame_from_query_result,find_equivalence,find_agreement,get_DataFrame_from_detection_df
from Controllers.MarkedCellController import MarkedCellController
from Controllers.SqlController import SqlController
import pandas as pd
from cell_extractor.CellAnnotationUtilities import CellAnnotationUtilities
from cell_extractor.AnnotationProximityTool import AnnotationProximityTool
from matplotlib.patches import Patch
from collections import Counter
from lib.UrlGenerator import UrlGenerator
from cell_extractor.QcGenerater import QcGenerater

animal = 'DK41'
controller = SqlController()
factor = controller.get_resolution(animal)

controller = MarkedCellController()
search_dict = {'FK_prep_id':animal,'FK_cell_type_id':26}
mixed = get_DataFrame_from_query_result(controller.get_marked_cells(search_dict),'Sure',factor)
mixed = find_equivalence(mixed)
