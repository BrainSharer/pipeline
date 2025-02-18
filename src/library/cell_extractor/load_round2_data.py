import numpy as np
import os,sys
sys.path.append(os.path.abspath('./../'))
from cell_extractor.diagnostics.ToolKit import get_DataFrame_from_query_result,find_equivalence,find_annotation_in_category,find_agreement,get_DataFrame_from_detection_df
from Controllers.MarkedCellController import MarkedCellController
from Controllers.SqlController import SqlController
from cell_extractor.CellAnnotationUtilities import CellAnnotationUtilities
from cell_extractor.CellDetectorBase import CellDetectorBase

def load_round2_data(animal):
    controller = SqlController()
    factor = controller.get_resolution(animal)
    controller = MarkedCellController()
    search_dict = {'FK_prep_id':animal,'FK_cell_type_id':16}
    round3_sures = get_DataFrame_from_query_result(controller.get_marked_cells(search_dict),'Sure',factor)
    round3_sures_tool = find_equivalence(round3_sures)
    search_dict = {'FK_prep_id':animal,'FK_cell_type_id':17}
    round3_unsures = get_DataFrame_from_query_result(controller.get_marked_cells(search_dict),'Unsure',factor)
    round3_unsures_tool = find_equivalence(round3_unsures)
    search_dict = {'FK_prep_id':animal,'FK_cell_type_id':18}
    false_negatives = get_DataFrame_from_query_result(controller.get_marked_cells(search_dict),'False_negative',factor)
    false_negatives_tool = find_equivalence(false_negatives,distance=5)
    positive = [['Sure_Marissa_POSITIVE', 'Sure_Julian_POSITIVE']]
    negative = [['Sure_Marissa_NEGATIVE', 'Sure_Julian_NEGATIVE']]
    disagree = [['Sure_Marissa_NEGATIVE', 'Sure_Julian_POSITIVE'],['Sure_Marissa_POSITIVE', 'Sure_Julian_NEGATIVE']]
    round3_sure_positive = find_annotation_in_category(round3_sures_tool,positive)
    round3_sure_negative = find_annotation_in_category(round3_sures_tool,negative)
    round3_sure_disagree = find_annotation_in_category(round3_sures_tool,disagree)
    positive = [['Unsure_Marissa_POSITIVE', 'Unsure_Julian_POSITIVE']]
    negative = [['Unsure_Marissa_NEGATIVE', 'Unsure_Julian_NEGATIVE']]
    disagree = [['Unsure_Marissa_POSITIVE', 'Unsure_Julian_NEGATIVE'],['Unsure_Marissa_NEGATIVE', 'Unsure_Julian_POSITIVE']]
    round3_unsure_positive = find_annotation_in_category(round3_unsures_tool,positive)
    round3_unsure_negative = find_annotation_in_category(round3_unsures_tool,negative)
    round3_unsure_disagree = find_annotation_in_category(round3_unsures_tool,disagree)
    agree = [['False_negative_Julian_POSITIVE', 'False_negative_Marissa_POSITIVE'],]
    disagree = [['False_negative_Julian_POSITIVE'],['False_negative_Marissa_POSITIVE']]
    false_negatives_agreed,false_negatives_disagreed = find_agreement(false_negatives_tool,agree,disagree)