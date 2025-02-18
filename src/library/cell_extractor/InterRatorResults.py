from Controllers.MarkedCellController import MarkedCellController
from Controllers.SqlController import SqlController
from cell_extractor.diagnostics.ToolKit import get_DataFrame_from_query_result,find_equivalence,find_agreement,get_DataFrame_from_detection_df
import pandas as pd

class DK41InterRator:
    def __init__(self):
        self.animal = 'DK41'
        controller = SqlController()
        self.factor = controller.get_resolution(self.animal)
        self.controller = MarkedCellController()
    
    def find_agreement_in_cell_type(self,cell_type_id,categories):
        search_dict = {'FK_prep_id':self.animal,'FK_cell_type_id':cell_type_id}
        cells = get_DataFrame_from_query_result(self.controller.get_marked_cells(search_dict),'Sure',self.factor)
        equivalence_tool = find_equivalence(cells)
        for categoryi in categories:
            yield equivalence_tool.find_annotation_in_category(categoryi)

    def get_sure_result(self):
        positive = [['Sure_Marissa_POSITIVE', 'Sure_Julian_POSITIVE']]
        negative = [['Sure_Marissa_NEGATIVE', 'Sure_Julian_NEGATIVE']]
        disagree = [['Sure_Marissa_NEGATIVE', 'Sure_Julian_POSITIVE'],['Sure_Marissa_POSITIVE', 'Sure_Julian_NEGATIVE']]
        return self.find_agreement_in_cell_type(self.sure_id,[positive,negative,disagree])

    def get_unsure_result(self):
        positive = [['Unsure_Marissa_POSITIVE', 'Unsure_Julian_POSITIVE']]
        negative = [['Unsure_Marissa_NEGATIVE', 'Unsure_Julian_NEGATIVE']]
        disagree = [['Unsure_Marissa_POSITIVE', 'Unsure_Julian_NEGATIVE'],['Unsure_Marissa_NEGATIVE', 'Unsure_Julian_POSITIVE']]
        return self.find_agreement_in_cell_type(self.unsure_id,[positive,negative,disagree])
    
    def get_false_negative_result(self):
        agree = [['False_negative_Julian_POSITIVE', 'False_negative_Marissa_POSITIVE'],]
        disagree = [['False_negative_Julian_POSITIVE'],['False_negative_Marissa_POSITIVE']]
        return self.find_agreement_in_cell_type(self.false_negative_id,[agree,disagree])

class Round3DK41MarissaJulianResult(DK41InterRator):
    def __init__(self):
        super().__init__()
        self.sure_id = 16
        self.unsure_id = 17
    
    def get_all_agreed_annotations(self):
        round3_sure_positive,round3_sure_negative,_ = self.get_sure_result()
        round3_unsure_positive,round3_unsure_negative,_ = self.get_unsure_result()
        false_negatives_agreed,_ = self.get_false_negative_result()
        return pd.concat([round3_sure_positive,round3_sure_negative,round3_unsure_positive,\
            round3_unsure_negative,false_negatives_agreed])

def get_beth_positive_marking_for_DK41():
    animal = 'DK41'
    controller = SqlController()
    factor = controller.get_resolution(animal)
    controller = MarkedCellController()
    search_dict = {'FK_prep_id':animal,'FK_cell_type_id':1,'FK_annotator_id':2}
    return get_DataFrame_from_query_result(controller.get_marked_cells(search_dict),'Original',factor)

class Round4DK41MarissaJulianResult(DK41InterRator):
    def __init__(self):
        super().__init__()
        self.sure_id = 22
        self.unsure_id = 23
    
    def get_all_agreed_annotations(self):
        sure_positive,sure_negative,_ = self.get_sure_result()
        unsure_positive,unsure_negative,_ = self.get_unsure_result()
        return pd.concat([sure_positive,unsure_positive,sure_negative,unsure_negative])