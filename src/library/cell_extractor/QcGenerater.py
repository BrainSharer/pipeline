from cell_extractor.BorderFinder import BorderFinder
from cell_extractor.CellDetectorBase import CellDetectorBase
from cell_extractor.utils import numpy_to_json
from Controllers.SqlController import SqlController
from cell_extractor.InterRatorResults import Round3DK41MarissaJulianResult,Round4DK41MarissaJulianResult,get_beth_positive_marking_for_DK41
from cell_extractor.utils import get_ids_in_subcategory
from cell_extractor.AnnotationProximityTool import AnnotationProximityTool,AcrossSectionProximity
from cell_extractor.diagnostics.ToolKit import get_DataFrame_from_detection_df
from collections import Counter
import pickle
import numpy as np
import os
import pandas as pd
class QcGenerater:
    def __init__(self,animal,round):
        self.animal = animal
        self.round = round
        self.base = CellDetectorBase(self.animal,round = self.round,segmentation_threshold=2000)
        self.load_detections()
        self.eliminate_border_cells()
        self.controller = SqlController(animal)
        self.eliminate_duplicate_sections()
        os.makedirs(os.path.join(self.base.ANIMAL_PATH,'QC'),exist_ok=True)
        self.QC_cell_path = os.path.join(self.base.ANIMAL_PATH,'QC',f'sure_and_unsure_without_duplication_round{round}_threshold_{2000}_animal_{animal}.pkl')
    
    def load_detections(self):
        self.detections = self.base.load_detections()
        self.detections = self.detections[self.detections.predictions>=0]

    def get_sure_and_unsure_cells(self,sure_and_unsure_sample):
        if os.path.exists(self.QC_cell_path):
            sure,unsure = pickle.load(open(self.QC_cell_path,'rb'))
        else:
            sure = self.detections[self.detections.predictions==2]
            unsure = self.detections[self.detections.predictions==0]
            unsure = self.delete_within_and_across_section_duplications(unsure)
            sure = self.delete_within_and_across_section_duplications(sure)
            sure.rename(columns = {'x' : 'col', 'y' : 'row'}, inplace = True)
            unsure.rename(columns = {'x' : 'col', 'y' : 'row'}, inplace = True)
            pickle.dump((sure,unsure),open(self.QC_cell_path,'wb'))
        sure_data = sure[['col','row','section']].sample(sure_and_unsure_sample[0]).sort_values('section').to_numpy().tolist()
        unsure_data = unsure[['col','row','section']].sample(sure_and_unsure_sample[1]).sort_values('section').to_numpy().tolist()
        sure_cells = numpy_to_json(sure_data,category = f'Round{self.round+1}_Sure')
        unsure_cells = numpy_to_json(unsure_data,color_hex='#1d66db',category = f'Round{self.round+1}_Unsure')
        return sure_cells,unsure_cells

    def get_mixed_cells(self,sure_and_unsure_sample):
        if os.path.exists(self.QC_cell_path):
            sure,unsure = pickle.load(open(self.QC_cell_path,'rb'))
        else:
            sure = self.detections[self.detections.predictions==2]
            unsure = self.detections[self.detections.predictions==0]
            unsure = self.delete_within_and_across_section_duplications(unsure)
            sure = self.delete_within_and_across_section_duplications(sure)
            sure.rename(columns = {'x' : 'col', 'y' : 'row'}, inplace = True)
            unsure.rename(columns = {'x' : 'col', 'y' : 'row'}, inplace = True)
            pickle.dump((sure,unsure),open(self.QC_cell_path,'wb'))
        sure_data = sure[['col','row','section']].sample(sure_and_unsure_sample[0]).sort_values('section').to_numpy().tolist()
        unsure_data = unsure[['col','row','section']].sample(sure_and_unsure_sample[1]).sort_values('section').to_numpy().tolist()
        mixed_cells = numpy_to_json(sure_data+unsure_data,category = f'mixed')
        return mixed_cells

    def eliminate_border_cells(self):
        finder = BorderFinder(self.animal)
        _,self.detections = finder.find_border_cells(self.detections)
    
    def eliminate_duplicate_sections(self):
        duplicate_sections = self.controller.get_duplicate_sections(self.animal)
        self.detections = self.detections[[i not in duplicate_sections[0] for i in self.detections.section]]

    def delete_same_section_duplicate_detections(self,detections):
        name = detections.name.unique()[0]
        tool = AnnotationProximityTool()
        tool.set_annotations_to_compare(detections)
        tool.find_equivalent_points()
        categories = Counter([tuple(i) for i in tool.pair_categories.values()]).most_common()
        categories = [list(i[0]) for i in categories if name in i[0]]
        single_detections = tool.find_annotation_in_category(categories)
        return single_detections

    def delete_across_section_duplicate_detections(self,detections):
        name = detections.name.unique()[0]
        tool = AcrossSectionProximity()
        tool.set_annotations_to_compare(detections)
        tool.find_equivalent_points()
        categories = Counter([tuple(i) for i in tool.pair_categories.values()]).most_common()
        categories = [list(i[0]) for i in categories if name in i[0] and list(i[0])!=[name]]
        index_to_delete = np.concatenate([tool.find_annotation_in_category([i]).index  for i in categories])
        single_detections = detections.drop(index_to_delete)
        return single_detections
    
    def delete_within_and_across_section_duplications(self,detections):
        detections = get_DataFrame_from_detection_df(detections,'detections')
        single_detections = self.delete_same_section_duplicate_detections(detections)
        single_detections = self.delete_across_section_duplicate_detections(single_detections)
        return single_detections

class QcGeneraterDK41(QcGenerater):
    def __init__(self,round):
        super().__init__('DK41',round)
        self.elimincate_existing_annotations()

    def elimincate_existing_annotations(self):
        inter_rator = Round3DK41MarissaJulianResult()
        round3_previous_detections = inter_rator.get_all_agreed_annotations()
        round3_annotated_ids = get_ids_in_subcategory(self.detections,round3_previous_detections)
        inter_rator = Round4DK41MarissaJulianResult()
        round4_previous_detections = inter_rator.get_all_agreed_annotations()
        round4_annotated_ids = get_ids_in_subcategory(self.detections,round4_previous_detections)
        beth_annotation = get_beth_positive_marking_for_DK41()
        beth_annotated_ids = get_ids_in_subcategory(self.detections,beth_annotation)
        all_annotated_ids = round3_annotated_ids+round4_annotated_ids+beth_annotated_ids
        self.detections = self.detections[[i not in all_annotated_ids for i in range(len(self.detections))]]

