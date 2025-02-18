import os,sys
sys.path.append(os.path.abspath('./../../'))
from Controllers.MarkedCellController import MarkedCellController,CellSources
from Controllers.SqlController import SqlController
from model.annotation_points import MarkedCellView
from sqlalchemy import inspect
inst = inspect(MarkedCellView)
attr_names = [c_attr.key for c_attr in inst.mapper.column_attrs]
import pandas as pd
from cell_extractor.AnnotationProximityTool import AnnotationProximityTool
from cell_extractor.CellDetector import CellDetector
from collections import Counter
import numpy as np
user_look_up = {38:'Marissa',41:'Julian',2:'Beth',3:'Hannah'}

def get_DataFrame_from_query_result(results,category,factor):
    values = []
    for i in results:
        source = i.source.value
        if '-' in source:
            source = source.split('-')[1]    
        x,y,z = np.array([i.x,i.y,i.z]).astype(float)/factor
        values.append([x,y,z,f'{category}_{user_look_up[i.FK_annotator_id]}_{source}'])
    df = pd.DataFrame(dict(zip(['x','y','section','name'],np.array(values).T)))
    df["x"] = pd.to_numeric(df["x"])
    df["y"] = pd.to_numeric(df["y"])
    df["section"] = pd.to_numeric(df["section"])
    return df

def get_DataFrame_from_detection_df(df,label):
    data = np.array([df.col.to_numpy().astype(int),df.row.to_numpy().astype(int),\
        df.section.to_numpy().astype(int)])
    data = pd.DataFrame(data.T,columns=['x','y','section'])
    data['name'] = [label]*len(df)
    return data

def get_all_qcs(animal):
    controller = MarkedCellController()
    search_dict = {'FK_prep_id':animal,'FK_cell_type_id':5}
    return controller.get_marked_cells(search_dictionary=search_dict)

def get_positive_and_negative_qcs(animal):
    qcs = get_all_qcs(animal)
    controller = SqlController()
    factor = controller.get_resolution(animal)
    positives = [i for i in qcs if i.source == CellSources.HUMAN_POSITIVE]
    negatives = [i for i in qcs if i.source == CellSources.HUMAN_NEGATIVE]
    positives = get_DataFrame_from_query_result(positives,'Round1',factor)
    negatives = get_DataFrame_from_query_result(negatives,'Round1',factor)
    

def print_human_machine_overlap(animal,annotator_id = 3):
    controller = SqlController()
    factor = controller.get_resolution(animal)
    controller = MarkedCellController()
    search_dict = {'FK_prep_id':animal,'FK_cell_type_id':1,'FK_annotator_id':annotator_id}
    cells = controller.get_marked_cells(search_dictionary=search_dict)
    cells = get_DataFrame_from_query_result(cells,'Round1',factor) 
    detector = CellDetector(animal,round=2)
    detections = detector.load_detections()
    sures = detections[detections.predictions==2]
    unsures = detections[detections.predictions==0]
    sures = get_DataFrame_from_detection_df(sures,'sures')
    unsures = get_DataFrame_from_detection_df(unsures,'unsures')
    tool = AnnotationProximityTool()
    tool.pair_distance=30
    tool.set_annotations_to_compare(pd.concat([sures,unsures,cells]))
    tool.find_equivalent_points()
    print(f'total sure {len(sures)} total unsure {len(unsures)}')
    for i in Counter([tuple(i) for i in tool.pair_categories.values()]).most_common():
        print(i)
        

def find_equivalence(points,distance = 0.1):
    tool = AnnotationProximityTool()
    tool.pair_distance=distance
    tool.set_annotations_to_compare(points)
    tool.find_equivalent_points()
    return tool

def find_agreement(tool,agree,disagree):
    agreed = tool.find_annotation_in_category(agree)
    disagreed = tool.find_annotation_in_category(disagree)
    return agreed,disagreed

def print_unique_combination(tool):
    dictionary = []
    for i in tool.pair_categories.values():
        if not i in dictionary:
            dictionary.append(i)
    for key,val in dictionary.items():
        print(key,val)