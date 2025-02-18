import sys
import os
sys.path.append(os.path.abspath('./../../../..'))
from cell_extractor.CellDetectorBase import CellDetectorBase
from lib.annotation_layer import Cell,random_string
import numpy as np
from lib.UrlGenerator import UrlGenerator
from cell_extractor.BorderFinder import BorderFinder
from cell_extractor.CellDetector import MultiThresholdDetector
from cell_extractor.CellAnnotationUtilities import CellAnnotationUtilities

def numpy_to_json(data,color_hex = None,category = 'Round3_Sure',description = None):
    cells = []
    for coord in data:
        cell = Cell(np.array(coord,dtype=float),random_string())
        cell.category = category
        cell.coord[-1]=cell.coord[-1]+0.5
        if hasattr(cell,'description'):
            del cell.description
        if description is not None:
            cell.description = description
        cell_json = cell.to_json()
        if color_hex is not None:
            cell_json["props"] = [color_hex]
        
        cells.append(cell_json)
    return cells

def create_QC_url(animal,sures,unsures,title):
    urlGen = UrlGenerator()
    urlGen.add_stack_image(animal,channel=1)
    urlGen.add_stack_image(animal,channel=2,color='red')
    urlGen.add_stack_image(animal,channel=3,color='green')
    urlGen.add_annotation_layer('Sure',annotations = sures)
    urlGen.add_annotation_layer('Unsure',annotations = unsures)
    return urlGen.add_to_database(title,34)

def get_multi_threshold_sure_and_unsure(animal):
    detector = MultiThresholdDetector(animal,round = 2)
    sure = detector.get_sures()
    unsure = detector.get_unsures()
    return sure,unsure

def get_single_threshold_sure_and_unsure(animal,threshold,round):
    detector = CellDetectorBase(animal,round = round,segmentation_threshold=threshold)
    detections = detector.load_detections()
    sures = detections[detections.predictions==2]
    unsure = detections[detections.predictions==0]
    nodetection = detections[detections.predictions<0]
    return sures,unsure,nodetection

def print_multi_threshold_sure_and_unsure_count(animal):
    sure,unsure = get_multi_threshold_sure_and_unsure(animal)
    print(len(sure),len(unsure))
    
def print_single_threshold_sure_and_unsure_count(animal,threshold = 2000,round=2):
    sure,unsure,_ = get_single_threshold_sure_and_unsure(animal,threshold,round)
    print(len(sure),len(unsure))

def generate_QC_link(animal,sample = None,round = 2,threshold = 2000):
    finder = BorderFinder(animal)
    detector = CellDetectorBase(animal,round = round,segmentation_threshold=threshold)
    detections = detector.load_detections()
    sure = detections[detections.predictions==2]
    unsure = detections[detections.predictions==0]
    sure.rename(columns = {'x' : 'col', 'y' : 'row'}, inplace = True)
    unsure.rename(columns = {'x' : 'col', 'y' : 'row'}, inplace = True)
    _,sure = finder.find_border_cells(sure)
    _,unsure = finder.find_border_cells(unsure)
    if sample is not None:
        sure_data_sample = sure[['col','row','section']].sample(sample).sort_values('section').to_numpy().tolist()
    else:
        sure_data_sample = sure[['col','row','section']].sort_values('section').to_numpy().tolist()
    sure_data = sure[['col','row','section']].sort_values('section').to_numpy().tolist()
    unsure_data = unsure[['col','row','section']].sort_values('section').to_numpy().tolist()
    sure_cells_sample = numpy_to_json(sure_data_sample,category = f'Round{round+1}_Sure')
    sure_cells = numpy_to_json(sure_data,category = f'Round{round+1}_Sure')
    unsure_cells = numpy_to_json(unsure_data,color_hex='#1d66db',category = f'Round{round+1}_Unsure')

    sc_id = create_QC_url(animal,sure_cells_sample,unsure_cells,f'Julian QC{round+1} '+animal)
    # false_negative_id = create_QC_url(animal,sure_cells,unsure_cells,'Julian false negative '+animal)
    print(f'Julian QC url id {animal} : https://activebrainatlas.ucsd.edu/ng/?id={sc_id}')
    # print(f'Julian false negative url id {animal} : https://activebrainatlas.ucsd.edu/ng/?id={false_negative_id}')

    sc_id = create_QC_url(animal,sure_cells_sample,unsure_cells,f'Marissa QC{round+1} '+animal)
    # false_negative_id = create_QC_url(animal,sure_cells,unsure_cells,'Marissa false negative '+animal)
    print(f'Marissa QC url {animal} : https://activebrainatlas.ucsd.edu/ng/?id={sc_id}')
    # print(f'Marissa false negative url id {animal} : https://activebrainatlas.ucsd.edu/ng/?id={false_negative_id}')
    
def get_ids_in_subcategory(big_category,subcatgory,max_distance=20):
    more_points = np.array([big_category.col,big_category.row,big_category.section]).T.astype(float)
    less_points = np.array([subcatgory.x,subcatgory.y,subcatgory.section]).T.astype(float)
    util = CellAnnotationUtilities()
    ids = util.find_cloest_neighbor_among_points(more_points,less_points,max_distance=max_distance)
    return ids
