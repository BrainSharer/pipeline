from .CellDetectorBase import CellDetectorBase
import os
import matplotlib.pyplot as plt
from tifffile import imread
import numpy as np
from scipy.signal import argrelextrema,find_peaks
import cv2
from concurrent.futures.process import ProcessPoolExecutor
import pickle
import pandas as pd

class BorderFinder(CellDetectorBase):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.BORDER_INFO = os.path.join(self.ANIMAL_PATH,'border_info.pkl')
        self.folder = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{self.animal}/preps/CH1/thumbnail_aligned/'
        self.SCALING_FACTOR = 0.03125

    def load_images(self):
        files = sorted(os.listdir(self.folder))
        self.imgs = []
        for file in files:
            img = imread(self.folder+file)
            self.imgs.append(img)

    def find_max_contour(self):
        self.max_contours = []
        for i,image in enumerate(self.imgs):
            mask = image>5000
            mask = np.array(mask*255).astype('uint8')
            _, thresh = cv2.threshold(mask, 200, 255, 0)
            contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            areas = area = [cv2.contourArea(i) for i in contours]
            max_id = np.argmax(areas)
            max_con = contours[max_id]
            max_con = max_con.reshape(-1,2)
            self.max_contours.append(max_con)

    def find_border_regions(self):
        file_keys = [self.max_contours,[i.shape for i in self.imgs]]
        with ProcessPoolExecutor(max_workers=5) as executor:
            results = executor.map(find_border, *file_keys)
        self.borders = []
        for i in results:
            self.borders.append(i)
        
    def calculate_border_regions(self):
        print('calculating borders for '+self.animal)
        self.load_images()
        print('image loaded')
        self.find_max_contour()
        print('contours found')
        self.find_border_regions()
        print('borders calculated')
        pickle.dump(self.borders,open(self.BORDER_INFO,'wb'))
    
    def load_border_regions(self):
        if os.path.isfile(self.BORDER_INFO):
            self.borders = pickle.load(open(self.BORDER_INFO,'rb'))
        else:
            self.calculate_border_regions()
            
    def find_border_cells(self,cells):
        self.load_border_regions()
        border_cell =[]
        good_cell = []
        for sectioni in cells.section.unique():
            df = cells[cells.section==sectioni]
            scale_factor = 0.03125
            data = np.array([df.row.to_list(),df.col.to_list()])*scale_factor
            border = self.borders[sectioni]
            npoints = data.shape[1]
            on_border = []
            for i in range(npoints):
                point = data[:,i].astype(int)
                on_border.append(border[point[0],point[1]]>0)
            on_border = np.array(on_border)
            border_cell.append(df[on_border])
            good_cell.append(df[np.logical_not(on_border)])
        border_cell = pd.concat(border_cell)
        good_cell = pd.concat(good_cell)
        return border_cell,good_cell

    def run_commands_in_parallel_with_executor(self, file_keys, workers, function):
        if self.function_belongs_to_a_pipeline_object(function):
            function = self.make_picklable_copy_of_function(function)
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = executor.map(function, *file_keys)
        return results

def find_border(max_con,shape):
    mask = np.zeros(shape)
    mask = cv2.fillPoly(mask, pts =[max_con.reshape(-1,1,2)], color=(255,255,255))
    # kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100,100))
    # morphed  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50,50))
    eroded  = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
    return np.logical_not(eroded)