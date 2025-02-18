import sys
import numpy as np
import pickle as pkl
from cell_extractor import compute_image_features 
import cv2
import pandas as pd
from cell_extractor.CellDetectorBase import CellDetectorBase,get_sections_with_annotation_for_animali
import os
class MeanImageCalculator(CellDetectorBase):
    def __init__(self,animal, *args, **kwargs):
        super().__init__(animal, *args, **kwargs)
        self.calulate_average_cell_image()
    
    def calulate_average_cell_image(self):
        examples = self.load_all_examples_in_brain(label=1)
        self.average_image_ch1 = self.calculate_average_cell_images(examples,channel = 1)
        self.average_image_ch3 = self.calculate_average_cell_images(examples,channel = 3)
        average_image = dict(zip(['CH1','CH3'],[self.average_image_ch1,self.average_image_ch3]))
        pkl.dump(average_image,open(self.AVERAGE_CELL_IMAGE_DIR,'wb'))

    def calculate_average_cell_images(self,examples,channel = 3):
        images = []
        for examplei in examples:
            images.append(examplei[f'image_CH{channel}'])
        images = np.stack(images)
        average = np.average(images,axis=0)
        average = (average - average.mean())/average.std()
        return average

if __name__ == '__main__':
    calculator = MeanImageCalculator('DK52')