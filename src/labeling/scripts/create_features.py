import numpy as np
import pickle as pkl
import cv2
import os

from library.cell_labeling.cell_manager import CellMaker

class FeatureFinder(CellMaker):
    """class to calculate feature vector for each extracted image pair (CH1, CH3)
    """
    def __init__(self, animal, section, *args, **kwargs):
        super().__init__(animal, section, *args, **kwargs)
        self.features = []
