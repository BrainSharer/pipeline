import numpy as np
import pickle as pkl
import cv2
import os

import compute_image_features 
from cell_detector_base import CellDetectorBase, parallel_process_all_sections

class FeatureFinder(CellDetectorBase):
    """class to calculate feature vector for each extracted image pair (CH1, CH3)
    """
    def __init__(self,animal,section, *args, **kwargs):
        super().__init__(animal,section, *args, **kwargs)
        del self.sqlController
        self.features = []
        print('DATA_DIR=%s'%(self.CH3))
        print(f'working on section {self.section}')
        self.load_average_cell_image()
    
    def copy_information_from_examples(self,example):
        for key in ['animal','section','index','label','area','height','width']:
            self.featurei[key] = example[key]
        self.featurei['row'] = example['row']+example['origin'][0]
        self.featurei['col'] = example['col']+example['origin'][1]

    def calculate_correlation_and_energy(self,example,channel = 3):  # calculate correlation and energy for channel 1 and 3
        image = example[f'image_CH{channel}']
        average_image = getattr(self, f'average_image_ch{channel}')
        corr,energy = compute_image_features.calc_img_features(image,average_image)
        self.featurei[f'corr_CH{channel}'] = corr
        self.featurei[f'energy_CH{channel}'] = energy
    
    def features_using_center_connectd_components(self,example):   
        def mask_mean(mask,image):
            mean_in=np.mean(image[mask==1])
            mean_all=np.mean(image.flatten())
            return (mean_in-mean_all)/(mean_in+mean_all)    # calculate the contrast: mean

        def append_string_to_every_key(dictionary, post_fix): 
            return dict(zip([keyi + post_fix for keyi in dictionary.keys()],dictionary.values()))
        
        def calc_moments_of_mask(mask):   # calculate moments (how many) and Hu Moments (7)
            mask = mask.astype(np.float32)
            moments = cv2.moments(mask)
            """
            Moments(
                    double m00,
                    double m10,
                    double m01,
                    double m20,
                    double m11,
                    double m02,
                    double m30,
                    double m21,
                    double m12,
                    double m03
                    );
            Hu Moments are described in this paper: https://www.researchgate.net/publication/224146066_Analysis_of_Hu's_moment_invariants_on_image_scaling_and_rotation
            """
            huMoments = cv2.HuMoments(moments)
            moments = append_string_to_every_key(moments,f'_mask')
            self.featurei.update(moments)
            self.featurei.update({'h%d'%i+f'_mask':huMoments[i,0]  for i in range(7)})
        
        def calc_contrasts_relative_to_mask(mask,image1,image3):
            self.featurei['contrast1']=mask_mean(mask,image1)
            self.featurei['contrast3']=mask_mean(mask,image3)

        image1 = example['image_CH1']
        image3 = example['image_CH3']
        mask = example['mask']  
        calc_moments_of_mask(mask)
        calc_contrasts_relative_to_mask(mask,image1,image3)

    def calculate_features(self):
        """ Master function, calls methods to calculate the features that are then stored in self.features
        """
        self.load_examples()
        for tilei in range(len(self.Examples)):
            print(f'processing {tilei}')
            examples_in_tilei = self.Examples[tilei]
            if examples_in_tilei != []:
                for examplei in range(len(examples_in_tilei)):
                    example=examples_in_tilei[examplei]
                    self.featurei={}
                    self.copy_information_from_examples(example)
                    self.calculate_correlation_and_energy(example,channel=1)
                    self.calculate_correlation_and_energy(example,channel=3)
                    self.features_using_center_connectd_components(example)
                    self.features.append(self.featurei)

def create_features_for_all_sections(animal,*args,njobs = 10,**kwargs):
    parallel_process_all_sections(animal,create_features_for_one_section,*args,njobs = njobs,**kwargs)

def create_features_for_one_section(animal,section,*args,**kwargs):
    finder = FeatureFinder(animal,section = section,*args,**kwargs)
    if not os.path.exists(finder.get_feature_save_path()):
        finder.calculate_features()
        finder.save_features()
