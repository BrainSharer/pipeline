import sys
sys.path.append('/home/zhw272/programming/pipeline_utility/in_development/Will/')
from cell_extractor.FeatureFinder import create_features_for_all_sections 
from cell_extractor.CellDetectorBase import CellDetectorBase
import argparse

if __name__ =='__main__':
    base = CellDetectorBase()
    animals = base.get_available_animals()
    animals.remove('DK39')
    for animali in animals:
        print(f'working on animal {animali}')
        create_features_for_all_sections(animali,disk = '/net/birdstore/Active_Atlas_Data/',segmentation_threshold=2000)
