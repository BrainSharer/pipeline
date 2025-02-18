import os ,sys
sys.path.append(os.path.abspath('./../..'))
from cell_extractor.FeatureFinder import FeatureFinder
def calculate_one_section(animal,section,disk,segmentation_threshold):
    finder = FeatureFinder(animal,section = section,disk = disk,segmentation_threshold = segmentation_threshold)
    finder.calculate_features()
    finder.save_features()

if __name__ =='__main__':
    animal = 'DK39'
    section = 180
    disk = '/net/birdstore/Active_Atlas_Data'
    # for threshold in [2000,3000,4000]:
    calculate_one_section(animal,section,disk=disk,segmentation_threshold = 2100)
