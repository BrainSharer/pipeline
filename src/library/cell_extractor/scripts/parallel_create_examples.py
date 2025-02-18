from cell_extractor.ExampleFinder import create_examples_for_all_sections 
import argparse

if __name__ =='__main__':
    animal = 'DK52'
    for threshold in [2100,2200,2300,2700]:
        create_examples_for_all_sections(animal,disk = '/net/birdstore/Active_Atlas_Data/',segmentation_threshold=threshold,njobs=7)
