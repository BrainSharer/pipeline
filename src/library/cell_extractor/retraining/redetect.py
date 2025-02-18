import sys
sys.path.append('/home/zhw272/programming/pipeline_utility/in_development/Will')
from cell_extractor.CellDetector import detect_cell,detect_cell_multithreshold
animals = ['DK50','DK40','DK46','DK52','DK41','DK55']
unprocessed = ['DK54']
# thresholds = [2000,2100,2200,2300,2700]
# for animal in unprocessed:
#     for threshold in thresholds:
#         detect_cell(animal = animal,segmentation_threshold = threshold,round = 2)

for animal in animals:
    detect_cell_multithreshold(animal = animal,round = 2)