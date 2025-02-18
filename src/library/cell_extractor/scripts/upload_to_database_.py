from cell_extractor.CellDetectorBase import CellDetectorBase
base = CellDetectorBase(animal = 'DK52',segmentation_threshold=2000,round=2)
detections = base.load_detections()
print('')