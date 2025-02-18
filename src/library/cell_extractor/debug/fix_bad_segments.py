from cell_extractor.CellDetectorBase import CellDetectorBase 
from lib.TiffSegmentor import TiffSegmentor
import shutil
import argparse
import os 
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--animal', type=str, help='Animal ID')
    parser.add_argument('--disk', type=str, help='data root')
    args = parser.parse_args()
    animal = args.animal
    disk = args.disk
    base = CellDetectorBase(animal,disk = disk)
    unfinished = base.get_sections_without_example()
    for fi in unfinished:
        shutil.rmtree(os.path.join(base.CH3,f'{fi:03}'))
        shutil.rmtree(os.path.join(base.CH1,f'{fi:03}'))
    segmentor = TiffSegmentor(animal)
    segmentor.generate_tiff_segments(channel = 1,create_csv = False)
    segmentor.generate_tiff_segments(channel = 3,create_csv = True)