from cell_extractor.CellDetectorBase import CellDetectorBase 
from lib.TiffSegmentor import TiffSegmentor
import argparse

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--animal', type=str, help='Animal ID')
    parser.add_argument('--disk', type = str, help = 'Storage Disk')
    args = parser.parse_args()
    base = CellDetectorBase(args.animal,disk = args.disk)
    unfinished = base.get_sections_without_example()
    for fi in unfinished:
        print(fi)
