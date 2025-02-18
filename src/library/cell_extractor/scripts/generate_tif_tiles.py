import os 
import sys
sys.path.append(os.path.abspath('./../../'))
from lib.TiffSegmentor import TiffSegmentor
import argparse
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--animal', type=str, help='Animal ID')
    parser.add_argument('--disk', type=str, help='Storage Disk')
    parser.add_argument('--njobs', type=int, help='Number of parallel jobs',default=10)
    args = parser.parse_args()
    animal = args.animal
    disk = args.disk
    njobs = args.njobs
    segmentor = TiffSegmentor(animal,disk = disk,n_workers = njobs)
    segmentor.generate_tiff_segments(channel = 1,create_csv = False)
    segmentor.generate_tiff_segments(channel = 3,create_csv = False)
