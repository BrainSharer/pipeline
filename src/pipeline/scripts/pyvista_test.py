import os
import sys
from tqdm import tqdm
import tifffile as tiff
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from vedo import Volume

INPUT = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/X/preps/C1/full/"
files = sorted(os.listdir(INPUT))
scaling_factor = 10
len_files = len(files) // scaling_factor
midpoint = len_files // 2
infile = os.path.join(INPUT, files[midpoint])
midfile = tiff.imread(infile)
rows = midfile.shape[0]//scaling_factor
cols = midfile.shape[1]//scaling_factor
volume = np.empty((rows, cols, len_files), dtype=np.uint8)
print(f'Shape of volume={volume.shape}')
#sys.exit()

index = 0
for i in tqdm(range(0, len(files), scaling_factor)):
    if index == len_files - 1:
        print(f'breaking at index={index}')
        break
    infile = os.path.join(INPUT, files[i])            
    try:
        im = Image.open(infile)
    except IOError as ioe:
        print(f'could not open {infile} {ioe}')
    
    try:
        width, height = im.size
        im = im.resize((width//scaling_factor, height//scaling_factor))
        img = np.array(im)

        data = img.astype(np.uint8)
        data[data > 0] = 255
    except:
        print(f'could not resize {infile} with shape={img.shape}')

    volume[:,:,index] = data
    index += 1



ids, counts = np.unique(volume, return_counts=True)
print(f'volume dtype={volume.dtype}')
print(f'ids={ids}')
print(f'counts={counts}')
vol = Volume(volume).print().gaussianSmooth(sigma=0.5)
iso = vol.isosurface()
iso.write("vedo.obj")
