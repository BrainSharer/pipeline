import pyvista as pv
import os
from tqdm import tqdm
import tifffile as tiff
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from vedo import Volume, show
import sys

INPUT = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/X/preps/C1/full/"
files = sorted(os.listdir(INPUT))
scaling_factor = 20
len_files = len(files) // scaling_factor
midpoint = len_files // 2
infile = os.path.join(INPUT, files[midpoint])
midfile = tiff.imread(infile)
grid = pv.ImageData()
rows = midfile.shape[0]//scaling_factor
cols = midfile.shape[1]//scaling_factor
grid = pv.ImageData(dimensions=(rows, cols, len_files))
print(f'Dimensions={grid.dimensions}')
volume = np.empty((rows, cols, len_files))

index = 0
for i in tqdm(range(0, len_files, scaling_factor)):
    if index == len_files // scaling_factor:
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

    #fdata = data.flatten(order="F")
    #print(f'fdata shape={fdata.shape} XxY={rows*cols}')
    volume[:,:,index] = data
    index += 1

"""
ids, counts = np.unique(volume, return_counts=True)
print(ids)
print(counts)
sys.exit()
grid.point_data["data"] = volume.flatten(order="F")
grid.spacing = (1,1,1)
print(grid)
threshed = grid.threshold_percent((0.002,0.4))
surf = threshed.extract_surface()
grid.plot(cmap='gist_earth_r', show_scalar_bar=True, show_edges=True)
print(surf)
pv.save_meshio('test.obj', surf)
"""
vol = Volume(volume).print()
iso = vol.isosurface(35)
iso.write("iso.obj")
show(vol, iso, N=2, axes=1)