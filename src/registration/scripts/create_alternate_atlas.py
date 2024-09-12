import argparse
import numpy as np
import os
from skimage import io
from tqdm import tqdm
import cv2
from scipy.ndimage import zoom
import sys
from pathlib import Path

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())


from library.utilities.utilities_process import read_image, write_image



def create_boundary_mesh(animal):

    ROOT = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps/CH1'

    boundary_tiff_path = os.path.join(ROOT, 'allen_mouse_25um', 'annotation.tiff')
    volume = io.imread(boundary_tiff_path)

    print(f'volume info {volume.shape}, {volume.dtype}')

    fpath = os.path.join(ROOT, 'thumbnail_aligned')
    files = os.listdir(fpath)
    z = len(files)
    midfile = str(z // 2).zfill(3) + ".tif"
    midfilepath = os.path.join(ROOT, f'thumbnail_aligned/{midfile}')
    mid_arr = io.imread(midfilepath)
    print(f'midfile into {midfile}, {mid_arr.shape}, {z}') 

    boundary_outpath = os.path.join(ROOT, 'boundary')
    os.makedirs(boundary_outpath, exist_ok=True)
    arr = volume.copy()
    sagittal_arr = np.zeros((arr.shape[1], arr.shape[0], arr.shape[2]))
    endsection = arr.shape[2]   
    for i in tqdm(range(0, endsection, 1)):
        img = arr[:,:,i]
        img = np.rot90(img, 3)
        img = np.flip(img, axis=1)
        img[img > 0] = 255
        sagittal_arr[:,:,i] = img    

    sagittal_arr = sagittal_arr.astype(np.uint8)
    print(f'sagittal info {sagittal_arr.shape}, {sagittal_arr.dtype}')

    sagittal_np_path = os.path.join(ROOT, 'allen_mouse_25um', 'sagittal_boundary')
    np.save(sagittal_np_path, sagittal_arr)

    change_y = mid_arr.shape[0] / sagittal_arr.shape[0]
    change_x = mid_arr.shape[1] / sagittal_arr.shape[1]
    change_z = z / sagittal_arr.shape[2]
    print(f'Deltas between boundary and downsampled aligned images y={change_y}, x={change_x}, z={change_z}')

    zoomed_sagittal_arr = zoom(sagittal_arr, (change_y, change_x, change_z))
    del sagittal_arr
    print('zoomed volume info',zoomed_sagittal_arr.shape)
    print('original volume info', mid_arr.shape, z)

    zoomed_sagittal_np_path = os.path.join(ROOT, 'allen_mouse_25um', 'zoomed_sagittal_boundary')
    zoomed_sagittal_arr = zoomed_sagittal_arr.astype(np.uint8)
    np.save(zoomed_sagittal_np_path, zoomed_sagittal_arr)

    zoomed_boundary_outpath = os.path.join(ROOT, 'allen_mouse_25um', 'zoomed_boundary')
    os.makedirs(zoomed_boundary_outpath, exist_ok=True)
    endsection = zoomed_sagittal_arr.shape[2]   
    for i in tqdm(range(0, endsection, 1)):
        img = zoomed_sagittal_arr[:,:,i]
        img = img.astype(np.uint8)
        img[img > 0] = 255
        f = str(i).zfill(3) + '.tif'
        outpath = os.path.join(zoomed_boundary_outpath, f)
        cv2.imwrite(outpath, img)


def create_sagittal(um):

    ROOT = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
    atlas_path = os.path.join(ROOT, f'Allen_{um}um_coronal.tif')
    coronal = read_image(atlas_path)
    print(f'coronal info {coronal.shape}, {coronal.dtype}')

    swapped = np.swapaxes(coronal, 0,2)
    sagittal = swapped.astype(np.uint16)
    print(f'sagittal info {sagittal.shape}, {sagittal.dtype}')

    #allen10 800,1320,1140
    #DK41 1000,1800,456
    rowpad = 200
    colpad = 500
    padded = sagittal.copy()
    print(f'padded volume shape={padded.shape} dtype={padded.dtype}')

    padded = np.concatenate((padded, np.zeros((padded.shape[0], rowpad, padded.shape[2])) ), axis=1)
    print(f'padded volume shape={padded.shape} dtype={padded.dtype}')

    padded = np.concatenate((padded, np.zeros((padded.shape[0], padded.shape[1], colpad)) ), axis=2)
    padded = padded.astype(np.uint16)
    print(f'padded volume shape={padded.shape} dtype={padded.dtype}')

    outfile = 'Allen_10um_sagittal_padded.tif'
    outpath = os.path.join(ROOT, outfile)
    write_image(outpath, padded)
    return

    change_y = mid_arr.shape[0] / sagittal_arr.shape[0]
    change_x = mid_arr.shape[1] / sagittal_arr.shape[1]
    change_z = z / sagittal_arr.shape[2]
    print(f'Deltas between boundary and downsampled aligned images y={change_y}, x={change_x}, z={change_z}')

    zoomed_sagittal_arr = zoom(sagittal_arr, (change_y, change_x, change_z))
    del sagittal_arr
    print('zoomed volume info',zoomed_sagittal_arr.shape)
    print('original volume info', mid_arr.shape, z)

    zoomed_sagittal_np_path = os.path.join(ROOT, 'allen_mouse_25um', 'zoomed_sagittal_boundary')
    zoomed_sagittal_arr = zoomed_sagittal_arr.astype(np.uint8)
    np.save(zoomed_sagittal_np_path, zoomed_sagittal_arr)

    zoomed_boundary_outpath = os.path.join(ROOT, 'allen_mouse_25um', 'zoomed_boundary')
    os.makedirs(zoomed_boundary_outpath, exist_ok=True)
    endsection = zoomed_sagittal_arr.shape[2]   
    for i in tqdm(range(0, endsection, 1)):
        img = zoomed_sagittal_arr[:,:,i]
        img = img.astype(np.uint8)
        img[img > 0] = 255
        f = str(i).zfill(3) + '.tif'
        outpath = os.path.join(zoomed_boundary_outpath, f)
        cv2.imwrite(outpath, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=False)
    parser.add_argument('--um', help='Enter the um', required=False, default=25, type=int)
    args = parser.parse_args()
    animal = args.animal
    um = args.um
    #create_boundary_mesh(animal)
    create_sagittal(um)

