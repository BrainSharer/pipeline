import argparse
import os
import tifffile
import numpy as np

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache



def create_sagittal_slices(resolution):
    base_dir = "/net/birdstore/Active_Atlas_Data/data_root"
    output_dir = os.path.join(base_dir, 'pipeline_data/Allen/preps/C1', f'source_aligned.{resolution}')
    os.makedirs(output_dir, exist_ok=True)
    # Instantiate with 10um resolution
    mcc = MouseConnectivityCache(resolution=resolution)
    annotation_volume, meta = mcc.get_annotation_volume()
    print(annotation_volume.shape, annotation_volume.dtype)
    print(meta)

    # To slice sagittally, change your array index axis:
    # The native array orientation is typically (Saggital/AP, Superior/DV, Left-Right/ML)
    # depending on your CCF version, allowing direct index slicing:
    
    for i in range(annotation_volume.shape[2]):
        img_int32 = annotation_volume[:, :, i]
        img_clipped = np.clip(img_int32, 0, 65535)
        img_uint16 = img_clipped.astype(np.uint16)        
        outfile = str(i).zfill(3)+ ".tif"
        outpath = os.path.join(output_dir, outfile)
        tifffile.imwrite(outpath, img_uint16)
        print(f'Wrote TIF to: {outpath}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Atlas')
    parser.add_argument('--debug', required=False, default='false', type=str)
    parser.add_argument('--um', required=False, default=10, type=int)

    args = parser.parse_args()
    debug = bool({'true': True, 'false': False}[args.debug.lower()])    
    um = args.um

 
    create_sagittal_slices(um)