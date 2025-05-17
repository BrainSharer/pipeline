# .mat are actually dictionnary. This function support .mat from
# antsRegistration that encode a 4x4 transformation matrix.
import argparse
import shutil
import ants
import os
import sys
from pathlib import Path
from scipy.ndimage import zoom
import numpy as np

import dask.array as da
from dask.diagnostics import ProgressBar

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.utilities.utilities_process import read_image, write_image

class AntsRegistration:
    """
    This class is used to register images using ANTsPy
    """

    def __init__(self, moving,  xy_um, z_um, transformation='Affine', debug=False):
        self.moving = moving
        self.fixed = 'Allen'
        self.xy_um = xy_um
        self.z_um = z_um
        self.transformation = transformation
        self.debug = debug
        self.reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'


    def create_registration(self):

        moving_path = os.path.join(self.reg_path, self.moving)
        fixed_path = os.path.join(self.reg_path, self.fixed)
        fixed_filepath = os.path.join(fixed_path, f'{self.fixed}_{self.xy_um}um_sagittal.tif')
        moving_filepath = os.path.join(moving_path, f'{self.moving}_{self.xy_um}x{self.xy_um}x{self.xy_um}um_sagittal.tif')

        if not os.path.isfile(moving_filepath):
            print(f"Moving image not found at {moving_filepath}")
            exit(1)
        else:
            print(f"Moving image found at {moving_filepath}.")

        if not os.path.isfile(fixed_filepath):
            print(f"Fixed image not found at {fixed_path}")
            exit(1)
        else:
            print(f"Fixed image found at {fixed_filepath}")
                                        

        moving_image = ants.image_read(moving_filepath)
        fixed_image = ants.image_read(fixed_filepath)
        tx = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform = (self.transformation) )
        print(tx)
        mywarpedimage = ants.apply_transforms( fixed=fixed_image, moving=moving_image, transformlist=tx['fwdtransforms'], defaultvalue=0 )
        outpath = os.path.join(moving_path, f'{self.moving}_{self.fixed}_{self.um}um_sagittal.tif')
        ants.image_write(mywarpedimage, outpath)

        original_filepath = tx['fwdtransforms'][0]
        transform_filepath = os.path.join(moving_path, f'{self.moving}_{self.fixed}_{um}um_sagittal_to_Allen.mat')
        shutil.move(original_filepath, transform_filepath)


    def create_matrix(self):
        new_filepath = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/ALLEN771602/ls_to_template_rigid_0GenericAffine.mat'
        transfo_dict = loadmat(new_filepath)
        lps2ras = np.diag([-1, -1, 1])

        for k,v in transfo_dict.items():
            print(k, v)


        rot = transfo_dict['AffineTransform_float_3_3'][0:9].reshape((3, 3))
        trans = transfo_dict['AffineTransform_float_3_3'][9:12]
        offset = transfo_dict['fixed']
        r_trans = (np.dot(rot, offset) - offset - trans).T * [1, 1, -1]


        matrix = np.eye(4)
        matrix[0:3, 3] = r_trans
        matrix[:3, :3] = np.dot(np.dot(lps2ras, rot), lps2ras)

        print(matrix)

        translation = (matrix[..., 3][0:3])
        #translation = 0
        matrix = matrix[:3, :3]
        volume_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/ALLEN771602/ALLEN771602_25um_sagittal.tif'
        volume = read_image(volume_path)
        transformed_volume = affine_transform(volume, matrix, offset=translation, order=1)
        outpath = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/ALLEN771602/ALLEN771602_25um_sagittal_transformed.tif'
        write_image(outpath, transformed_volume)
        print(f'Wrote transformed volume to {outpath}')

    def create_big_volume(self):
        reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/Allen'
        allenpath = os.path.join(reg_path, 'Allen_10um_sagittal_padded.tif')
        allen_arr = read_image(allenpath)
        change_z = 10/self.z_um
        change_y = 10/self.xy_um
        change_x = 10/self.xy_um
        new_shape = (int(allen_arr.shape[0] * change_z), int(allen_arr.shape[1] * change_y), int(allen_arr.shape[2] * change_x))
        scaling_factors = (change_z, change_y, change_x)
        chunk_size = (64,64,64)
        print(f'change_z={change_z} change_y={change_y} change_x={change_x} {new_shape=} {chunk_size=}')

        zoomed = zoom_large_3d_array(allen_arr, scale_factors=scaling_factors, chunks=chunk_size)

        with ProgressBar():
            result = zoomed.compute()

        outpath = os.path.join(reg_path, f'Allen_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal.tif')
        write_image(outpath, result)
        print(f'Wrote zoomed volume to {outpath}')
        print(f'With shape {result.shape} and dtype {result.dtype}')

def zoom_large_3d_array(input_array, scale_factors, chunks=(100, 100, 100)):
    """
    Zoom a large 3D array using chunked processing with Dask.

    Parameters:
    - input_array: NumPy ndarray or Dask array (3D)
    - scale_factors: Tuple of (z_scale, y_scale, x_scale)
    - chunks: Tuple defining chunk sizes (z, y, x)

    Returns:
    - zoomed Dask array
    """

    def zoom_block(block, block_info=None):
        block_zoom = tuple(s for s in scale_factors)
        return zoom(block, zoom=block_zoom, order=1)

    # Convert to Dask array if necessary
    if not isinstance(input_array, da.Array):
        input_array = da.from_array(input_array, chunks=chunks)

    # Map zoom function to each block (adjusted scale to preserve continuity)
    zoomed = input_array.map_blocks(zoom_block, dtype=input_array.dtype)
    return zoomed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--moving', help='Enter the animal (moving)', required=True, type=str)
    parser.add_argument("--xy_um", help="Enter xy um", required=True, default=10, type=float)
    parser.add_argument("--z_um", help="Enter z um", required=True, default=10, type=float)
    parser.add_argument("--task", help="Enter the task you want to perform", required=True, default="status", type=str)
    parser.add_argument("--transformation", help="Enter the transformation you want to perform", required=False, default="Affine", type=str)
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)
    
    args = parser.parse_args()
    moving = args.moving
    xy_um = args.xy_um
    z_um = args.z_um
    task = str(args.task).strip().lower()
    transformation = str(args.transformation).strip()
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])

    pipeline = AntsRegistration(moving, xy_um, z_um, transformation, debug=debug)


    function_mapping = {'zoom_volume': pipeline.create_big_volume,
                        'create_matrix': pipeline.create_matrix,
                        'create_registration': pipeline.create_registration,
    }

    if task in function_mapping:
        function_mapping[task]()
    else:
        print(f'{task} is not a correct task. Choose one of these:')
        for key in function_mapping.keys():
            print(f'\t{key}')

