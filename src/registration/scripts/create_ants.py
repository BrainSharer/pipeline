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
from tqdm import tqdm
import dask.array as da
from tifffile import TiffWriter
import zarr
from numcodecs import Blosc
from dask.diagnostics import ProgressBar

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.image_manipulation.filelocation_manager import FileLocationManager
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
        self.fileLocationManager = FileLocationManager(moving)
        self.moving_path = os.path.join(self.reg_path, self.moving)
        self.fixed_path = os.path.join(self.reg_path, self.fixed)
        self.fixed_filepath = os.path.join(self.fixed_path, f'{self.fixed}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal.tif')
        self.moving_filepath = os.path.join(self.moving_path, f'{self.moving}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal.tif')
        self.transform_filepath = os.path.join(self.moving_path, f'{self.moving}_{self.fixed}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal_to_Allen.mat')



    def create_registration(self):
        print('Starting registration')

        if not os.path.isfile(self.moving_filepath):
            print(f"Moving image not found at {self.moving_filepath}")
            exit(1)
        else:
            print(f"Moving image found at {self.moving_filepath}")

        if not os.path.isfile(self.fixed_filepath):
            print(f"Reference image not found at {self.fixed_filepath}")
            exit(1)
        else:
            print(f"Reference image found at {self.fixed_filepath}")
                                        
        moving_image = ants.image_read(self.moving_filepath)
        print(f"Moving image loaded and shape: {moving_image.shape} dtype: {moving_image.dtype}")
        reference = ants.image_read(self.fixed_filepath)
        print(f"Reference image loaded and shape: {reference.shape} dtype: {reference.dtype}")

        if os.path.isfile(self.transform_filepath):
            print(f"Transform file already exists at {self.transform_filepath}")
        else:
            print(f"Transform file not and now creating ...")
            tx = ants.registration(fixed=reference, moving=moving_image, 
                               type_of_transform = (self.transformation))
            original_filepath = tx['fwdtransforms'][0]
            shutil.move(original_filepath, self.transform_filepath)
            print(f"Transform file moved to {self.transform_filepath}")

        mywarpedimage = ants.apply_transforms( fixed=reference, moving=moving_image, 
                                              transformlist=self.transform_filepath, defaultvalue=0)
        outpath = os.path.join(self.moving_path, f'{self.moving}_{self.fixed}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal.tif')
        ants.image_write(mywarpedimage, outpath)
        print(f'Wrote transformed volume to {outpath}')



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
        inpath = os.path.join(self.reg_path, self.moving, f'{self.moving}_10um_sagittal.tif')
        if not os.path.isfile(inpath):
            print(f"File not found at {inpath}")
            exit(1)
        else:
            print(f"File found at {inpath}.")
        outpath = os.path.join(self.reg_path, self.moving, f'{self.moving}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal.tif')
        arr = read_image(inpath)
        change_z = 10/self.z_um
        change_y = 10/self.xy_um
        change_x = 10/self.xy_um
        scale_factors = (change_z, change_y, change_x)
        print(f'change_z={change_z} change_y={change_y} change_x={change_x} {arr.shape=} {arr.dtype=}')
        zoomed = zoom(arr, scale_factors)
        # Write incrementally to TIFF
        with TiffWriter(outpath, bigtiff=True) as tif:
            for i in tqdm(range(zoomed.shape[0])):
                slice = zoomed[i]
                tif.write(slice.astype(arr.dtype), contiguous=True)

        print(f'Wrote zoomed volume to {outpath}')

    def create_big_dask_volume(self):
        inpath = os.path.join(self.reg_path, self.moving, f'{self.moving}_10um_sagittal.tif')
        if not os.path.isfile(inpath):
            print(f"File not found at {inpath}")
            exit(1)
        else:
            print(f"File found at {inpath}.")
        arr = read_image(inpath)
        change_z = 10/self.z_um
        change_y = 10/self.xy_um
        change_x = 10/self.xy_um
        chunk_size = 'auto'
        scale_factors = (change_z, change_y, change_x)
        print(f'change_z={change_z} change_y={change_y} change_x={change_x} {chunk_size=} {arr.shape=} {arr.dtype=}')
        outpath = os.path.join(self.reg_path, self.moving, f'{self.moving}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal.zarr')
        if os.path.exists(outpath):
            print(f"Zarr file already exists at {outpath}")
        else:
            zoomed = zoom_large_3d_array(arr, scale_factors=scale_factors, chunks=chunk_size)
            # Save to Zarr
            with ProgressBar():
                da.to_zarr(zoomed, outpath, overwrite=True, mode='w')

        print(f"Written scaled volume to: {outpath}")
        volume = zarr.open(outpath, 'r')
        print(volume.info)
        print(f'volume.shape={volume.shape}')
        return


        # Write incrementally to TIFF
        with TiffWriter(outpath, bigtiff=True) as tif:
            for i in tqdm(range(zoomed.shape[0])):
                slice = zoomed[i]
                slice_i = slice.compute()
                tif.write(slice_i.astype(allen_arr.dtype), contiguous=True)

        print(f'Wrote zoomed volume to {outpath}')

    def split_big_volume(self):
        filepath = os.path.join(self.reg_path, self.moving, 'ALLEN771602_Allen_8um_sagittal.tif')
        if not os.path.isfile(filepath):
            print(f"File not found at {filepath}")
            exit(1)
        else:
            print(f"File found at {filepath}.")
        arr = read_image(filepath)
        outpath = os.path.join(self.fileLocationManager.prep, 'C1', f'{self.xy_um}_{self.z_um}')
        os.makedirs(outpath, exist_ok=True) 
        for i in tqdm(range(arr.shape[0])):
            slice_i = arr[i]
            slice_i = slice_i.astype(np.uint16)
            outpath_slice = os.path.join(outpath, f'{str(i).zfill(4)}.tif')
            write_image(outpath_slice, slice_i)
            print(f'Wrote slice {i} to {outpath_slice}')

    def repack_big_volume(self):
        filespath = os.path.join(self.fileLocationManager.prep, 'C1', str(0))
        if not os.path.isdir(filespath):
            print(f"Dir not found at {filespath}")
            exit(1)
        else:
            print(f"Dir found at {filespath}.")
        output_tif_path = os.path.join(self.reg_path, self.moving, f'{self.moving}_{self.xy_um}x{self.xy_um}x{self.z_um}um_sagittal.tif')
        files = sorted(os.listdir(filespath))
        # Write incrementally to TIFF
        with TiffWriter(output_tif_path, bigtiff=True) as tif:
            for file in tqdm(files):
                filepath = os.path.join(filespath, file)
                img = read_image(filepath)
                if img.ndim != 2:
                    img = img.reshape((img.shape[-2], img.shape[-1]))
                #print(f'img shape: {img.shape} img dtype: {img.dtype}')
                tif.write(img, contiguous=True)
                del img


def zoom_large_3d_array_to_tiff(input_array_path, output_tif_path, shape, dtype, scale_factors, chunk_size=(64, 64, 64)):
    """
    Zoom a large 3D array and save it incrementally to a TIFF file.

    Parameters:
        input_array_path (str): Path to a .npy or .zarr file storing the array.
        output_tif_path (str): Destination path for the TIFF file.
        shape (tuple): Original shape of the array (z, y, x).
        dtype (np.dtype): Data type of the array.
        scale_factors (tuple): Scaling factors (z_scale, y_scale, x_scale).
        chunk_size (tuple): Chunk size for dask processing.
    """

    # Create a Dask array from the file (memory-mapped if .npy)
    if input_array_path.endswith('.npy'):
        memmap = np.load(input_array_path, mmap_mode='r')
        darr = da.from_array(memmap, chunks=chunk_size)
    elif input_array_path.endswith('.tif'):
        darr = da.from_array(read_image(input_array_path), chunks=chunk_size)
    else:
        darr = da.from_zarr(input_array_path)

    zf, yf, xf = scale_factors

    # Calculate new shape
    new_shape = tuple(int(s * f) for s, f in zip(shape, scale_factors))

    # Define a function for dask to map over blocks
    def blockwise_zoom(block):
        zoom_factors_block = (
            zf * block.shape[0] / chunk_size[0],
            yf * block.shape[1] / chunk_size[1],
            xf * block.shape[2] / chunk_size[2],
        )
        return zoom_chunk(block, zoom_factors_block)

    # Map zoom over all blocks
    zoomed = darr.map_blocks(blockwise_zoom, dtype=dtype)

    # Write incrementally to TIFF
    with TiffWriter(output_tif_path, bigtiff=True) as tif:
        for i in tqdm(range(zoomed.shape[0])):
            slice_i = zoomed[i].compute()
            tif.write(slice_i.astype(dtype), contiguous=True, compression='LZW')

    print(f"Saved zoomed volume to: {output_tif_path}")

def zoom_chunk(chunk, zoom_factors):
    return zoom(chunk, zoom_factors, order=1)  # linear interpolation

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

def apply_ants_transform_zarr(input_zarr_path, output_zarr_path, transform_list, reference_image_path, chunk_size=(64, 64, 64)):
    """
    Apply ANTs transform to a large 3D Zarr volume in chunks.

    Parameters:
    - input_zarr_path: path to the input Zarr array (3D).
    - output_zarr_path: path to the output Zarr array.
    - transform_list: list of ANTs transform files (e.g., from ants.registration).
    - reference_image_path: path to reference image (defines output space).
    - chunk_size: tuple defining the size of chunks to process at a time.
    """

    input_zarr = zarr.open(input_zarr_path, mode='r')
    shape = input_zarr.shape
    dtype = input_zarr.dtype

    # Create output zarr with same shape and dtype
    compressor = Blosc(cname='zstd', clevel=3)
    output_zarr = zarr.open(output_zarr_path, mode='w')
    output_array = output_zarr.create('data', shape=shape, chunks=chunk_size, dtype=dtype, compressor=compressor)

    # Load reference image
    reference = ants.image_read(reference_image_path)

    # Iterate through chunks
    for z in range(0, shape[0], chunk_size[0]):
        for y in range(0, shape[1], chunk_size[1]):
            for x in tqdm(range(0, shape[2], chunk_size[2])):
                z0, y0, x0 = z, y, x
                z1, y1, x1 = min(z+chunk_size[0], shape[0]), min(y+chunk_size[1], shape[1]), min(x+chunk_size[2], shape[2])
                
                # Load chunk
                chunk = input_zarr[z0:z1, y0:y1, x0:x1]
                chunk_img = ants.from_numpy(chunk.astype(np.float32), origin=(x0, y0, z0), spacing=reference.spacing, direction=reference.direction)

                # Apply transform
                warped_chunk = ants.apply_transforms(fixed=reference, moving=chunk_img, transformlist=transform_list, interpolator='linear')
                
                # Write transformed chunk to output Zarr
                transformed_array = warped_chunk.numpy()
                output_array[z0:z1, y0:y1, x0:x1] = transformed_array

                print(f"Processed chunk: z={z0}:{z1}, y={y0}:{y1}, x={x0}:{x1}")

    print("Transformation complete.")

def scale_3d_volume_with_dask(input_array: da.Array, scale_factors: tuple, output_zarr_path: str, chunks: tuple = None):
    """
    Scales a 3D Dask array using scipy.ndimage.zoom and writes it to a Zarr file.

    Parameters:
    - input_array: dask.array.Array
        The input 3D Dask array (z, y, x).
    - scale_factors: tuple of 3 floats
        Scaling factors for (z, y, x) dimensions.
    - output_zarr_path: str
        Path to save the output Zarr dataset.
    - chunks: tuple, optional
        Chunk size for the output Dask array. Defaults to input_array.chunks.
    """
    if input_array.ndim != 3:
        raise ValueError("Input array must be 3-dimensional (z, y, x)")

    if chunks is None:
        chunks = input_array.chunks

    # Wrap zoom to be Dask-compatible via map_blocks
    def zoom_block(block, zoom_factors):
        return zoom(block, zoom_factors, order=1)

    zoomed = input_array.map_blocks(
        zoom_block,
        zoom_factors=scale_factors,
        dtype=input_array.dtype
    )

    # Rechunk to the desired output shape
    zoomed = zoomed.rechunk(chunks)

    # Save to Zarr
    da.to_zarr(zoomed, output_zarr_path, overwrite=True)
    print(f"Written scaled volume to: {output_zarr_path}")

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
                        'create_big_dask_volume': pipeline.create_big_dask_volume,
                        'create_matrix': pipeline.create_matrix,
                        'create_registration': pipeline.create_registration,
                        'split_volume': pipeline.split_big_volume,
                        'repack_volume': pipeline.repack_big_volume,
    }

    if task in function_mapping:
        function_mapping[task]()
    else:
        print(f'{task} is not a correct task. Choose one of these:')
        for key in function_mapping.keys():
            print(f'\t{key}')

