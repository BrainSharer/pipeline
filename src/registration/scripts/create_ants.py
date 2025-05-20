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
from skimage.util import view_as_blocks
from scipy.ndimage import affine_transform
from itertools import product

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.image_manipulation.filelocation_manager import FileLocationManager
from library.utilities.utilities_process import read_image, write_image

def normalize16(img):
    if img.dtype == np.uint32:
        print('image dtype is 32bit')
        return img.astype(np.uint16)
    else:
        mn = img.min()
        mx = img.max()
        mx -= mn
        img = ((img - mn)/mx) * 2**16 - 1
        return np.round(img).astype(np.uint16) 


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
        self.fixed_filepath_zarr = os.path.join(self.fixed_path, f'{self.fixed}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal.zarr')
        self.moving_filepath_zarr = os.path.join(self.moving_path, f'{self.moving}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal.zarr')
        self.transform_filepath = os.path.join(self.moving_path, f'{self.moving}_{self.fixed}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal_to_Allen.mat')

    def zarr2tif(self):
        """
        output_tif_path = os.path.join(self.moving_path, f'{self.moving}_{self.fixed}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal_registered')
        output_zarr_path = os.path.join(self.moving_path, f'{self.moving}_{self.fixed}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal_registered.zarr')
        os.makedirs(output_tif_path, exist_ok=True)
        volume = zarr.open(output_zarr_path, mode='r')
        for i in tqdm(range(volume.shape[0])):
            outfile = os.path.join(output_tif_path, f'{str(i).zfill(4)}.tif')
            if os.path.exists(outfile):
                continue
            section = volume[i, ...]
            if section.ndim > 2:
                section = section.reshape(section.shape[-2], section.shape[-1])
            write_image(outfile, section)

        outpath = os.path.join(self.moving_path, 'registered.tif')
        if os.path.exists(outpath):
            print(f"Removing tiff file already exists at {outpath}")
            os.remove(outpath)

        with TiffWriter(outpath, bigtiff=True) as tif:
            for i in tqdm(range(volume.shape[0])):
                section = volume[i, ...]
                if section.ndim > 2:
                    section = section.reshape(section.shape[-2], section.shape[-1])
                
                tif.write(section.astype(np.uint16), contiguous=True)

        print(f'Wrote transformed volume to {outpath}')
        """
        matrix = np.array(
            [[0.9946726,   0.06427754 ],
            [ -0.04618119,  1.00449967]]            
        )
        #offset = (-47.81493201,  31.04401488)
        offset = (0,0)
        inpath = os.path.join(self.fileLocationManager.prep, 'C1', 'normalized_3')
        outpath = os.path.join(self.fileLocationManager.prep, 'C1', 'thumbnail_aligned')
        if os.path.exists(outpath):
            print(f"Removing tiff file already exists at {outpath}")
            shutil.rmtree(outpath)
        os.makedirs(outpath, exist_ok=True)
        files = sorted(os.listdir(inpath))
        for file in tqdm(files):
            fileinpath = os.path.join(inpath, file)
            fileoutpath = os.path.join(outpath, file)
            img = read_image(fileinpath)
            if img.ndim != 2:
                img = img.reshape((img.shape[-2], img.shape[-1]))

            transformed = affine_transform(
                img,
                matrix,
                offset=offset,
                order=0,
                mode='constant',
                cval=0
            )
                
            
            write_image(fileoutpath, transformed.astype(np.uint16))

    def apply_registration(self):
        self.check_registration(self.fixed_filepath_zarr)

        output_zarr_path = os.path.join(self.moving_path, f'{self.moving}_{self.fixed}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal_registered.zarr')
        if os.path.exists(output_zarr_path):
            print(f"Removing zarr file already exists at {output_zarr_path}")
            shutil.rmtree(output_zarr_path)
        # Open Zarr arrays
        #fixed_z = zarr.open(self.fixed_filepath_zarr, mode='r')
        #moving_z = zarr.open(self.moving_filepath_zarr, mode='r')

        moving_z, fixed_z = pad_to_symmetric_shape(self.moving_filepath_zarr, self.fixed_filepath_zarr)

        x = fixed_z.shape[2] // 8
        y = fixed_z.shape[1] // 1
        z = fixed_z.shape[0] // 1
        
        tile_shape = (z, y, x)  # Customize based on memory
        moving_z = moving_z.rechunk(tile_shape)
        fixed_z = fixed_z.rechunk(tile_shape)
        #output_z = zarr.open(output_zarr_path, mode='w', shape=fixed_z.shape, chunks=tile_shape, dtype='float32')
        print(f'tile_shape={tile_shape} {fixed_z.shape=} {moving_z.shape=}')

        # Downsample for global registration (in memory)
        print(f"Downsampling for global registration...")
        lowres_factor = 2
        fixed_lowres = ants.from_numpy(np.array(fixed_z[::lowres_factor, ::lowres_factor, ::lowres_factor]))
        moving_lowres = ants.from_numpy(np.array(moving_z[::lowres_factor, ::lowres_factor, ::lowres_factor]))

        # Perform global registration
        print(f"Performing global registration...")
        reg = ants.registration(fixed_lowres, moving_lowres, type_of_transform=self.transformation)
        transform = reg['fwdtransforms']

        # Apply transform block-wise
        def apply_transform_block(block, block_info=None):
            # block: numpy array of shape (z, y, x)
            img = ants.from_numpy(block)
            warped = ants.apply_transforms(fixed=img, moving=img, transformlist=transform, interpolator='linear', defaultvalue=0)
            return warped.numpy()

        # Apply the transform lazily over blocks
        print(f"Applying transform to blocks...")
        with ProgressBar():
            registered_dask = moving_z.map_blocks(apply_transform_block, dtype=moving_z.dtype)

        # Write to output Zarr
        print(f"Writing registered dataset to Zarr...")
        with ProgressBar():
            da.to_zarr(registered_dask, output_zarr_path, overwrite=True)
        print(f"Registered dataset written to {output_zarr_path}")


        """
        for start in tqdm(tile_generator(fixed_z.shape, tile_shape)):
            try:
                fixed_tile = load_zarr_tile(fixed_z, start, tile_shape)
                moving_tile = load_zarr_tile(moving_z, start, tile_shape)
                warped_tile = self.register_tile(fixed_tile, moving_tile)
                output_z[tuple(slice(s, s + sz) for s, sz in zip(start, tile_shape))] = warped_tile
            except Exception as e:
                print(f"Error processing tile at {start}: {e}")
        """



        outpath = os.path.join(self.moving_path, 'registered.tif')
        if os.path.exists(outpath):
            print(f"Removing tiff file already exists at {outpath}")
            os.remove(outpath)
        volume = zarr.open(output_zarr_path, mode='r')
        with TiffWriter(outpath, bigtiff=True) as tif:
            for i in tqdm(range(volume.shape[0])):
                section = volume[i, ...]
                if section.ndim > 2:
                    section = section.reshape(section.shape[-2], section.shape[-1])
                tif.write(section.astype(np.uint16), contiguous=True)
        print(f'Wrote transformed volume to {outpath}')


    def check_registration(self):

        print('\nChecking for registration files ...')
        if os.path.isfile(self.moving_filepath):
            print(f"Moving image found at {self.moving_filepath}")
        else:
            print(f"Moving image not found at {self.moving_filepath}")
        if os.path.isfile(self.fixed_filepath):
            print(f"Moving image found at {self.fixed_filepath}")
        else:
            print(f"Moving image not found at {self.fixed_filepath}")
        
        print('\nChecking for registration zarr dirs ...')
        if os.path.isdir(self.moving_filepath_zarr):
            print(f"Moving image dir found at {self.moving_filepath_zarr}")
        else:
            print(f"Moving image not found at {self.moving_filepath_zarr}")
        if os.path.isdir(self.fixed_filepath_zarr):
            print(f"Fixed image dir found at {self.fixed_filepath_zarr}")
        else:
            print(f"Fixed image dir not found at {self.fixed_filepath_zarr}")

        print('\nChecking for registration transformation files ...')
        if os.path.isfile(self.transform_filepath):
            print(f"Transformation found at {self.transform_filepath}")
        else:
            print(f"Transformation not found at {self.transform_filepath}")


    def create_registration_in_memory(self):
        if os.path.isfile(self.moving_filepath):
            print(f"Moving image found at {self.moving_filepath}")
        else:
            print(f"Moving image not found at {self.moving_filepath}")
            exit(1)
        if os.path.isfile(self.fixed_filepath):
            print(f"Moving image found at {self.fixed_filepath}")
        else:
            print(f"Moving image not found at {self.fixed_filepath}")
            exit(1)

        if os.path.isfile(self.transform_filepath):
            print(f"Removing {self.transform_filepath}")
            os.remove(self.transform_filepath)

        print(f'Reading moving image from {self.moving_filepath}')
        moving = ants.image_read(self.moving_filepath)
        print("Moving image loaded")
        print(f'Reading fixed image from {self.fixed_filepath}')
        fixed = ants.image_read(self.fixed_filepath)
        print("Fixed image loaded")

        print("Starting registration ...")
        registration = ants.registration(fixed=fixed, moving=moving, type_of_transform=self.transformation)

        print(registration)
        original_filepath = registration['fwdtransforms'][0]
        shutil.move(original_filepath, self.transform_filepath)
        print(f"Transform file moved to {self.transform_filepath}")
        output_tif_path = os.path.join(self.fileLocationManager.prep, 'C1', f'{self.xy_um}_{self.z_um}')
        if os.path.exists(output_tif_path):
            print(f"Removing tiff file already exists at {output_tif_path}")
            shutil.rmtree(output_tif_path)
        os.makedirs(output_tif_path, exist_ok=True)

        warped_moving = registration['warpedmovout']
        # Convert to numpy and save as Zarr
        warped_np = warped_moving.numpy()
        del warped_moving
        print(f'Warped image shape: {warped_np.shape} dtype: {warped_np.dtype}')
        for i in range(warped_np.shape[0]):
            slice_i = warped_np[i]
            slice_i = slice_i.astype(np.uint16)
            outpath_slice = os.path.join(output_tif_path, f'{str(i).zfill(4)}.tif')
            write_image(outpath_slice, slice_i)
            print(f'Wrote slice {i} to {outpath_slice}')
            del slice_i


    def create_registration(self):

        if os.path.isfile(self.transform_filepath):
            print(f"Removing {self.transform_filepath}")
            os.remove(self.transform_filepath)

        # Example usage
        fixed_zarr = self.fixed_filepath_zarr
        moving_zarr = self.moving_filepath_zarr

        registration = register_zarr_images(fixed_zarr, moving_zarr)
        print(registration)
        original_filepath = registration['fwdtransforms'][0]
        shutil.move(original_filepath, self.transform_filepath)
        print(f"Transform file moved to {self.transform_filepath}")
        output_tif_path = os.path.join(self.moving_path, f'{self.moving}_{self.fixed}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal_registered')
        output_tif_path = os.path.join(self.fileLocationManager.prep, 'C1', f'{self.xy_um}_{self.z_um}')
        if os.path.exists(output_tif_path):
            print(f"Removing tiff file already exists at {output_tif_path}")
            shutil.rmtree(output_tif_path)
        os.makedirs(output_tif_path, exist_ok=True)

        warped_moving = registration['warpedmovout']
        # Convert to numpy and save as Zarr
        warped_np = warped_moving.numpy()
        del warped_moving
        print(f'Warped image shape: {warped_np.shape} dtype: {warped_np.dtype}')
        for i in range(warped_np.shape[0]):
            slice_i = warped_np[i]
            slice_i = slice_i.astype(np.uint16)
            outpath_slice = os.path.join(output_tif_path, f'{str(i).zfill(4)}.tif')
            write_image(outpath_slice, slice_i)
            print(f'Wrote slice {i} to {outpath_slice}')
            del slice_i


        """
        moving_image = ants.image_read(self.moving_filepath)
        print(f"Moving image loaded and shape: {moving_image.shape} dtype: {moving_image.dtype}")
        reference = ants.image_read(self.fixed_filepath)
        print(f"Reference image loaded and shape: {reference.shape} dtype: {reference.dtype}")

        if os.path.isfile(self.transform_filepath):
            print(f"Transform file already exists at {self.transform_filepath}")
        else:
            print(f"Transform file not and now creating ...")
            tx = ants.registration(fixed=reference, moving=moving_image,  type_of_transform = (self.transformation))
            original_filepath = tx['fwdtransforms'][0]
            shutil.move(original_filepath, self.transform_filepath)
            print(f"Transform file moved to {self.transform_filepath}")

        mywarpedimage = ants.apply_transforms( fixed=reference, moving=moving_image, 
                                              transformlist=self.transform_filepath, defaultvalue=0)
        outpath = os.path.join(self.moving_path, f'{self.moving}_{self.fixed}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal.tif')
        ants.image_write(mywarpedimage, outpath)
        print(f'Wrote transformed volume to {outpath}')
        """



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
            print(f"File found at {inpath}")
        outpath = os.path.join(self.reg_path, self.moving, f'{self.moving}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal.tif')
        arr = read_image(inpath)
        change_z = 10/self.z_um
        change_y = 10/self.xy_um
        change_x = 10/self.xy_um
        scale_factors = np.array((change_z, change_y, change_x))
        print(f'change_z={change_z} change_y={change_y} change_x={change_x} {arr.shape=} {arr.dtype=}')
        zoomed = zoom(arr, scale_factors)
        # Write incrementally to TIFF
        with TiffWriter(outpath, bigtiff=True) as tif:
            for i in tqdm(range(zoomed.shape[0])):
                sliced = zoomed[i]
                tif.write(sliced.astype(arr.dtype), contiguous=True)

        print(f'Wrote zoomed volume to {outpath}')

    def resize_volume(self):
        inpath = os.path.join(self.reg_path, self.moving, f'{self.moving}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal.tif')
        if not os.path.isfile(inpath):
            print(f"File not found at {inpath}")
            exit(1)
        else:
            print(f"File found at {inpath}")
        arr = read_image(inpath)
        allen_shape = np.array((356, 347,632))
        arr_shape = np.array(arr.shape)
        change_z = allen_shape[0] / arr_shape[0]
        change_y = allen_shape[1] / arr_shape[1]
        change_x = allen_shape[2] / arr_shape[2]
        scale_factors = (change_z, change_y, change_x)
        print(f'change_z={change_z} change_y={change_y} change_x={change_x} {arr.shape=} {arr.dtype=}')
        print(f'Allen shape: {allen_shape} new shape {arr_shape * np.array(scale_factors)}')
        zoomed = zoom(arr, scale_factors)
        # Write incrementally to TIFF
        with TiffWriter(inpath, bigtiff=True) as tif:
            for i in tqdm(range(zoomed.shape[0])):
                sliced = zoomed[i]
                tif.write(sliced.astype(arr.dtype), contiguous=True)

        print(f'Wrote zoomed volume to {inpath}')

    def create_zarr(self):
        inpath = os.path.join(self.reg_path, self.moving, f'{self.moving}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal.tif')
        if not os.path.isfile(inpath):
            print(f"File not found at {inpath}")
            exit(1)
        else:
            print(f"File found at {inpath}.")
        arr = read_image(inpath)
        chunk_size = (64,64,64)
        outpath = os.path.join(self.reg_path, self.moving, f'{self.moving}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal.zarr')
        if os.path.exists(outpath):
            print(f"Zarr file already exists at {outpath}")
        else:
            zoomed = da.from_array(arr, chunks=chunk_size)
            delayed = zoomed.to_zarr(outpath, compute=False, overwrite=True)
            with ProgressBar():
                delayed.compute()

            print(f'zoomed.shape={zoomed.shape} {zoomed.dtype=}')

        
        print(f"Written scaled volume to: {outpath}")
        volume = zarr.open(outpath, 'r')
        print(volume.info)
        print(f'volume.shape={volume.shape}')
        return
        

        # Write incrementally to TIFF
        with TiffWriter(outpath, bigtiff=True) as tif:
            for i in tqdm(range(zoomed.shape[0])):
                sliced = zoomed[i]
                slice_i = sliced.compute()
                tif.write(slice_i.astype(arr.dtype), contiguous=True)

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
        filespath = os.path.join(self.fileLocationManager.prep, 'C1', 'normalized_1')
        if not os.path.isdir(filespath):
            print(f"Dir not found at {filespath}")
            exit(1)
        else:
            print(f"Dir found at {filespath}.")
        output_tif_path = os.path.join(self.reg_path, self.moving, f'{self.moving}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal.tif')
        if os.path.exists(output_tif_path):
            print(f"Removing tiff file already exists at {output_tif_path}")
            os.remove(output_tif_path)
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

    def register_tile(self, fixed_tile, moving_tile):
        if np.all(moving_tile == 0) or np.all(fixed_tile == 0):
            return moving_tile.astype(np.float32)  # skip empty tiles
        fixed_img = ants.from_numpy(fixed_tile.astype(np.float32))
        moving_img = ants.from_numpy(moving_tile.astype(np.float32))
        tx = ants.registration(fixed=fixed_img, moving=moving_img, type_of_transform=self.transformation)
        warped = ants.apply_transforms(fixed=fixed_img, moving=moving_img, transformlist=tx['fwdtransforms'])
        return warped.numpy()

def zoom_large_3d_array_to_tiff(input_array_path, shape, dtype, scale_factors, chunk_size=(64, 64, 64)):
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
    return zoomed

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
    reference = zarr.open(reference_image_path, mode='r')

    # Iterate through chunks
    for z in range(0, shape[0], chunk_size[0]):
        for y in range(0, shape[1], chunk_size[1]):
            for x in range(0, shape[2], chunk_size[2]):
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

def load_zarr_as_numpyXXX(zarr_path, chunk_size=(64, 64, 64)):
    """Load a Zarr dataset as a Dask array and convert it to NumPy."""
    print(f"Loading: {zarr_path}")
    z = zarr.open(zarr_path, mode='r')
    dask_array = da.from_zarr(z).rechunk(chunk_size)
    # Force compute into memory; for out-of-core registration, advanced ANTs customization is needed
    numpy_array = dask_array.compute()
    return numpy_array

def load_zarr_slice(zarr_slice, chunk_size=(64, 64, 64)):
    """Load a Zarr dataset as a Dask array and convert it to NumPy."""
    dask_array = da.from_zarr(zarr_slice).rechunk(chunk_size)
    # Force compute into memory; for out-of-core registration, advanced ANTs customization is needed
    numpy_array = dask_array.compute()
    return numpy_array

def register_zarr_images(fixed_path, moving_path, output_transform_prefix="output"):
    # Load datasets
    fixed_np = load_zarr_as_numpy(fixed_path)
    moving_np = load_zarr_as_numpy(moving_path)

    # Convert to ANTs images
    fixed = ants.from_numpy(fixed_np)
    moving = ants.from_numpy(moving_np)

    # Perform registration
    print("Starting registration...")
    registration = ants.registration(fixed=fixed, moving=moving, type_of_transform='Affine')

    # Apply transform
    #warped_moving = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=registration['fwdtransforms'], defaultvalue=0)

    print("Registration completed.")
    return registration

# Load Zarr images (assuming shape [z, y, x] or [x, y, z])
def load_zarr_as_numpy(zarr_path):
    store = zarr.open(zarr_path, mode='r')
    return np.array(store)

# Convert numpy to ANTs image
def numpy_to_ants_image(np_array, spacing=(1.0, 1.0, 1.0)):
    return ants.from_numpy(np_array, spacing=spacing)

# Apply affine transform
def apply_affine_transform(fixed_zarr_path, moving_zarr_path, output_path, spacing=(1.0, 1.0, 1.0)):
    fixed_np = load_zarr_as_numpy(fixed_zarr_path)
    moving_np = load_zarr_as_numpy(moving_zarr_path)

    fixed = numpy_to_ants_image(fixed_np, spacing=spacing)
    moving = numpy_to_ants_image(moving_np, spacing=spacing)

    # Compute affine registration
    tx = ants.registration(fixed=fixed, moving=moving, type_of_transform='Affine')

    # Apply transformation
    warped_moving = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=tx['fwdtransforms'])

    # Convert to numpy and save as Zarr
    warped_np = warped_moving.numpy()

    zarr.save(output_path, warped_np)
    print(f"Warped image saved to: {output_path}")


def register_tileXXX(fixed_tile, moving_tile):
    print(f"Registering tile with shape: {fixed_tile.shape} and {moving_tile.shape}")
    if np.all(moving_tile == 0) or np.all(fixed_tile == 0):
        return moving_tile  # skip empty tiles
    
    if fixed_tile.shape != moving_tile.shape:
        raise ValueError("Fixed and moving tiles must have the same shape.")

    fixed_img = ants.from_numpy(fixed_tile)
    moving_img = ants.from_numpy(moving_tile)

    tx = ants.registration(fixed_img, moving_img, type_of_transform="Affine")
    warped = ants.apply_transforms(
        fixed_img, moving_img, transformlist=tx["fwdtransforms"]
    )

    return warped.numpy()

def load_zarr_tile(z, start, size):
    slices = tuple(slice(s, s + sz) for s, sz in zip(start, size))
    return z[slices]


def tile_generator(shape, tile_size):
    """Yield tile start indices based on dataset shape and tile size."""
    steps = [range(0, dim, tile) for dim, tile in zip(shape, tile_size)]
    return product(*steps)


def pad_to_symmetric_shape(moving_path, fixed_path):
    # Load the Zarr arrays
    moving_arr = da.from_zarr(moving_path)
    fixed_arr = da.from_zarr(fixed_path)

    # Determine target shape
    shape1 = moving_arr.shape
    shape2 = fixed_arr.shape
    max_shape = tuple(max(s1, s2) for s1, s2 in zip(shape1, shape2))

    # Compute padding needed for each array
    def compute_padding(shape, target):
        return [(0, t - s) for s, t in zip(shape, target)]

    pad1 = compute_padding(shape1, max_shape)
    pad2 = compute_padding(shape2, max_shape)

    # Pad with zeros
    moving_arr_padded = da.pad(moving_arr, pad_width=pad1, mode='constant', constant_values=0)
    fixed_arr_padded = da.pad(fixed_arr, pad_width=pad2, mode='constant', constant_values=0)
    return moving_arr_padded, fixed_arr_padded

def register_zarr_datasets(fixed_zarr_path, moving_zarr_path, output_zarr_path, lowres_factor=4):
    # Load the Zarr arrays as Dask arrays
    fixed_zarr = zarr.open(fixed_zarr_path, mode='r')
    moving_zarr = zarr.open(moving_zarr_path, mode='r')
    
    fixed_dask = da.from_zarr(fixed_zarr)
    moving_dask = da.from_zarr(moving_zarr)

    # Downsample for global registration (in memory)
    fixed_lowres = ants.from_numpy(np.array(fixed_dask[::lowres_factor, ::lowres_factor, ::lowres_factor]))
    moving_lowres = ants.from_numpy(np.array(moving_dask[::lowres_factor, ::lowres_factor, ::lowres_factor]))

    # Perform global registration
    reg = ants.registration(fixed_lowres, moving_lowres, type_of_transform='SyN')
    transform = reg['fwdtransforms']

    # Apply transform block-wise
    def apply_transform_block(block, block_info=None):
        # block: numpy array of shape (z, y, x)
        img = ants.from_numpy(block)
        warped = ants.apply_transforms(fixed=img, moving=img, transformlist=transform, interpolator='linear')
        return warped.numpy()

    # Apply the transform lazily over blocks
    registered_dask = moving_dask.map_blocks(apply_transform_block, dtype=moving_dask.dtype)

    # Write to output Zarr
    da.to_zarr(registered_dask, output_zarr_path, overwrite=True)
    print(f"Registered dataset written to {output_zarr_path}")

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
                        'create_zarr': pipeline.create_zarr,
                        'create_matrix': pipeline.create_matrix,
                        'create_registration': pipeline.create_registration_in_memory,
                        'split_volume': pipeline.split_big_volume,
                        'repack_volume': pipeline.repack_big_volume,
                        'apply_registration': pipeline.apply_registration,
                        'resize_volume': pipeline.resize_volume,
                        'zarr2tif': pipeline.zarr2tif,
                        'status': pipeline.check_registration,
    }

    if task in function_mapping:
        function_mapping[task]()
    else:
        print(f'{task} is not a correct task. Choose one of these:')
        for key in function_mapping.keys():
            print(f'\t{key}')
