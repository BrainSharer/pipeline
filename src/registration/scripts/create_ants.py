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
import SimpleITK as sitk
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform
import pandas as pd
from scipy.io import loadmat

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
        self.inverse_transform_filepath = os.path.join(self.moving_path, f'{self.moving}_{self.fixed}_{self.z_um}x{self.xy_um}x{self.xy_um}um_{self.transformation}_inverse.mat')
        self.transform_filepath = os.path.join(self.moving_path, f'{self.moving}_{self.fixed}_{self.z_um}x{self.xy_um}x{self.xy_um}um_{self.transformation}.mat')

    def apply_ants_transform_zarr(self):
        """
        Apply ANTs transform to a large 3D Zarr volume in chunks.

        Parameters:
        - input_zarr_path: path to the input Zarr array (3D).
        - output_zarr_path: path to the output Zarr array.
        - transform_list: list of ANTs transform files (e.g., from ants.registration).
        - reference_image_path: path to reference image (defines output space).
        - chunk_size: tuple defining the size of chunks to process at a time.
        """
        self.check_registration()

        output_zarr_path = os.path.join(self.moving_path, f'{self.moving}_{self.fixed}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal_registered.zarr')
        if os.path.exists(output_zarr_path):
            print(f"Removing zarr file already exists at {output_zarr_path}")
            shutil.rmtree(output_zarr_path)
        # Open Zarr arrays
        #fixed_z = zarr.open(self.fixed_filepath_zarr, mode='r')
        #moving_z = zarr.open(self.moving_filepath_zarr, mode='r')

        moving_z, reference = pad_to_symmetric_shape(self.moving_filepath_zarr, self.fixed_filepath_zarr)

        print(type(reference))

        reference_ants = ants.from_numpy(np.array(reference))
        moving_ants = ants.from_numpy(np.array(moving_z))
        registration = ants.registration(fixed=reference_ants, moving=moving_ants, type_of_transform='Affine')

        shape = moving_z.shape
        dtype = moving_z.dtype

        dask_container = da.empty_like(moving_z, dtype=dtype, shape=shape)
        print(f'Creating Dask container with shape {shape} and dtype {dtype} {type(dask_container)=}')

        # Create output zarr with same shape and dtype
        #compressor = Blosc(cname='zstd', clevel=3)
        #output_zarr = zarr.open(output_zarr_path, mode='w')
        chunk_size = (moving_z.shape[0] // 5, moving_z.shape[1] // 1, moving_z.shape[2] // 2) # type: ignore
        print(f'Creating output Zarr at {output_zarr_path}\n with shape {shape} and chunk size {chunk_size} dtype {dtype}')
        #output_array = output_zarr.create('data', shape=shape, chunks=chunk_size, dtype=dtype, compressor=compressor)
        #reference = ants.image_read(self.fixed_filepath)

        chunk_count = 1
        # Iterate through chunks
        for z in range(0, shape[0], chunk_size[0]):
            for y in range(0, shape[1], chunk_size[1]):
                for x in range(0, shape[2], chunk_size[2]):
                    z0, y0, x0 = z, y, x
                    z1, y1, x1 = min(z+chunk_size[0], shape[0]), min(y+chunk_size[1], shape[1]), min(x+chunk_size[2], shape[2])
                    
                    # Load reference chunk
                    reference_chunk = reference[z0:z1, y0:y1, x0:x1]
                    reference_chunk_img = ants.from_numpy(reference_chunk.astype(np.float32), origin=(x0, y0, z0))
                    # Load moving chunk
                    moving_chunk = moving_z[z0:z1, y0:y1, x0:x1]
                    moving_chunk_img = ants.from_numpy(moving_chunk.astype(np.float32), origin=(x0, y0, z0),
                                                       spacing=reference_chunk_img.spacing, 
                                                       direction=reference_chunk_img.direction)

                    # Apply transform
                    #warped_chunk = ants.apply_transforms(fixed=reference, moving=chunk_img, transformlist=transform_list, interpolator='linear')
                    ids, counts = np.unique(moving_chunk_img.numpy(), return_counts=True)
                    print(f"Processing chunk {chunk_count}: z={z0}:{z1}, y={y0}:{y1}, x={x0}:{x1} - Unique IDs: {len(ids)}, Counts: {len(counts)}")
                    
                    
                    # Apply transform
                    warped_chunk = ants.apply_transforms(fixed=reference_chunk_img, moving=moving_chunk_img, 
                                                        transformlist=registration['fwdtransforms'], defaultvalue=0,
                                                        interpolator='linear')

                    # Write transformed chunk to output Zarr
                    transformed_array = warped_chunk.numpy()
                    #output_array[z0:z1, y0:y1, x0:x1] = transformed_array
                    dask_container[z0:z1, y0:y1, x0:x1] = transformed_array
                    chunk_count += 1

                    #zoomed = da.from_array(transformed_array, chunks=chunk_size)
                    #delayed = zoomed.to_zarr(outpath, compute=False, overwrite=True)
                    #with ProgressBar():
                    #    delayed.compute()

        dask_container.to_zarr(output_zarr_path, compute=True, overwrite=True)

        volume = zarr.open(output_zarr_path, 'r')
        print(volume.info)
        image_stack = []
        for i in tqdm(range(int(volume.shape[0]))): # type: ignore
            section = volume[i, ...]
            if section.ndim > 2: # type: ignore
                section = section.reshape(section.shape[-2], section.shape[-1]) # type: ignore
            image_stack.append(section)

        print('Stacking images ...')
        volume = np.stack(image_stack, axis=0)


        outpath = os.path.join(self.moving_path, 'registered.tif')
        if os.path.exists(outpath):
            print(f"Removing tiff file already exists at {outpath}")
            os.remove(outpath)
        write_image(outpath, volume.astype(dtype))





        print("Transformation complete.")


    def demons_registration(self, fixed_points=None, moving_points=None):

        if os.path.isfile(self.moving_filepath):
            print(f"Moving image found at {self.moving_filepath}")
        else:
            print(f"Moving image not found at {self.moving_filepath}")
            exit(1)
        if os.path.isfile(self.fixed_filepath):
            print(f"Fixed image found at {self.fixed_filepath}")
        else:
            print(f"Fixed image not found at {self.fixed_filepath}")
            exit(1)

        moving_image = sitk.ReadImage(self.moving_filepath, sitk.sitkFloat32)
        fixed_image = sitk.ReadImage(self.fixed_filepath, sitk.sitkFloat32)

        registration_method = sitk.ImageRegistrationMethod()

        # Create initial identity transformation.
        transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
        transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
        # The image returned from the initial_transform_filter is transferred to the transform and cleared out.
        initial_transform = sitk.DisplacementFieldTransform(
            transform_to_displacment_field_filter.Execute(sitk.Transform())
        )

        # Regularization (update field - viscous, total field - elastic).
        initial_transform.SetSmoothingGaussianOnUpdate(
            varianceForUpdateField=0.0, varianceForTotalField=2.0
        )

        registration_method.SetInitialTransform(initial_transform)

        registration_method.SetMetricAsDemons(
            10
        )  # intensities are equal if the difference is less than 10HU

        # Multi-resolution framework.
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8, 4, 0])

        registration_method.SetInterpolator(sitk.sitkLinear)
        # If you have time, run this code as is, otherwise switch to the gradient descent optimizer
        # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=20,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # If corresponding points in the fixed and moving image are given then we display the similarity metric
        # and the TRE during the registration.
        """
        if fixed_points and moving_points:
            registration_method.AddCommand(
                sitk.sitkStartEvent, rc.metric_and_reference_start_plot
            )
            registration_method.AddCommand(
                sitk.sitkEndEvent, rc.metric_and_reference_end_plot
            )
            registration_method.AddCommand(
                sitk.sitkIterationEvent,
                lambda: rc.metric_and_reference_plot_values(
                    registration_method, fixed_points, moving_points
                ),
            )
        """
        transform = registration_method.Execute(fixed_image, moving_image)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(transform)

        resampled = resampler.Execute(moving_image)


        outpath = os.path.join(self.moving_path, f'demons_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal_registered.tif')
        sitk.WriteImage(resampled, outpath)



    def zarr2tif(self):
        output_nii_path = os.path.join(self.moving_path, f'{self.moving}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal.nii.gz')
        input_zarr_path = os.path.join(self.moving_path, f'{self.moving}_{self.z_um}x{self.xy_um}x{self.xy_um}um_sagittal.zarr')
        if not os.path.isdir(input_zarr_path):
            print(f"Zarr dir not found at {input_zarr_path}")
            exit(1)
        volume = zarr.open(input_zarr_path, mode='r')
        image_stack = []
        for i in tqdm(range(int(volume.shape[0]))): # type: ignore
            section = volume[i, ...]
            if section.ndim > 2: # type: ignore
                section = section.reshape(section.shape[-2], section.shape[-1]) # type: ignore
            #img = sitk.GetImageFromArray(section)
            image_stack.append(section)

        print('Stacking images ...')
        volume = np.stack(image_stack, axis=0)
        # Create a NIfTI image
        print(f'Creating Nifti image from {volume.shape} dtype: {volume.dtype}')
        #sitk.WriteImage(volume, output_nii_path)
        # Create a NIfTI image
        affine = np.eye(4)  # Identity matrix as default affine
        nifti_img = nib.Nifti1Image(volume, affine) # type: ignore
        # Save to .nii file
        print(nib.aff2axcodes(nifti_img.affine)) # type: ignore
        desired_orientation = axcodes2ornt('SAR') # Example: to RAS orientation
        current_orientation = nib.orientations.io_orientation(nifti_img.affine)
        transform = ornt_transform(current_orientation, desired_orientation)
        reoriented_img = nifti_img.as_reoriented(transform)         # type: ignore
        print(nib.aff2axcodes(reoriented_img.affine)) # type: ignore
        nib.save(reoriented_img, output_nii_path) # type: ignore
        print(f'Wrote Nifti volume to {output_nii_path}')

        """
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

            
            write_image(fileoutpath, transformed.astype(np.uint16))
        """

    def apply_registration(self):
        self.check_registration()

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
        moving_z = moving_z.rechunk(tile_shape) # type: ignore
        fixed_z = fixed_z.rechunk(tile_shape) # type: ignore
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
            print(f"Fixed image found at {self.fixed_filepath}")
        else:
            print(f"Fixed image not found at {self.fixed_filepath}")
            exit(1)


        print(f'Reading moving image from {self.moving_filepath}')
        moving = ants.image_read(self.moving_filepath)
        print("Moving image loaded")
        print(f'Reading fixed image from {self.fixed_filepath}')
        fixed = ants.image_read(self.fixed_filepath)
        print("Fixed image loaded")

        if not os.path.isfile(self.transform_filepath):
            print("Starting registration ...")
            registration = ants.registration(fixed=fixed, moving=moving, type_of_transform=self.transformation)
            print("Registration completed")
            print(registration.keys())
            warped_moving = registration['warpedmovout']
            shutil.copy(registration['fwdtransforms'][0], self.transform_filepath)            

        else:
            print(f"Transform file already exists at {self.transform_filepath}")
            print("Applying registration ...")
            warped_moving = ants.apply_transforms( fixed=fixed, moving=moving, 
                                        transformlist=self.transform_filepath, defaultvalue=0)


        # Convert to numpy and save to disk
        warped_np = warped_moving.numpy()
        warped_np = np.swapaxes(warped_np, 0, 2)
        print(f'Warped image shape: {warped_np.shape} dtype: {warped_np.dtype}')
        output_tifs_path = os.path.join(self.fileLocationManager.prep, 'C1', f'{self.xy_um}_{self.z_um}_{self.transformation}')
        if os.path.exists(output_tifs_path):
            print(f"Removing tiff files already exists at {output_tifs_path}")
            shutil.rmtree(output_tifs_path)
        os.makedirs(output_tifs_path, exist_ok=True)
        print(f'Writing warped image slices to {output_tifs_path}')

        for i in tqdm(range(warped_np.shape[0])):
            slice_i = warped_np[i, ...]
            slice_i = slice_i.astype(np.uint16)
            outpath_slice = os.path.join(output_tifs_path, f'{str(i).zfill(4)}.tif')
            write_image(outpath_slice, slice_i)
            del slice_i

        outpath = os.path.join(self.moving_path, f'{self.moving}_{self.fixed}_{self.z_um}x{self.xy_um}x{self.xy_um}um_{self.transformation}.tif')
        with TiffWriter(outpath, bigtiff=True) as tif:
            for i in tqdm(range(warped_np.shape[0])):
                slice_i = warped_np[i, ...]
                slice_i = slice_i.astype(np.uint16)
                tif.write(slice_i.astype(np.uint16), contiguous=True)
        print(f'Wrote transformed volume to {outpath}')



    def register_points(self):
        if os.path.isfile(self.moving_filepath):
            print(f"Moving image found at {self.moving_filepath}")
        else:
            print(f"Moving image not found at {self.moving_filepath}")
            exit(1)
        if os.path.isfile(self.fixed_filepath):
            print(f"Fixed image found at {self.fixed_filepath}")
        else:
            print(f"Fixed image not found at {self.fixed_filepath}")
            exit(1)


        print(f'Reading moving image from {self.moving_filepath}')
        moving = ants.image_read(self.moving_filepath)
        print("Moving image loaded")
        print(f'Reading fixed image from {self.fixed_filepath}')
        fixed = ants.image_read(self.fixed_filepath)
        print("Fixed image loaded")

        if not os.path.isfile(self.transform_filepath):
            print("Starting registration ...")
            registration = ants.registration(fixed=moving, moving=fixed, type_of_transform=self.transformation)
            warped_moving = registration['warpedmovout']
            shutil.copy(registration['fwdtransforms'][0], self.inverse_transform_filepath)            
        else:
            print(f"Transform file already exists at {self.transform_filepath}")
            print("Applying registration to points")

        moving_points = np.array([
            [1119, 385, 517, 0],
            [111, 35, 57, 0],
        ]
        )

        print(f'moving points shape: {moving_points.shape} dtype: {moving_points.dtype}')
        """
        It's crucial to understand that the order of transformations for points is the opposite of that for 
        images. When transforming images, the inverse warps are applied, while for points, the forward warps 
        are used. This is because, for images, you're finding where each voxel in the output image comes 
        from in the input image, while for points, you're directly moving the points from the fixed 
        space to the moving space.
        """

        # Convert to DataFrame as required by ANTs
        pts = pd.DataFrame(moving_points, columns=['x', 'y', 'z', 't'])
        #pts = pd.DataFrame({'x': [10], 'y': [15], 'z': [12], 't': [0]})
        print("Moving points DataFrame:\n", pts.head())

        # Apply the  inverse transform (fixed -> moving space)
        transformed_points = ants.apply_transforms_to_points( 3, pts, self.inverse_transform_filepath)
        print("Transformed points in reference space:\n", transformed_points[['x', 'y', 'z', 't']].values)
        return


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
                        "demons" : pipeline.demons_registration,
                        "ants_range": pipeline.apply_ants_transform_zarr
    }

    if task in function_mapping:
        function_mapping[task]()
    else:
        print(f'{task} is not a correct task. Choose one of these:')
        for key in function_mapping.keys():
            print(f'\t{key}')
