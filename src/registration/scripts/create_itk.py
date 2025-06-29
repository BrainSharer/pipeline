import shutil
import dask.array as da
from dask.diagnostics import ProgressBar
from dask import delayed
from dask.array import map_overlap
import zarr
import numpy as np
import os
import sys
import SimpleITK as sitk
from tqdm import tqdm
from tifffile import TiffWriter
import ants
# Directory paths
from pathlib import Path
from scipy.ndimage import affine_transform


PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.image_manipulation.filelocation_manager import FileLocationManager
from library.utilities.utilities_process import read_image, write_image


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

# Convert Dask array chunk to SimpleITK image
def chunk_to_sitk(chunk, spacing=(1.0, 1.0, 1.0)):
    img = sitk.GetImageFromArray(chunk.astype(np.float32))
    img.SetSpacing(spacing)
    return img

# Apply affine transform to Dask array in chunks
def apply_affine_to_dask(source_dask, transform, reference_image, spacing=(1.0, 1.0, 1.0)):
    def apply_transform_block(block, block_info=None):
        src_img = chunk_to_sitk(block, spacing)
        resampled = sitk.Resample(src_img, reference_image, transform,
                                  sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        return sitk.GetArrayFromImage(resampled)
    
    # Apply transform blockwise
    transformed = source_dask.map_blocks(apply_transform_block, dtype=np.float32)
    return transformed

def command_iteration(method):
    """ Callback invoked when the optimization process is performing an iteration. """
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f} "
        + f": {method.GetOptimizerPosition()}"
    )


def estimate_affine_transform(fixed_dask, moving_dask, spacing=(1.0, 1.0, 1.0), downsample_factor=2):
    fixed_np = fixed_dask[::downsample_factor, ::downsample_factor, ::downsample_factor].compute()
    moving_np = moving_dask[::downsample_factor, ::downsample_factor, ::downsample_factor].compute()

    fixed_img = chunk_to_sitk(fixed_np, spacing)
    moving_img = chunk_to_sitk(moving_np, spacing)

    # Set up registration
    #registration_method = sitk.ImageRegistrationMethod()
    #registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    #registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=2500, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    #registration_method.SetInitialTransform(sitk.AffineTransform(fixed_img.GetDimension()))
    #registration_method.SetInterpolator(sitk.sitkLinear)
    #transform = registration_method.Execute(fixed_img, moving_img)
    final_transform1 = sitk.AffineTransform(3)
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsPowell(stepLength=0.1, numberOfIterations=25)
    #registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(final_transform1)
    #registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [16, 8, 4])
    #registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [4, 2, 0])
    #registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.Execute(fixed_img, moving_img)
    print(final_transform1)
    return final_transform1


    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, 0.01, 200)
    R.SetInitialTransform(sitk.TranslationTransform(fixed_img.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    transform = R.Execute(fixed_img, moving_img)
    print("-------")
    print(transform)
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {R.GetOptimizerIteration()}")
    print(f" Metric value: {R.GetMetricValue()}")
    transform.Get
    # Ensure it's a 3D transform
    if transform.GetDimension() != 3:
        raise ValueError("This function only supports 3D transforms.")
    print(type(transform))


    return transform

def save_chunk_to_zarr(arr, path, shape, chunks, dtype):
    store = zarr.DirectoryStore(path)
    root = zarr.group(store=store)
    z = root.empty('data', shape=shape, chunks=chunks, dtype=dtype)
    return z, store

def apply_affine_to_block(block, affine_matrix, origin):
    """
    Apply the affine transform to a block using SimpleITK.
    """
    # Convert numpy block to SimpleITK image
    image = sitk.GetImageFromArray(block.astype(np.float32))
    image.SetOrigin(origin)
    image.SetSpacing((1.0, 1.0, 1.0))  # Adjust if needed

    # Set up affine transform
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(affine_matrix[:3, :3].flatten())
    transform.SetTranslation(affine_matrix[:3, 3])
    # Resample using the affine transform
    resampled = sitk.Resample(
        image,
        image.GetSize(),
        transform,
        sitk.sitkLinear,
        image.GetOrigin(),
        image.GetSpacing(),
        0.0,
        image.GetPixelID(),
    )
    return sitk.GetArrayFromImage(resampled)


def process_block(block):
    """
    Apply affine transformation to a block.
    """
    # Compute the location of this block in the global array
    #slicing = tuple(slice(start, stop) for start, stop in zip(block_info[None]['array-location'][0], block_info[None]['array-location'][1]))
    # Apply the affine transformation
    transformed = affine_transform(
        block,
        matrix=affine_matrix,
        offset=offset,
        order=1,
        mode='nearest',
        prefilter=False
    )
    return transformed


def register_large_zarr_datasets():
    # Helper: apply transform to a block
    reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'

    moving_image_path = os.path.join(reg_path, 'ALLEN771602', 'ALLEN771602_32.0x28.8x28.8um_sagittal.zarr')
    fixed_image_path = os.path.join(reg_path, 'Allen', 'Allen_32.0x28.8x28.8um_sagittal.zarr')
    outpath = os.path.join(reg_path, 'ALLEN771602', 'registered_32.0x28.8x28.8um_sagittal.zarr')

    if os.path.exists(outpath):
        print(f"Output file {outpath} already exists. removing")
        shutil.rmtree(outpath)

    if os.path.exists(moving_image_path):
        print(f"Moving image {moving_image_path}")
    else:
        print(f"Moving image {moving_image_path} does not exist. Exiting.")
        sys.exit()
    
    if os.path.exists(fixed_image_path):
        print(f"Fixed image {fixed_image_path}")
    else:
        print(f"Fixed image {fixed_image_path} does not exist. Exiting.")
        sys.exit()

    # Pad to symmetric shape
    moving_dask, fixed_dask = pad_to_symmetric_shape(moving_image_path, fixed_image_path)
    assert moving_dask.shape == fixed_dask.shape, "Source and target must have the same shape"
    #chunks = (fixed_dask.shape[0]//8, fixed_dask.shape[1]//8, fixed_dask.shape[2]//8) 
    #chunks = CHUNK_SIZE
    chunk_size = (moving_dask.shape[0] // 5, moving_dask.shape[1] // 1, moving_dask.shape[2] // 1)
    OVERLAP = (8, 8, 8)  # overlap between chunks to avoid seams
    moving_dask = moving_dask.rechunk(chunk_size) 
    fixed_dask = fixed_dask.rechunk(chunk_size)

    print(f"Moving image shape: {moving_dask.shape} chunks: {moving_dask.chunksize}")
    print(f"Fixed image shape: {fixed_dask.shape} chunks: {fixed_dask.chunksize}")
    
    # Define affine transform using SimpleITK
    # Pad source array to handle overlap
    padded_source = da.overlap.overlap(moving_dask, depth=OVERLAP, boundary='nearest')

    #process_block(block, affine_matrix, offset, block_info=None
    transformed = padded_source.map_overlap(
        process_block,
        depth=OVERLAP,
        dtype=moving_dask.dtype,
        boundary='reflect'
    )
    """

    print("Estimating affine transform...")
    #transform = estimate_affine_transform(fixed_dask, moving_dask)

    def transform_wrapper(block, block_info=None):
            block_shape = block.shape
            loc = block_info[None]['array-location']
            origin = [loc[i][0] for i in range(3)]

            return apply_ants_to_block(block, four_by_four, origin)

    print(f"Applying affine transform to blocks with shape {padded_source.shape} and chunks {padded_source.chunksize}...")
    # Map blocks with overlap and transform
    transformed = padded_source.map_overlap(
        transform_wrapper,
        depth=OVERLAP,
        boundary='reflect',
        dtype=np.float32
    )
    """

    print(f'Transformed shape: {transformed.shape} chunks: {transformed.chunksize}')
    # Trim the overlap to remove padded areas
    trimmed = da.overlap.trim_overlap(transformed, depth=OVERLAP, boundary='nearest')   
    trimmed = trimmed.rechunk(chunk_size)  # Rechunk to desired size



    # Store result to Zarr
    if os.path.exists(outpath):
        print(f"Output file {outpath} already exists. removing")
        shutil.rmtree(outpath)
    print(f"Writing registered image to {outpath}")
    print(f'trimmed {type(trimmed)=} trimmed shape: {trimmed.shape} chunks: {trimmed.chunksize}')
    # Save output to new Zarr store
    with ProgressBar():
        trimmed.to_zarr(outpath, overwrite=True)
        
    volume = zarr.open(outpath, mode='r')
    print(f'volume.info: {volume.info}')
    output_tif_path = os.path.join(reg_path, 'ALLEN771602', 'registered_slices')
    if os.path.exists(output_tif_path):
        print(f"Output TIF {output_tif_path} already exists. removing")
        shutil.rmtree(output_tif_path)
    os.makedirs(output_tif_path, exist_ok=True)
    print(f"Writing registered TIF image to {output_tif_path}")
    end = volume.shape[0]
    for i in tqdm(range(end)): # type: ignore
        section = volume[i, ...]
        if section.ndim > 2: # type: ignore
            section = section.reshape(section.shape[-2], section.shape[-1]) # type: ignore
        fileoutpath = os.path.join(output_tif_path, f'{i:04d}.tif')
        write_image(fileoutpath, section.astype(np.uint8))

    print(f"Registered image written to {output_tif_path}")



def register_large_zarr(
    input_zarr_path,
    output_zarr_path,
    affine_matrix,
    chunk_size=(128, 128, 128),
    overlap=(16, 16, 16)):

    # Open input dataset as Dask array
    input_z = zarr.open(input_zarr_path, mode='r')
    input_da = da.from_zarr(input_z)

    # Pad the array with reflect to avoid edge artifacts
    padded = da.pad(input_da, pad_width=[(o, o) for o in overlap], mode="reflect")

    def transform_wrapper(block, block_info=None):
        block_shape = block.shape
        loc = block_info[None]['array-location']
        origin = [loc[i][0] for i in range(3)]

        return apply_affine_to_block(block, affine_matrix, origin)

    # Map blocks with overlap and transform
    transformed = padded.map_overlap(
        transform_wrapper,
        depth=overlap,
        boundary='reflect',
        dtype=np.float32
    )

    # Store result to new Zarr file
    transformed.to_zarr(output_zarr_path, overwrite=True, compute=False)

    with ProgressBar():
        transformed.compute()



def scale_affine_transform(transform_path, scale_factor, output_transform_path):
    # Load the transform
    transform = ants.read_transform(transform_path)
    
    # Ensure it's an affine transform
    if not isinstance(transform, ants.ANTsTransform):
        raise TypeError("Loaded transform is not an ANTsTransform.")

    if 'AffineTransform' not in transform.transform_type:
        raise ValueError("Transform is not an affine transform.")

    # Extract the affine matrix and translation
    matrix = transform.parameters[:transform.dimension**2]
    translation = transform.parameters[transform.dimension**2:]
    print(f"Original translation: {translation}")
    #Extract the fixed stuff
    fixed = transform.fixed_parameters
    print(f"Fixed parameters: {fixed}")
    # Scale the fixed
    scaled_fixed = [t * scale_factor for t in fixed]
    print(f"Scaled fixed: {scaled_fixed}")

    # Scale the translation
    scaled_translation = [t * scale_factor for t in translation]
    print(f"Scaled translation: {scaled_translation}")

    s = 0.25  # downscaling factor
    S = np.diag([s, s, s, 1])        # scale full -> small
    S_inv = np.diag([1/s, 1/s, 1/s, 1])  # scale small -> full

    A_full = S_inv @ matrix @ S    

    # Set new parameters
    new_params = list(matrix) + scaled_translation
    transform.set_parameters(new_params)
    transform.set_fixed_parameters(scaled_fixed)
    

    # Save the new transform
    ants.write_transform(transform, output_transform_path)

def convert_tif_to_nii():
    reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
    moving_tif_path = os.path.join(reg_path, 'ALLEN771602', 'ALLEN771602_16.0x14.4x14.4um_sagittal.tif')
    fixed_tif_path = os.path.join(reg_path, 'Allen', 'Allen_16.0x14.4x14.4um_sagittal.tif')
    moving_nii_path = os.path.join(reg_path, 'ALLEN771602', 'ALLEN771602_16.0x14.4x14.4um_sagittal.nii')
    fixed_nii_path = os.path.join(reg_path, 'Allen', 'Allen_16.0x14.4x14.4um_sagittal.nii')
    transform_path = os.path.join(reg_path, 'ALLEN771602', 'ALLEN771602_Allen_8x7.2x7.2um_sagittal_to_Allen.mat')

    if os.path.exists(moving_tif_path) and not os.path.exists(moving_nii_path):
        print(f"Output NII {moving_tif_path} already exists")
        moving_tif = sitk.ReadImage(moving_tif_path)
        print ('direction of nii', moving_tif.GetDirection())
        sitk.WriteImage(moving_tif, moving_nii_path)
        print(f"Converted {moving_tif_path} to {moving_nii_path}")
        del moving_tif
    else:
        print(f"Output NII {moving_tif_path} does not exist")
 

    if os.path.exists(fixed_tif_path) and not os.path.exists(fixed_nii_path):
        print(f"Output NII {fixed_tif_path} already exists")
        fixed_tif = sitk.ReadImage(fixed_tif_path)
        print ('direction of nii', fixed_tif.GetDirection())
        sitk.WriteImage(fixed_tif, fixed_nii_path)
        print(f"Converted {fixed_tif_path} to {fixed_nii_path}")
    else:
        print(f"Output NII {fixed_tif_path} does not exist")

    if os.path.exists(transform_path):
        print(f"Output transform {transform_path} already exists")
        scale_affine_transform(transform_path, 4, os.path.join(reg_path, 'ALLEN771602', 'scaled_transform.mat'))
        print(f"Converted {transform_path} to scaled_transform.mat")
    else:
        print(f"Output transform {transform_path} does not exist")


def rescale_transform():
    reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'

    small_fixed_path = os.path.join(reg_path, 'Allen', 'Allen_32.0x28.8x28.8um_sagittal.tif')
    large_fixed_path = os.path.join(reg_path, 'Allen', 'Allen_32.0x28.8x28.8um_sagittal.tif')
    large_moving_path = os.path.join(reg_path, 'ALLEN771602', 'ALLEN771602_32.0x28.8x28.8um_sagittal.tif')
    small_fixed = sitk.ReadImage(small_fixed_path)
    large_fixed = sitk.ReadImage(large_fixed_path)
    large_moving = sitk.ReadImage(large_moving_path)

    transform_path = os.path.join(reg_path, 'ALLEN771602', 'ALLEN771602_Allen_32.0x28.8x28.8um_sagittal.tfm')
    numpy_path = os.path.join(reg_path, 'ALLEN771602', 'ALLEN771602_Allen_32.0x28.8x28.8um_sagittal.npy')
    out_path = os.path.join(reg_path, 'ALLEN771602', 'rescaled.tif')

    # Load or get the transform from small volume registration
    # Example: affine_transform = sitk.AffineTransform(3)
    # You may have gotten this from registration: registration_method.GetTransform()
    print(f"Loading transform from {transform_path}")
    #affine_matrix = np.load(transform_path)
    transform = sitk.ReadTransform(transform_path)
    fourxfour = np.load(numpy_path)
    transform = sitk.AffineTransform(transform)
    print("Loading saved elastix affine transform to 4x4 matrix...")

    # Load or define your large image (fixed or moving)
    print(f"Loading large image from {large_moving_path}")

    scaled_transform = create_large_affine(large_fixed, small_fixed, transform, fourxfour)
    resultImage = apply_transform_to_large_image(large_moving, large_fixed, scaled_transform, )
    # Save or use the resampled volume
    print(f'Saving resampled image to {out_path}')
    #resultImage = sitk.Cast(sitk.RescaleIntensity(resultImage), sitk.sitkUInt16)
    sitk.WriteImage(resultImage, out_path)



def create_large_affine(large_fixed, small_fixed, transform_small, fourxfour):
    """
    Scale an affine transform estimated on a small image to a larger image.

    Args:
        transform_small: sitk.AffineTransform estimated on small volumes.
        small_fixed: sitk.Image used as fixed image in the registration.
        large_fixed: sitk.Image that corresponds to the larger version.

    Returns:
        Scaled sitk.AffineTransform for the large image.
    """

    # Compute physical center of both volumes
    def get_physical_center(image):
        size = np.array(image.GetSize())
        spacing = np.array(image.GetSpacing())
        origin = np.array(image.GetOrigin())
        direction = np.array(image.GetDirection()).reshape(3, 3)
        center_index = size / 2.0
        center_physical = origin + direction @ (spacing * center_index)
        return center_physical



    small_center = get_physical_center(small_fixed)
    x,y,z = small_center
    small_center = [z,y,x]
    large_center = get_physical_center(large_fixed)
    x,y,z = large_center
    large_center = [z,y,x]
    print(f'small center={small_center}')
    print(f'large center={large_center}')

    # Extract matrix and translation
    matrix = np.array(transform_small.GetMatrix()).reshape(3, 3)
    translation = np.array(transform_small.GetTranslation())
    #translation = [ 54.7211 -30.3489   8.2287]
    #translation = np.array([8.22, -30.34, 54.7])

    # Transform about the center of the small image
    centered_translation = (
        small_center
        - matrix @ small_center
        + translation
    )

    print('Original rotation from transform')
    print(matrix)
    print('Original translation from transform')
    print(centered_translation)
    print('Numpy 4x4')
    print(fourxfour)


    # Construct scaled transform
    transform_large = sitk.AffineTransform(3)
    transform_large.SetMatrix(matrix.ravel())
    new_translation = large_center - matrix @ large_center + centered_translation
    transform_large.SetTranslation(new_translation)


    print('Scaled translation')
    print(new_translation)
    del large_fixed

    return transform_large

def apply_transform_to_large_image(large_moving, reference_image, transform):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    return resampler.Execute(large_moving)

# Convert affine to sitk transform
def sitk_transform_from_numpy(affine):
    transform = sitk.AffineTransform(3)
    matrix = affine[:3, :3].flatten()
    translation = affine[:3, 3]
    transform.SetMatrix(matrix)
    transform.SetTranslation(translation)
    return transform

# Convert Dask array to SimpleITK image metadata (without loading full array)
def make_sitk_image_metadata(shape, spacing, dtype):
    img = sitk.Image(shape[::-1], sitk.sitkFloat32 if np.issubdtype(dtype, np.floating) else sitk.sitkUInt16)
    img.SetSpacing(spacing[::-1])
    return img

def apply_affine_transform_to_large_zarr(
    input_zarr_path,
    output_zarr_path,
    affine_matrix):
    """
    Apply an affine transformation to a very large 3D zarr volume using Dask and SimpleITK.

    Parameters:
        input_zarr_path (str): Path to the input Zarr volume.
        output_zarr_path (str): Path to write the transformed Zarr volume.
        affine_matrix (np.ndarray): 4x4 affine transformation matrix.
        chunk_size (tuple): Size of chunks for Dask.
        voxel_spacing (tuple): Spacing in physical units for each voxel.
    """

    # Load input volume using Dask + Zarr
    z = zarr.open(input_zarr_path, mode='r')
    volume_dask = da.from_zarr(z)
    shape = volume_dask.shape
    dtype = volume_dask.dtype
    chunk_size = (volume_dask.shape[0]//5, volume_dask.shape[1]//1, volume_dask.shape[2]//1)  # Adjust chunk size as needed
    volume_dask = volume_dask.rechunk(chunk_size)
    print(f"Input volume shape: {shape}, dtype: {dtype}, chunks: {volume_dask.chunksize}")


    sitk_tx = sitk_transform_from_numpy(affine_matrix)

    # Read whole array lazily and compute transformed image in chunks
    def transform_block(block, block_info=None):
        # Get the location of the block in global coordinates
        z0, y0, x0 = [s[0] for s in block_info[None]['array-location']]
        z1, y1, x1 = [s[1] for s in block_info[None]['array-location']]

        # Extract subvolume as SimpleITK image
        subimage = sitk.GetImageFromArray(block.astype(np.float32))
        subimage.SetSpacing(1.0, 1.0, 1.0)  # Adjust spacing if needed
        subimage.SetOrigin((z0, y0, x0))
        # Apply transformation
        #resampled = sitk.Resample(subimage, reference_image[z0:z1, y0:y1, x0:x1], sitk_tx,
        #                          sitk.sitkLinear, 0.0, subimage.GetPixelID())
        resampled = sitk.Resample(subimage, subimage, sitk_tx,
                                  sitk.sitkLinear, 0.0, subimage.GetPixelID())
        
        print(f"resampled at z: {z0}-{z1}, y: {y0}-{y1}, x: {x0}-{x1}, size: {resampled.GetSize()} origin: {resampled.GetOrigin()} spacing: {resampled.GetSpacing()}")
        result = sitk.GetArrayFromImage(resampled).astype(dtype)
        if result.shape != block.shape:
            print(f"\tTransformed block shape {result.shape} does not match original block shape {block.shape}\n")
            raise ValueError(
                f"Transformed block shape {result.shape} does not match original block shape {block.shape}"
            )

        return result

    # Apply transformation lazily across volume
    transformed_dask = volume_dask.map_blocks(transform_block, dtype=dtype)
    print(f"Transformed volume shape: {transformed_dask.shape}, dtype: {transformed_dask.dtype}, chunks: {transformed_dask.chunksize}")
    # Save to Zarr
    #with ProgressBar():
    transformed_dask.to_zarr(output_zarr_path, overwrite=True)

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


    outpath = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/ALLEN771602/registered.tif'
    if os.path.exists(outpath):
        print(f"Removing tif file at {outpath}")
        os.remove(outpath)
    write_image(outpath, volume.astype(np.uint8))


def process_block_with_ants(reference_chunk, moving_chunk, transform_filepath):
    moving_chunk_img = ants.from_numpy(moving_chunk.astype(np.float32))
    reference_chunk_img = ants.from_numpy(reference_chunk.astype(np.float32))
    transformed_block = ants.apply_transforms(fixed=reference_chunk_img, moving=moving_chunk_img, transformlist=transform_filepath, 
                                                interpolator='linear', defaultvalue=0)
    
    return transformed_block.numpy()


def apply_affine_transform_zarr(
    zarr_input_path: str,
    zarr_output_path: str,
    affine_matrix: np.ndarray):
    """
    Apply an affine transformation to a large 3D volume stored in Zarr using Dask and ANTs.

    Parameters:
        zarr_input_path (str): Path to the input Zarr volume.
        zarr_output_path (str): Path to save the output Zarr volume.
        affine_matrix (np.ndarray): 4x4 affine matrix.
        chunk_size (tuple): Chunk size for Dask array.
        overlap (int): Number of voxels to overlap between chunks.
    """
    
    # Load the input Zarr store as a Dask array
    zarr_input = zarr.open(zarr_input_path, mode='r')
    dask_volume = da.from_zarr(zarr_input)
    chunk_size = (dask_volume.shape[0] // 5, dask_volume.shape[1] // 1, dask_volume.shape[2] // 1)  # Adjust chunk size as needed
    dask_volume = dask_volume.rechunk(chunk_size)
    print(f"Input volume shape: {dask_volume.shape}, dtype: {dask_volume.dtype}, chunks: {dask_volume.chunksize}")

    # Image spacing (assumed isotropic 1.0, change as needed)
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    direction = np.eye(3)
    dtype = dask_volume.dtype
    overlap = 8

    

    def wrapper(block, block_info=None):
        return process_block_with_ants(block, block_info, reference_chunk, transform_filepath)

    # Apply affine with overlap
    transformed = map_overlap(
        wrapper,
        dask_volume,
        depth=overlap,
        boundary='reflect',
        dtype=dtype
    )

    # Save the result to a new Zarr store
    transformed.to_zarr(zarr_output_path, overwrite=True, compute=False)

    with ProgressBar():
        transformed.compute()



def elastix_to_affine_matrix(filepath):
    def parse_line(line):
        if line.startswith('(TransformParameters'):
            line = line.replace('(TransformParameters ', '').replace(')', '')
            parameters = list(map(float, line.split()))
        if line.startswith('(CenterOfRotationPoint'):
            line = line.replace('(CenterOfRotationPoint ', '').replace(')', '')
            parameters = list(map(float, line.split()))
        return parameters

    with open(filepath, 'r') as f:
        lines = f.readlines()

    transform_params = None
    center_of_rotation = None

    for line in lines:
        if line.startswith('(TransformParameters'):
            values = parse_line(line)
            transform_params = list(map(float, values))
        elif line.startswith('(CenterOfRotationPoint'):
            values = parse_line(line)
            center_of_rotation = np.array(list(map(float, values)))

    if transform_params is None or center_of_rotation is None:
        raise ValueError("Required parameters not found in the file.")

    matrix_3x3 = np.array(transform_params[:9]).reshape((3, 3))
    translation = np.array(transform_params[9:])

    # Apply: x' = M @ (x - c) + c + t = Mx + (t - M @ c + c)
    adjusted_translation = translation + center_of_rotation - matrix_3x3 @ center_of_rotation

    # Construct the 4x4 matrix
    affine_4x4 = np.eye(4)
    affine_4x4[:3, :3] = matrix_3x3
    affine_4x4[:3, 3] = adjusted_translation

    return affine_4x4

def elastix_to_affine_4x4(filepath):
    """
    Converts an Elastix TransformParameters.0.txt affine file into a 4x4 numpy matrix in ZYX order.
    """
    import re
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract TransformParameters
    params_match = re.search(r'\(TransformParameters\s+(.*?)\)', content, re.DOTALL)
    if not params_match:
        raise ValueError("TransformParameters not found in the file.")
    params = list(map(float, params_match.group(1).split()))

    if len(params) != 12:
        raise ValueError(f"Expected 12 TransformParameters, got {len(params)}")

    # Extract CenterOfRotationPoint
    center_match = re.search(r'\(CenterOfRotationPoint\s+(.*?)\)', content)
    if not center_match:
        raise ValueError("CenterOfRotationPoint not found in the file.")
    center = np.array(list(map(float, center_match.group(1).split())))

    # Build the affine matrix A and translation vector t
    A = np.array(params[:9]).reshape(3, 3)
    t = np.array(params[9:])

    # Apply the formula: M = A·(x - c) + t + c  => Full affine:
    # M = A x + (t - A·c + c) = A x + offset
    offset = t - A @ center + center

    # Compose into 4x4 affine matrix
    affine = np.eye(4)
    affine[:3, :3] = A
    affine[:3, 3] = offset

    # Reorder from XYZ to ZYX by permuting rows and columns
    reorder = [2, 1, 0]  # Z, Y, X
    affine_zyx = np.eye(4)
    affine_zyx[:3, :3] = affine[:3, :3][reorder, :][:, reorder]
    affine_zyx[:3, 3] = affine[:3, 3][reorder]

    return affine_zyx

def apply_affine_transform_to_volume(volume: sitk.Image, affine_matrix: np.ndarray) -> sitk.Image:
    """
    Apply a 4x4 affine transformation to a 3D SimpleITK volume.

    Parameters:
        volume (sitk.Image): The input 3D volume.
        affine_matrix (np.ndarray): A 4x4 affine transformation matrix.

    Returns:
        sitk.Image: The transformed volume.
    """
    if affine_matrix.shape != (4, 4):
        raise ValueError("Affine matrix must be 4x4")

    # Extract the 3x3 matrix (rotation + scale + shear) and the translation vector
    matrix_3x3 = affine_matrix[:3, :3].flatten().tolist()
    translation = affine_matrix[:3, 3].tolist()

    # Create the affine transform
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(matrix_3x3)
    transform.SetTranslation(translation)

    # Set the center of the transform to the center of the image
    center = np.array(volume.GetSize()) / 2.0
    center_physical = volume.TransformContinuousIndexToPhysicalPoint(center.tolist())
    transform.SetCenter(center_physical)

    # Resample the volume using the transform
    resampled = sitk.Resample(
        volume,
        volume,  # Use same reference image
        transform,
        sitk.sitkLinear,
        0.0,  # Default pixel value for out-of-bounds
        volume.GetPixelID()
    )

    return resampled


def read_ants_affine_transform(file_path):
    """
    Reads an ANTs .mat affine transform file and returns a 4x4 numpy affine matrix.

    Args:
        file_path (str): Path to the .mat file.

    Returns:
        np.ndarray: A 4x4 affine transformation matrix.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Parse parameters
    params_line = next(line for line in lines if line.startswith("Parameters:"))
    fixed_params_line = next(line for line in lines if line.startswith("FixedParameters:"))

    params = list(map(float, params_line.strip().split()[1:]))
    fixed_params = list(map(float, fixed_params_line.strip().split()[1:]))

    matrix = np.array(params[:9]).reshape(3, 3)
    translation = np.array(params[9:12])
    center = np.array(fixed_params)

    # Build the affine transform with rotation center
    affine = np.eye(4)
    affine[:3, :3] = matrix
    affine[:3, 3] = translation + center - matrix @ center

    return affine

if __name__ == "__main__":
    #register_large_zarr_datasets()
    #rescale_transform()
    from scipy.io import loadmat
    
    affine = np.array([
        [1.0, 0.0, 0.0, 2],
        [0.0, 1.0, 0.0, 5],
        [0.0, 0.0, 1.0, 15],
        [0.0, 0.0, 0.0, 1.0]
    ])
    affine = np.load('/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/ALLEN771602/ALLEN771602_Allen_32.0x28.8x28.8um_sagittal.npy')
    rot = affine[:3, :3]
    rot = np.rot90(rot, k=2, axes=(0, 1))
    affine[:3, :3] = rot
    print("Affine matrix from elastix numpy")
    print(affine)
    print()
    transform_filepath = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/ALLEN771602/ALLEN771602_Allen_32.0x28.8x28.8um_Affine.mat'
    transfo_dict = loadmat(transform_filepath)
    lps2ras = np.diag([-1, -1, 1])

    rot = transfo_dict['AffineTransform_float_3_3'][0:9].reshape((3, 3))
    trans = transfo_dict['AffineTransform_float_3_3'][9:12]
    offset = transfo_dict['fixed']
    print("Ants offset")
    print(offset)
    print()
    r_trans = (np.dot(rot, offset) - offset - trans).T * [1, 1, -1]

    affine = np.eye(4)
    affine[0:3, 3] = r_trans
    affine[:3, :3] = np.dot(np.dot(lps2ras, rot), lps2ras)

    print("Affine matrix from ants mat")
    print(affine)
    print()


    print("Affine matrix from ants mat transformed to 4x4")
    transform_filepath = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/ALLEN771602/ALLEN771602_Allen_32.0x28.8x28.8um_Affine.mat'
    affine = read_ants_affine_transform(transform_filepath)
    print(affine)
    print()


    filepath = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/ALLEN771602/ALLEN771602_Allen_32.0x28.8x28.8um_sagittal/elastix_output/TransformParameters.0.txt'
    affine_elastix = elastix_to_affine_4x4(filepath)
    print("Affine matrix from elastix TransformParameters.0.txt")
    print(affine_elastix)

    moving_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/ALLEN771602/ALLEN771602_32.0x28.8x28.8um_sagittal.tif'
    moving = read_image(moving_path)
    reference_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/Allen/Allen_32.0x28.8x28.8um_sagittal.tif'
    reference = read_image(reference_path)

    output_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/ALLEN771602/transformed.tif'
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. removing")
        os.remove(output_path)
    transformed = apply_affine_transform_to_volume(sitk.GetImageFromArray(moving),  affine)
    sitk.WriteImage(transformed, output_path)
    """
    ants_img = process_block_with_ants(reference, moving, transform_filepath)
    write_image(output_path, ants_img)
    """
    print(f"Transformed image saved to {output_path}")


    exit(1)
    
    reg_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
    moving_path = os.path.join(reg_path, 'ALLEN771602', 'ALLEN771602_32.0x28.8x28.8um_sagittal.zarr')
    output_path = os.path.join(reg_path, 'ALLEN771602', 'transformed_large.zarr')

    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. removing")
        shutil.rmtree(output_path)
    print(f"Applying affine transform to large Zarr dataset from \n{moving_path} to \n{output_path}...")


    #apply_affine_transform_to_large_zarr(
    #    input_zarr_path=moving_path,
    #    output_zarr_path=output_path,
    #    affine_matrix=affine
    #)
        
    apply_affine_transform_zarr(
        zarr_input_path=moving_path,
        zarr_output_path=output_path,
        affine_matrix=affine)