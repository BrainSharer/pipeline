import shutil
import dask.array as da
from dask.diagnostics import ProgressBar
from dask import delayed
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

if __name__ == "__main__":
    register_large_zarr_datasets()
    #rescale_transform()
    
