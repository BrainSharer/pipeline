import argparse
import os
import numpy as np
import tifffile
from tqdm import tqdm
import SimpleITK as sitk
from scipy.ndimage import zoom

def resample_numpy_volume(volume_arr, old_spacing, new_spacing, is_mask=False):
    # 1. Convert NumPy to SimpleITK
    image = sitk.GetImageFromArray(volume_arr)
    image.SetSpacing(old_spacing)
    
    # 2. Calculate new size
    orig_size = np.array(image.GetSize())
    orig_spacing = np.array(image.GetSpacing())
    new_size = orig_size * (orig_spacing / new_spacing)
    new_size = np.ceil(new_size).astype(int).tolist() # Dimensions must be integers
    
    # 3. Setup Resample Filter
    resample = sitk.ResampleImageFilter()
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    
    # Choose interpolator
    if is_mask:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
    
    # 4. Execute and convert back
    resampled_sitk = resample.Execute(image)
    return resampled_sitk


def create_moving_image(base_path):

    file_path = os.path.join(base_path, 'C1', 'thumbnail_aligned')

    files = sorted(os.listdir(file_path))
    files = [os.path.join(file_path, f) for f in tqdm(files)]
    vol = np.stack([tifffile.imread(f) for f in tqdm(files)])
    # vol is in z,y,x order, need to put in 
    print(f'{vol.shape=}')

    old_spacing = (1,1,1)
    xy = 10.4/10
    xy = 10/10.4
    new_spacing = (2, xy, xy)
    new_spacing = 0.075

    #image = resample_numpy_volume(vol, old_spacing, new_spacing, is_mask=False)
    #vol = sitk.GetArrayFromImage(image)
    vol = zoom(vol, new_spacing, order=1)
    #print("Voxel spacing,size:", image.GetSpacing(), image.GetSize())
    print(f'{vol.shape=} {vol.dtype=}')

    #vol = np.transpose(vol, (2, 0, 1))
    #print(f'after trans {vol.shape=} {vol.dtype=}')

    volpath = os.path.join(base_path, "testing.tif")
    tifffile.imwrite(
        volpath, vol.astype(np.uint16), bigtiff=True
    )
    print('Saved moving image to', volpath)
    return

    base_path = os.path.join(base_path, 'C1', 'registration')
    outpath = os.path.join(base_path, "moving.nii")
    sitk.WriteImage(image, outpath)


def fix_fixed(base_path):
    fixed_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/Allen/"
    fixed_file_path = os.path.join(fixed_path, "Allen_10um_coronal.tif")
    old_spacing = (1, 1, 1)
    new_spacing = (1, 1, 1)
    vol = tifffile.imread(fixed_file_path)
    #image = resample_numpy_volume(vol, old_spacing, new_spacing, is_mask=False)
    image = sitk.GetImageFromArray(vol)
    print("Voxel spacing (in mm):", image.GetSpacing())

    outpath = os.path.join(base_path, "fixed.nii")
    sitk.WriteImage(image, outpath)
    return

def register_images(base_path):
    fixed_file_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration/Allen/Allen_10.0x10.0x10.0um_sagittal.tif"
    fixed_image = sitk.Cast(sitk.ReadImage(fixed_file_path), sitk.sitkFloat32)
    print('Read fixed image from', fixed_file_path)
    fixed_image = sitk.ConstantPad(fixed_image, [0, 0, 0], [2200, 0, 0], 0)
    print(f'size={fixed_image.GetSize()}, spacing={fixed_image.GetSpacing()}')

    moving_file_path = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/CTB016/preps/moving.tif"
    moving_image = sitk.Cast(sitk.ReadImage(moving_file_path), sitk.sitkFloat32)
    print('Read moving image from', moving_file_path)
    print(f'size={moving_image.GetSize()}, spacing={moving_image.GetSpacing()}')

    #moving_mask = sitk.OtsuThreshold(moving_image, 0, 255)
    #fixed_mask = sitk.OtsuThreshold(fixed_image, 0, 255)

    # 2. Initial Alignment (Center to Center)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, 
        moving_image, 
        sitk.Euler3DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # 3. Setup Registration Method
    registration_method = sitk.ImageRegistrationMethod()
    # Limit metric evaluation to the masked regions
    #registration_method.SetMetricFixedMask(fixed_mask)
    #registration_method.SetMetricMovingMask(moving_mask)
    # Similarity metric settings
    #registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.025)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, 
        numberOfIterations=250, 
        convergenceMinimumValue=1e-6, 
        convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the rigid transform
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # 4. Execute Registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # 5. Resample the moving image to the fixed image's grid
    moving_resampled = sitk.Resample(
        moving_image, 
        fixed_image, 
        final_transform, 
        sitk.sitkLinear, 
        0.0, 
        moving_image.GetPixelID()
    )

    outpath = os.path.join(base_path, "aligned_output.nii")
    sitk.WriteImage(moving_resampled, outpath)
    print('Wrote aligned image to', outpath)
    print(f'size={moving_resampled.GetSize()}, spacing={moving_resampled.GetSpacing()}')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)
    args = parser.parse_args()

    animal = args.animal

    base_dir = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps'


    create_moving_image(base_dir)
    #fix_fixed(base_dir)
    base_dir = os.path.join(base_dir, 'C1', 'registration')
    #register_images(base_dir)
