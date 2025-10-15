import os
import sys
import numpy as np
import SimpleITK as sitk

from pathlib import Path


PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.atlas.atlas_utilities import center_images_to_largest_volume

def numpy_to_sitk(arr):
    """Convert numpy 3D array to SimpleITK image."""
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing((1.0, 1.0, 1.0))
    return img

def sitk_to_numpy(img):
    """Convert SimpleITK image back to numpy."""
    return sitk.GetArrayFromImage(img)

def rigid_register(fixed, moving):
    """Perform rigid registration using mutual information."""
    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(fixed)
    elastix.SetMovingImage(moving)

    # Rigid registration parameters
    params = sitk.GetDefaultParameterMap("rigid")
    params["Metric"] = ["AdvancedMattesMutualInformation"]
    params["MaximumNumberOfIterations"] = ["128"]
    params["AutomaticTransformInitialization"] = ["true"]
    params["AutomaticTransformInitializationMethod"] = ["GeometricalCenter"]

    params["AutomaticScalesEstimation"] = ["true"]
    params["NumberOfSamplesForExactGradient"] = ["10000"]
    elastix.SetParameterMap(params)
    elastix.LogToConsoleOff()
    elastix.Execute()
    result = elastix.GetResultImage()
    transform = elastix.GetTransformParameterMap()
    return result, transform

def apply_transform(moving, transform):
    """Apply an existing transform to an image."""
    transformix = sitk.TransformixImageFilter()
    transformix.SetMovingImage(moving)
    transformix.SetTransformParameterMap(transform)
    transformix.LogToConsoleOff()
    transformix.Execute()
    return transformix.GetResultImage()

def groupwise_registration(sitk_vols, max_iterations=5):
    """
    Perform groupwise registration to maximize overlap of binary volumes.
    None of the arrays is the reference.
    """
    
    # Initialize with average image
    avg_img = sum(sitk_vols) / len(sitk_vols)

    for iteration in range(max_iterations):
        registered_imgs = []
        transforms = []

        for img in sitk_vols:
            result, transform = rigid_register(avg_img, img)
            registered_imgs.append(result)
            transforms.append(transform)
        
        # Compute new average (consensus)
        avg_img = sum(registered_imgs) / len(registered_imgs)

        print(f"Iteration {iteration+1}/{max_iterations} complete")

    # Convert results to numpy
    registered_arrays = [sitk_to_numpy(img) for img in registered_imgs]
    avg_array = sitk_to_numpy(avg_img)

    return registered_arrays, avg_array, transforms


if __name__ == "__main__":
    # Example: 3 random binary blobs
    # Example: align 3 binary masks
    unaligned_arrays = []
    animals = ['MD585', 'MD589', 'MD594']
    structure = 'SC'
    data_path = "/net/birdstore/Active_Atlas_Data/data_root/atlas_data"
    for animal in animals:
        inpath = os.path.join(data_path, animal, "structure", f"{structure}.npy")
        arr = np.load(inpath)
        print(f'Loaded {inpath} with shape {arr.shape} with dtype {arr.dtype}')
        sitk_arr = numpy_to_sitk(arr)
        del arr
        unaligned_arrays.append(sitk_arr)

    centered_images = center_images_to_largest_volume(unaligned_arrays)

    registered, consensus, transforms = groupwise_registration(centered_images, max_iterations=3)
    for i in registered:
        print(f'Registered shape: {i.shape}, dtype: {i.dtype}')
    print("Registration complete.")