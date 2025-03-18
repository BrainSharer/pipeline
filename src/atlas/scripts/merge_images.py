import SimpleITK as sitk
import numpy as np


def resample_image(image, reference_image):
    """
    Resamples an image to match the reference image in size, spacing, and direction.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkLinear)  # Linear interpolation for resampling
    resampler.SetDefaultPixelValue(0)  # Fill with zero if needed
    return resampler.Execute(image)

def average_images(image_paths):
    """
    Loads multiple 3D images, resamples them to a common reference, and averages them.
    """
    images = [sitk.ReadImage(path) for path in image_paths]
    
    # Choose the reference image (first image in the list)
    reference_image = images[0]

    # Resample all images to the reference
    resampled_images = [resample_image(img, reference_image) for img in images]

    # Convert images to numpy arrays and compute the average
    image_arrays = [sitk.GetArrayFromImage(img) for img in resampled_images]
    avg_array = np.mean(image_arrays, axis=0)

    # Convert back to SimpleITK image
    avg_image = sitk.GetImageFromArray(avg_array)
    avg_image.CopyInformation(reference_image)  # Copy metadata

    return avg_image

# Example usage
if __name__ == "__main__":
    # Load example volumes (assuming Nifti format)
    image1 = sitk.ReadImage("volume1.nii.gz")
    image2 = sitk.ReadImage("volume2.nii.gz")
    image3 = sitk.ReadImage("volume3.nii.gz")
    
    merged_image = average_images([image1, image2, image3])
    
    # Save the merged image
    sitk.WriteImage(merged_image, "merged_volume.nii.gz")