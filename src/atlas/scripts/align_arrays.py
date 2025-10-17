import os
import sys
import numpy as np
import SimpleITK as sitk
from pathlib import Path
PIPELINE_ROOT = Path('./src').absolute()
PIPELINE_ROOT = PIPELINE_ROOT.as_posix()
sys.path.append(PIPELINE_ROOT)
print(PIPELINE_ROOT)

from library.atlas.atlas_utilities import center_images_to_largest_volume

def numpy_to_sitk(arr):
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetOrigin((0, 0, 0))
    img.SetSpacing((1, 1, 1))
    return img

def sitk_to_numpy(img):
    return sitk.GetArrayFromImage(img)

def rigid_register(fixed, moving):
    """Return the rigid transform aligning moving to fixed."""
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(32)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Set initial transform (rigid)
    initial_transform = sitk.VersorRigid3DTransform()
    initial_transform.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint(np.array(fixed.GetSize()) / 2.0))
    registration_method.SetInitialTransform(initial_transform, inPlace=False)



    #registration_method.SetInitialTransform(sitk.CenteredTransformInitializer(
    #    fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS
    #))
    registration_method.SetMetricSamplingPercentage(0.2, sitk.sitkWallClock)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetShrinkFactorsPerLevel([4,2,1])
    registration_method.SetSmoothingSigmasPerLevel([2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    transform = registration_method.Execute(fixed, moving)
    return transform

def average_transforms(transforms):
    """Average multiple 3D Euler transforms."""
    tx, ty, tz, rx, ry, rz = [], [], [], [], [], []
    for t in transforms:
        p = t.GetParameters()
        rx.append(p[0]); ry.append(p[1]); rz.append(p[2])
        tx.append(p[3]); ty.append(p[4]); tz.append(p[5])
    avg = sitk.Euler3DTransform()
    avg.SetParameters([
        np.mean(rx), np.mean(ry), np.mean(rz),
        np.mean(tx), np.mean(ty), np.mean(tz)
    ])
    return avg

def groupwise_registration(arrays, iterations=3):
    """Align multiple 3D binary arrays without a fixed reference."""
    sitk_images = [numpy_to_sitk(a) for a in arrays]
    transforms = [sitk.Euler3DTransform() for _ in sitk_images]

    for it in range(iterations):
        print(f"Iteration {it+1}/{iterations}")

        # Compute the current average image in the "group space"
        transformed_images = [sitk.Resample(img, sitk_images[0], t, sitk.sitkLinear, 0.0) 
                              for img, t in zip(sitk_images, transforms)]
        avg_img = sum(transformed_images) / len(transformed_images)

        # Register each image to this average (not a fixed reference)
        new_transforms = []
        for img, t_init in zip(sitk_images, transforms):
            transform = rigid_register(avg_img, img)
            # Combine with previous transform
            new_t = sitk.Euler3DTransform()
            new_t.SetParameters(transform.GetParameters())
            new_transforms.append(new_t)

        # Average all transforms to get group consensus
        consensus = average_transforms(new_transforms)
        transforms = [sitk.Euler3DTransform(consensus) for _ in new_transforms]

    # Apply final transforms and make averaged overlap image
    final_images = [sitk.Resample(img, sitk_images[0], t, sitk.sitkLinear, 0.0) 
                    for img, t in zip(sitk_images, transforms)]
    avg_final = sum(final_images) / len(final_images)

    return sitk_to_numpy(avg_final), [t.GetParameters() for t in transforms]

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
 
    unaligned_arrays = []
    animals = ['MD585', 'MD589', 'MD594']
    structure = 'SC'
    data_path = "/net/birdstore/Active_Atlas_Data/data_root/atlas_data"
    for animal in animals:
        inpath = os.path.join(data_path, animal, "structure", f"{structure}.npy")
        arr = np.load(inpath)
        print(f'Loaded {inpath} with shape {arr.shape} with dtype {arr.dtype}')
        sitk_arr = numpy_to_sitk(arr)
        #del arr
        unaligned_arrays.append(sitk_arr)

    centered_images = center_images_to_largest_volume(unaligned_arrays)


    """
    registered, consensus, transforms = groupwise_registration(centered_images, max_iterations=3)
    for i in registered:
        print(f'Registered shape: {i.shape}, dtype: {i.dtype}')
    print("Registration complete.")

    result = np.mean(registered, axis=0)
    print(f"Result shape: {result.shape}, dtype: {result.dtype}")
    ids, counts = np.unique(result, return_counts=True)
    print(f"Unique ids in result: {ids}, counts: {counts}")
    print(f"Consensus shape: {consensus.shape}, dtype: {consensus.dtype}")
    """
    centered_images = [sitk_to_numpy(img) for img in centered_images]
    for c in centered_images:
        print(f'Centered shape: {c.shape}, dtype: {c.dtype}')
    avg, transforms = groupwise_registration(centered_images, iterations=3)

    print("Final average shape:", avg.shape)
    print("Transforms (rx, ry, rz, tx, ty, tz):")
    for t in transforms:
        print(np.round(t, 3))

    exit(1)
 