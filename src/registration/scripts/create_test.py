import numpy as np
import dask.array as da
import SimpleITK as sitk
from dask import delayed
from dask.array import map_blocks


def register_block(moving_block, fixed_block, block_origin):
    """
    Perform affine registration of a moving block against the fixed block using SimpleITK.
    """
    # Convert to sitk images
    moving_sitk = sitk.GetImageFromArray(moving_block.astype(np.float32))
    fixed_sitk = sitk.GetImageFromArray(fixed_block.astype(np.float32))

    # Center images for transform initialization
    transform = sitk.CenteredTransformInitializer(
        fixed_sitk, moving_sitk, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # Registration method
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    registration_method.SetInitialTransform(transform, inPlace=False)
    registration_method.SetShrinkFactorsPerLevel([2])
    registration_method.SetSmoothingSigmasPerLevel([1])
    
    final_transform = registration_method.Execute(fixed_sitk, moving_sitk)

    # Resample moving block
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(final_transform)
    registered_block = resampler.Execute(moving_sitk)

    return sitk.GetArrayFromImage(registered_block)


def blend_weights(shape, overlap):
    """
    Create blending weights for smooth overlapping.
    """
    z, y, x = [np.linspace(0, 1, s) for s in shape]
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')

    wz = np.clip(np.minimum(zz, 1 - zz) * shape[0] / overlap[0], 0, 1)
    wy = np.clip(np.minimum(yy, 1 - yy) * shape[1] / overlap[1], 0, 1)
    wx = np.clip(np.minimum(xx, 1 - xx) * shape[2] / overlap[2], 0, 1)

    return wz * wy * wx


def chunkwise_register(moving, fixed, block_shape, overlap):
    """
    Apply registration to each chunk with overlap and blend the results.
    """
    # Pad volumes to handle overlap
    moving = moving.map_blocks(lambda x: np.pad(x, [(overlap[i], overlap[i]) for i in range(3)]), dtype=moving.dtype)
    fixed = fixed.map_blocks(lambda x: np.pad(x, [(overlap[i], overlap[i]) for i in range(3)]), dtype=fixed.dtype)

    # Compute new chunking with overlap
    chunks = tuple(s for s in block_shape)
    out_shape = moving.shape

    out = da.zeros(out_shape, dtype=np.float32)
    weight_sum = da.zeros_like(out)

    for z in range(0, out.shape[0] - block_shape[0] + 1, block_shape[0] - overlap[0]):
        for y in range(0, out.shape[1] - block_shape[1] + 1, block_shape[1] - overlap[1]):
            for x in range(0, out.shape[2] - block_shape[2] + 1, block_shape[2] - overlap[2]):
                zs, ys, xs = z, y, x
                ze, ye, xe = z + block_shape[0], y + block_shape[1], x + block_shape[2]

                # Extract subvolumes
                moving_block = moving[zs:ze, ys:ye, xs:xe]
                fixed_block = fixed[zs:ze, ys:ye, xs:xe]

                # Delayed block registration
                result = delayed(register_block)(moving_block.compute(), fixed_block.compute(), (zs, ys, xs))

                # Create blend weights
                weights = blend_weights(block_shape, overlap)

                # Add weighted block
                weighted_block = da.from_delayed(delayed(lambda r: r * weights)(result),
                                                 shape=block_shape, dtype=np.float32)

                out[zs:ze, ys:ye, xs:xe] += weighted_block
                weight_sum[zs:ze, ys:ye, xs:xe] += weights

    # Avoid division by zero
    weight_sum = da.where(weight_sum == 0, 1, weight_sum)
    return out / weight_sum


# Example usage
if __name__ == "__main__":
    # Load example large volumes (use real zarr/N5 in practice)
    shape = (100, 100, 100)
    chunk_size = (shape[0]//2, shape[1]//2, shape[0]//2)  # Example chunk size
    overlap = (2, 2, 2)

    # Simulated volumes
    moving = da.random.random(shape, chunks=chunk_size)
    fixed = da.random.random(shape, chunks=chunk_size)

    result = chunkwise_register(moving, fixed, block_shape=chunk_size, overlap=overlap)

    # Save or compute result
    result.to_zarr("registered_volume.zarr", overwrite=True)
