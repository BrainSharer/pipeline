import os
import psutil
import dask.array as da
import dask_image.imread
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
import numpy as np


def compute_optimal_chunks(shape, dtype, n_channels=1, safety_factor=0.5):
    """
    Compute optimal chunk size based on system memory and array shape.
    """
    dtype_16bit = np.dtype(dtype)
    dask_image = da.zeros(shape, chunks=shape, dtype=dtype_16bit)
    available_mem = psutil.virtual_memory().available
    n_workers = os.cpu_count()
    #bytes_per_element = da.core.dtype(dtype).itemsize
    bytes_per_element = dask_image.itemsize
    total_elements = shape[1] * shape[2] * n_channels  # height x width x channels
    max_chunk_bytes = (available_mem / n_workers) * safety_factor

    # Start with 64 or full image depth if smaller than 64
    chunk_planes = min(64, shape[0])
    chunk_bytes = chunk_planes * total_elements * bytes_per_element

    print(f'available memory: {available_mem}, max_chunk_bytes: {max_chunk_bytes}, chunk_bytes: {chunk_bytes}')

    # Adjust number of planes per chunk to fit into max_chunk_bytes
    while chunk_bytes > max_chunk_bytes and chunk_planes > 1:
        chunk_planes = chunk_planes // 2
        chunk_bytes = chunk_planes * total_elements * bytes_per_element

    return (chunk_planes, shape[1], shape[2], n_channels)


def tif_stack_to_zarr(input_folder, output_zarr_path):
    """
    Reads single-plane RGB TIFFs from a folder and writes them as a Dask array to Zarr.
    """
    # Start local Dask cluster
    #cluster = LocalCluster()
    #client = Client(cluster)

    # Read all TIFFs using dask-image
    tif_pattern = os.path.join(input_folder, "*.tif")
    img = dask_image.imread.imread(tif_pattern)  # shape: (num_images, height, width, 3)

    # Infer dtype and shape
    dtype = img.dtype
    shape = img.shape  # (Z, Y, X, C)

    # Compute chunk sizes
    chunk_size = compute_optimal_chunks(shape, dtype)

    print(f"Using optimal chunk size {chunk_size} dtype={dtype} shape={shape}")
    exit(1)
    # Rechunk
    img_rechunked = img.rechunk(chunk_size)

    # Write to Zarr
    img_rechunked.to_zarr(output_zarr_path, overwrite=True, compute=False)

    with ProgressBar():
        img_rechunked.compute()



    #da.to_zarr(img_rechunked, output_zarr_path, overwrite=True)

    print(f"Zarr written to: {output_zarr_path}")
    client.close()


tif_input = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK37/preps/C1/full_aligned"
outpath_zarr = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK37/www/neuroglancer_data/C1.zarr"

tif_stack_to_zarr(tif_input, outpath_zarr)