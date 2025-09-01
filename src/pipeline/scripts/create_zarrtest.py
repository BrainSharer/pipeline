
#!/usr/bin/env python3
"""
Build a Zarr volume from a folder of single-plane TIFFs (grayscale or RGB),
choosing near-optimal Dask chunk sizes based on available RAM.

Dependencies:
  pip install dask[delayed] distributed tifffile zarr numcodecs psutil

Usage:
  python tiffs_to_zarr.py /path/to/tiffs /path/to/out.zarr
"""

import os
import math
import glob
import psutil
from typing import Sequence, Tuple, Optional

import dask
import dask.array as da
from dask.distributed import LocalCluster, Client
import numpy as np
import tifffile
import zarr
from numcodecs import Blosc
from distributed import progress

# ----------------------------
# Chunk sizing heuristics
# ----------------------------

def _nice_tile(n: int, target: int) -> int:
    """
    Pick a 'nice' tile size <= n and close to target (power-of-two-ish),
    useful for Y/X chunk dimensions.
    """
    if n <= target:
        return n
    # prefer common tile sizes: 512, 1024, 2048...
    candidates = [128, 256, 512, 1024, 1536, 2048, 3072, 4096]
    # include exact divisors near target
    candidates += [max(1, n // k) for k in range(2, 8)]
    # bound and pick best
    candidates = sorted(set(c for c in candidates if 1 <= c <= n), key=lambda c: abs(c - target))
    return candidates[0] if candidates else min(n, target)


def compute_optimal_chunks(
    volume_shape: Sequence[int],
    dtype: np.dtype,
    available_bytes: int,
    n_workers: int,
    is_rgb: bool
) -> Tuple[int, int, int, Optional[int]]:
    """
    Compute chunk sizes (z, y, x[, c]) targeting a fraction of available memory per worker.

    Heuristic:
      - budget_per_worker = 0.6 * available / n_workers
      - target_chunk = ~1/4 of budget_per_worker (keeps multiple chunks in-flight)
      - prefer Y/X tiles ~ 1024 (or smaller if image is small)
      - derive Z chunk so that (z*y*x*c*bytes_per_pixel) ~ target_chunk

    Returns:
      (cz, cy, cx[, cc])
    """
    shape = tuple(int(s) for s in volume_shape)
    if is_rgb:
        z, y, x, c = shape
        cc = c  # store full color channel in each chunk
    else:
        z, y, x = shape
        cc = None

    itemsize = np.dtype(dtype).itemsize

    # Memory budget
    budget_per_worker = (0.60 * available_bytes) / max(1, n_workers)
    target_chunk_bytes = budget_per_worker / 4.0  # aim to keep ~4 chunks resident

    # Start with a reasonable Y/X tile
    cy = _nice_tile(y, 2048 if y >= 2048 else y)
    cx = _nice_tile(x, 2048 if x >= 2048 else x)

    # Compute Z depth from target bytes
    per_voxel_channels = (cc if is_rgb else 1)
    plane_bytes = cy * cx * per_voxel_channels * itemsize

    # Guard against extremely small planes
    if plane_bytes == 0:
        plane_bytes = max(itemsize, 1)

    cz = max(1, int(target_chunk_bytes // plane_bytes))
    # Don't exceed available Z
    cz = min(cz, z)

    # Keep cz in a "friendly" range: prefer powers of two-ish
    if cz > 1:
        pow2 = 2 ** int(math.log2(cz))
        # choose between pow2 and next power if closer
        next_pow2 = pow2 * 2
        cz = pow2 if abs(pow2 - cz) <= abs(next_pow2 - cz) or next_pow2 > z else next_pow2

    # Avoid tiny chunk counts: bump cy/cx if chunk is too small (< 4 MB)
    chunk_bytes = cz * plane_bytes
    if chunk_bytes < 4 * 1024 * 1024:
        # try to increase tiles up to full dims while keeping under target
        for _ in range(2):
            # expand Y
            ny = min(y, cy * 2)
            nb = cz * ny * cx * per_voxel_channels * itemsize
            if nb <= target_chunk_bytes:
                cy = ny
                chunk_bytes = nb
            # expand X
            nx = min(x, cx * 2)
            nb = cz * cy * nx * per_voxel_channels * itemsize
            if nb <= target_chunk_bytes:
                cx = nx
                chunk_bytes = nb

    if is_rgb:
        return (cz, cy, cx, cc)
    else:
        return (cz, cy, cx, None)

# ----------------------------
# TIFF loading as Dask array
# ----------------------------

def _read_tiff(path: str) -> np.ndarray:
    """Load a single TIFF plane (grayscale HxW or color HxWxC)."""
    with tifffile.TiffFile(path) as tf:
        arr = tf.asarray()
    return arr

def build_dask_stack_from_folder(folder: str) -> Tuple[da.Array, bool]:
    """
    Reads all TIFF files in a folder (sorted by name) and returns a Dask array:
      grayscale -> (Z, Y, X)
      RGB/RGBA  -> (Z, Y, X, C)

    Returns:
      (dask_array, is_rgb)
    """
    paths = sorted([p for p in glob.glob(os.path.join(folder, "*.tif*"))])
    if not paths:
        raise FileNotFoundError(f"No TIFFs found in: {folder}")

    # Inspect first image for shape/dtype
    with tifffile.TiffFile(paths[0]) as tf:
        sample = tf.asarray()
        dtype = sample.dtype
        is_rgb = (sample.ndim == 3 and sample.shape[-1] in (3, 4))

    # delayed reads
    print(f'sample shape={sample.shape} dtype={dtype}')
    delayed_slices = [dask.delayed(_read_tiff)(p) for p in paths]
    arrays = [da.from_delayed(ds, shape=sample.shape, dtype=dtype) for ds in delayed_slices]

    #arrays = [da.from_delayed(lazy_image,           # Construct a small Dask array
    #                        dtype=self.dtype,   # for every lazy value
    #                        shape=self.img_shape)
    #        for lazy_image in lazy_images]



    # Stack along Z
    stack = da.stack(arrays, axis=0)
    # Ensure expected dims
    if is_rgb:
        if stack.ndim != 4:
            raise ValueError(f"Expected (Z,Y,X,C) for color images, got {stack.shape}")
    else:
        if stack.ndim != 3:
            raise ValueError(f"Expected (Z,Y,X) for grayscale images, got {stack.shape}")

    return stack, is_rgb

# ----------------------------
# Dask cluster sizing
# ----------------------------

def make_local_cluster() -> Tuple[Client, LocalCluster, int]:
    """
    Create a LocalCluster sized to system resources.
    - threads_per_worker = 1 (image IO benefits from process-level parallelism)
    - n_workers = min(physical cores, max 2) but also bounded by memory
      such that each worker gets at least ~4GB when possible.
    """
    vm = psutil.virtual_memory()
    total = vm.total


    # Prefer physical cores
    try:
        n_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    except Exception:
        n_cores = os.cpu_count() or 1

    # Aim for at least ~3–4 GB per worker (tuneable)
    min_gb_per_worker = 3.5 * (1024 ** 3)
    max_workers_by_mem = max(1, int(total // min_gb_per_worker))
    n_workers = max(2, min(n_cores, max_workers_by_mem)) if n_cores >= 2 else 1
    n_workers = 1

    # Memory limit per worker uses ~80% of available divided across workers
    per_worker_limit = int(0.85 * total / n_workers)

    print(f'Creating Dask cluster with {n_workers} workers, {per_worker_limit} bytes per worker')
    #exit(1)

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=4,
        memory_limit=per_worker_limit,
        dashboard_address=":8787",  # set ":8787" if you want the dashboard
    )
    client = Client(cluster)
    return client, cluster, n_workers

# ----------------------------
# Main build function
# ----------------------------

def build_zarr_from_tiffs(in_folder: str, out_zarr: str) -> None:
    """
    Build a Zarr array at `out_zarr` from TIFFs in `in_folder`, chunked optimally.

    Returns a dict with metadata: shape, dtype, chunks, workers, path.
    """
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

    # 1) Start cluster
    client, cluster, n_workers = make_local_cluster()

    try:
        # 2) Load TIFF stack lazily
        darr, is_rgb = build_dask_stack_from_folder(in_folder)
        shape = tuple(int(s) for s in darr.shape)
        dtype = darr.dtype

        # 3) Compute optimal chunks from available mem (fresh)
        avail = psutil.virtual_memory().available
        cz, cy, cx, cc = compute_optimal_chunks(shape, dtype, avail, n_workers, is_rgb)
        chunks = (cz, cy, cx, cc) if is_rgb else (cz, cy, cx)
        print(f'optimum chunks={chunks}')

        # 4) Rechunk (logical only; dask computes on write)
        stack = darr.rechunk(chunks)

        # 5) Create Zarr store and write
        if os.path.exists(out_zarr):
            # allow removing pre-existing zarr dir
            import shutil
            print(f'Removing {out_zarr}')
            shutil.rmtree(out_zarr)

        store = zarr.storage.NestedDirectoryStore(out_zarr)

        z = zarr.zeros(
            stack.shape,
            chunks=chunks,
            store=store,
            overwrite=True,
            compressor=compressor,
            dtype=stack.dtype,
        )

        with Client(cluster) as client:
            to_store = da.store(stack, z, lock=False, compute=False)
            to_store = client.compute(to_store)
            progress(to_store)
            to_store = client.gather(to_store)


    finally:
        cluster.close()

# ----------------------------
# CLI
# ----------------------------


if __name__ == "__main__":
    tif_input = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK37/preps/C1/full_aligned"
    outpath_zarr = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK37/www/neuroglancer_data/C1.zarr"
    build_zarr_from_tiffs(tif_input, outpath_zarr)
    print("Wrote Zarr:")
    store = zarr.storage.NestedDirectoryStore(outpath_zarr)
    volume = zarr.open(store, 'r')
    print(volume.info)
    print(f'volume.shape={volume.shape}')



