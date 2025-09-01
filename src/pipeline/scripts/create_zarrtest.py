"""
dask_zarr_writer.py

Produces optimized chunk sizes for a Dask array (built from many TIFFs), starts
a local Dask cluster taking system RAM and CPUs into account, and writes an
NGFF/OME-Zarr multiscale (pyramid) with progress reporting.

Author: ChatGPT (GPT-5 Thinking mini)
"""

import os
import math
import json
import psutil
import glob
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import tifffile
import zarr
import dask.array as da
from dask import delayed
from distributed import LocalCluster, Client, progress

# Optional downsampling
try:
    from skimage.transform import downscale_local_mean
except Exception:
    downscale_local_mean = None


def compute_available_memory_bytes(fraction: float = 0.8) -> int:
    """
    Returns an approximation of usable system RAM in bytes.
    fraction: fraction of total system memory we consider usable (0 < fraction <= 1).
    """
    vm = psutil.virtual_memory()
    # consider fraction of available memory (available instead of total)
    usable = int(vm.available * fraction)
    return usable


def start_local_dask_cluster(n_workers: Optional[int] = None,
                             threads_per_worker: Optional[int] = None,
                             memory_fraction_per_worker: float = 0.9) -> Client:
    """
    Starts a LocalCluster sized to the machine. Returns a dask.distributed.Client.
    n_workers: if None, defaults to number of physical CPUs.
    threads_per_worker: default 1.
    memory_fraction_per_worker: fraction of psutil.available each worker may use (heuristic).
    """
    n_workers = 1
    threads_per_worker = 2

    # compute per-worker memory limit
    total_usable = compute_available_memory_bytes(fraction=0.9)
    per_worker_mem = max(256 * 1024**2, int(total_usable / max(1, n_workers)))  # at least 256MB
    mem_limit_str = f"{per_worker_mem}B"
    print(f"Starting Dask LocalCluster with {n_workers} workers, {threads_per_worker} threads per worker, "
          f"and {mem_limit_str} memory limit per worker.")
    cluster = LocalCluster(n_workers=n_workers,
                           threads_per_worker=threads_per_worker,
                           memory_limit=per_worker_mem,
                           silence_logs=logging_level_to_int("info"))
    client = Client(cluster)
    return client


def logging_level_to_int(level_name: str) -> int:
    import logging
    return getattr(logging, level_name.upper(), logging.INFO)


def read_tiff_delayed(path: str):
    """
    Return a delayed object that reads a single tiff (using tifffile) and returns numpy array.
    """
    @delayed
    def _read(p):
        arr = tifffile.imread(p)
        # ensure shape is (Z=1?, Y, X, C) or (Y, X, C)
        return arr
    return _read(path)


def build_dask_array_from_folder(folder: str, pattern: str = "*.tif",
                                 sample_index: int = 0) -> Tuple[da.Array, Dict]:
    """
    Read a folder of single-plane TIFFs (450 files typical), create a Dask array
    with shape (Z, Y, X, C) or (Z, Y, X) depending on images.
    Returns (dask_array, metadata) where metadata includes dtype, shape, channels info.
    """
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if len(files) == 0:
        raise FileNotFoundError(f"No tiff files found in {folder} matching {pattern}")

    # read a sample to infer shape/dtype
    sample = tifffile.imread(files[sample_index])
    sample = np.asarray(sample)
    # normalize sample dims -> (Y, X) or (Y, X, C)
    if sample.ndim == 2:
        height, width = sample.shape
        channels = 1
    elif sample.ndim == 3:
        # Could be (Z, Y, X) if multi-page, but user said single-plane slides
        # assume (Y, X, C) if len==3 and last dim <= 4
        if sample.shape[0] <= 4 and len(files) == 1:
            # one file with channels first? fallback
            height, width, channels = sample.shape
        else:
            height, width = sample.shape[:2]
            channels = sample.shape[2] if sample.ndim == 3 else 1
    else:
        raise ValueError("Unexpected sample tiff dimensions: %s" % (sample.shape,))

    dtype = sample.dtype
    z = len(files)

    # create delayed readers for each file
    delayed_reads = [read_tiff_delayed(p) for p in files]

    # wrap each into dask array chunk -> assume each file is a single slice (Y,X[,C])
    # We'll create array of shape (z, y, x, c) if channels>1 else (z, y, x)
    sample_arr = sample
    if channels == 1 and sample_arr.ndim == 3 and sample_arr.shape[2] == 1:
        sample_arr = sample_arr[..., 0]

    # create a single-chunk dask array per file and stack them
    da_slices = []
    for d in delayed_reads:
        # build from_delayed with shape of sample slice
        if channels == 1:
            shp = (height, width)
        else:
            shp = (height, width, channels)
        arr = da.from_delayed(d, shape=shp, dtype=dtype)
        da_slices.append(arr)

    stacked = da.stack(da_slices, axis=0)  # shape (Z, Y, X) or (Z, Y, X, C)

    metadata = dict(shape=stacked.shape, dtype=str(dtype), channels=channels)
    return stacked, metadata


def compute_optimal_chunks(shape: Tuple[int, ...],
                           dtype: np.dtype,
                           channels: int,
                           total_mem_bytes: int,
                           n_workers: int,
                           target_chunk_bytes: Optional[int] = None,
                           max_chunk_bytes: int = 256 * 1024**2,
                           xy_align: int = 256,
                           tile_boundary: Optional[int] = None,
                           prefer_z_chunks: int = 1) -> Tuple[int, ...]:
    """
    Heuristic to compute chunk sizes.

    shape: global array shape (Z, Y, X[, C])
    dtype: numpy dtype
    channels: number of channels
    total_mem_bytes: usable memory in bytes (system-level or per-worker? pass per-worker*something)
    n_workers: number of dask workers
    target_chunk_bytes: desired bytes per chunk (optional)
    max_chunk_bytes: hard cap on chunk size in bytes
    xy_align: make X,Y divisible by this (e.g., 256 for cloud)
    tile_boundary: if tif tile grid is known (e.g., 512), align XY to tile boundary
    prefer_z_chunks: number of z slices per chunk (try to keep small)
    """
    itemsize = np.dtype(dtype).itemsize
    # default target: smaller of max_chunk_bytes and per-worker_mem/4
    if target_chunk_bytes is None:
        per_worker_mem = max(256 * 1024**2, int(total_mem_bytes / max(1, n_workers)))
        t = max(16 * 1024**2, int(per_worker_mem / 4))
        target_chunk_bytes = min(t, max_chunk_bytes)

    # shape unpack
    # Handle shapes: (Z, Y, X) or (Z, Y, X, C)
    Z = shape[0]
    Y = shape[1]
    X = shape[2]
    # channels param used separately

    # Start with Z-chunk small (prefer 1 or a few slices)
    z_chunk = min(max(1, prefer_z_chunks), Z)

    # compute XY area that results in target bytes:
    # bytes_per_xy = target_chunk_bytes / (z_chunk * itemsize * channels)
    bytes_per_xy = max(1, int(target_chunk_bytes / (z_chunk * itemsize * max(1, channels))))
    # aim for square-ish XY chunk:
    approx_side = int(math.sqrt(bytes_per_xy))
    # ensure at least 16 px
    approx_side = max(16, approx_side)

    # align to xy_align and optionally tile_boundary
    def align(n, align_to):
        return int(math.ceil(n / align_to) * align_to)

    y_chunk = align(min(Y, approx_side), xy_align)
    x_chunk = align(min(X, approx_side), xy_align)

    # if aligning pushed bytes too large, reduce z_chunk or xy sizes
    def chunk_bytes(zc, yc, xc):
        return zc * yc * xc * itemsize * max(1, channels)

    # iterative adjust if chunk too big
    current_bytes = chunk_bytes(z_chunk, y_chunk, x_chunk)
    while current_bytes > target_chunk_bytes and (y_chunk > xy_align or x_chunk > xy_align or z_chunk > 1):
        # prefer shrinking XY then Z
        if y_chunk > xy_align:
            y_chunk = max(xy_align, int(y_chunk / 2))
            y_chunk = align(y_chunk, xy_align)
        elif x_chunk > xy_align:
            x_chunk = max(xy_align, int(x_chunk / 2))
            x_chunk = align(x_chunk, xy_align)
        elif z_chunk > 1:
            z_chunk = max(1, int(z_chunk / 2))
        else:
            break
        current_bytes = chunk_bytes(z_chunk, y_chunk, x_chunk)

    # ensure divisibility of X,Y by 2 (not strictly necessary)
    y_chunk = min(Y, y_chunk)
    x_chunk = min(X, x_chunk)

    # final fallback: if resulting chunk too small, increase a bit
    if current_bytes < (4 * 1024**2):  # < 4MB - inefficient
        # increase XY until reasonable or hit image size
        while current_bytes < (16 * 1024**2) and (y_chunk * 2 <= Y or x_chunk * 2 <= X):
            if y_chunk * 2 <= Y:
                y_chunk = align(min(Y, y_chunk * 2), xy_align)
            if x_chunk * 2 <= X:
                x_chunk = align(min(X, x_chunk * 2), xy_align)
            current_bytes = chunk_bytes(z_chunk, y_chunk, x_chunk)

    # Return a chunk tuple depending on channels presence
    if channels > 1:
        # shape: (Z, Y, X, C) -> keep channel axis small (full channel)
        return (z_chunk, y_chunk, x_chunk, channels)
    else:
        return (z_chunk, y_chunk, x_chunk)


def make_multiscale(darr: da.Array,
                    out_path: str,
                    base_level: int = 0,
                    downscale_factors: List[int] = [2, 2, 2, 2],
                    compressor=None,
                    chunks=None,
                    overwrite: bool = False,
                    ngff_version: str = "0.4") -> None:
    """
    Write a multiscale pyramid to Zarr following NGFF-ish structure.
    downscale_factors: list of ints describing successive downsampling factors (e.g., [2,2,2] -> 2x, 4x, 8x).
    This implementation downsamples in XY only (not Z) which is typical for microscopy pyramids.
    """
    root = zarr.open_group(out_path, mode='w')

    multiscales = []
    # level 0: original array
    # Use to_zarr per level
    current = darr

    # store level 0
    ds_name = "0"
    grp0 = root.require_group(ds_name)
    # write array
    print(f"Writing level 0 to {out_path}/{ds_name} ...")
    grp0 = os.path.join(out_path, ds_name)
    current.to_zarr(out_path, component=ds_name, overwrite=overwrite, compute=True)



    multiscales.append({"version": f"{ngff_version}", "name": "multiscale",
                        "datasets": [{"path": "0"}], "type": "image"})

    # prepare subsequent levels: downsample XY by 2 repeatedly
    level = 1
    downsample = 2
    prev = current
    downscale_factors = []
    for factor in downscale_factors:
        # downsample in XY by factor (only on last two axes if channels exist)
        # determine axes indices for downscale
        shape = prev.shape
        if prev.ndim == 4:
            # (Z, Y, X, C)
            z_axis, y_axis, x_axis, c_axis = 0, 1, 2, 3
        elif prev.ndim == 3:
            # (Z, Y, X)
            z_axis, y_axis, x_axis = 0, 1, 2
        else:
            raise ValueError("Unexpected array ndim for pyramid creation.")

        # define downsampling function (delayed) that uses local mean if available
        if downscale_local_mean is None:
            # fallback: use simple slicing with = either average by block using block_reduce if available
            try:
                from skimage.measure import block_reduce
                _block_reduce = block_reduce
            except Exception:
                _block_reduce = None
        else:
            _block_reduce = downscale_local_mean

        def downsample_chunk(np_chunk, factor):
            # np_chunk shape could be (z, y, x[, c]) or (y, x[, c]) depending
            # We always downsample XY only.
            if _block_reduce is not None:
                if np_chunk.ndim == 4:
                    # apply block_reduce with block_size=(1, factor, factor, 1)
                    return _block_reduce(np_chunk, block_size=(1, factor, factor, 1), func=np.mean)
                elif np_chunk.ndim == 3:
                    return _block_reduce(np_chunk, block_size=(1, factor, factor), func=np.mean)
            # fallback (naive): reshape and mean
            if np_chunk.ndim == 4:
                z, y, x, c = np_chunk.shape
                ny = y // factor
                nx = x // factor
                # crop to divisible
                np_chunk = np_chunk[:, :ny*factor, :nx*factor, :]
                np_chunk = np_chunk.reshape(z, ny, factor, nx, factor, c).mean(axis=(2,4))
                return np_chunk
            elif np_chunk.ndim == 3:
                z, y, x = np_chunk.shape
                ny = y // factor
                nx = x // factor
                np_chunk = np_chunk[:, :ny*factor, :nx*factor]
                np_chunk = np_chunk.reshape(z, ny, factor, nx, factor).mean(axis=(2,4))
                return np_chunk
            else:
                raise ValueError("Unexpected chunk dims in downsample_chunk")

        # map blocks with da.map_blocks
        new = prev.map_blocks(lambda block, f=factor: downsample_chunk(block, f),
                              dtype=prev.dtype,
                              chunks=(prev.chunks[0],) + tuple([max(1, int(c / factor)) for c in prev.chunks[1]]) if hasattr(prev, 'chunks') else None)
        ds_name = str(level)
        print(f"Writing level {level} to {out_path}/{ds_name} ...")
        grp = root.require_group(ds_name)
        new.to_zarr(store=grp, component=None, overwrite=overwrite, compute=True, compressor=compressor, chunks=chunks)
        multiscales[0]["datasets"].append({"path": ds_name})
        prev = new
        level += 1

    # write minimal .zattrs for NGFF
    zattrs = {
        "multiscales": multiscales,
        "omero": None  # placeholder
    }
    root.attrs.put(zattrs)
    print("Wrote root .zattrs with minimal NGFF multiscale info.")


def write_ome_zarr_multiscale(folder_in: str,
                              zarr_out: str,
                              downscale_levels: int = 4,
                              align_xy: int = 256,
                              tile_boundary: Optional[int] = None,
                              force_divisible_by_256: bool = True,
                              overwrite: bool = True):
    """
    High-level function to build dask array from TIFF folder, start cluster, compute chunks,
    and write NGFF/OME-Zarr multiscale with progress reporting.
    """
    # Start client
    total_cpus = os.cpu_count() or 1
    client = start_local_dask_cluster(n_workers=total_cpus, threads_per_worker=1)

    print("Building dask array from folder ...")
    darr, meta = build_dask_array_from_folder(folder_in)

    print(f"Array shape: {darr.shape}, dtype: {meta['dtype']}, channels: {meta['channels']}")

    # compute memory heuristics
    total_usable = compute_available_memory_bytes(fraction=0.9)
    #n_workers = len(client.scheduler_info()['workers'])
    n_workers = 1
    print(f"Detected workers: {n_workers}, usable memory (bytes): {total_usable}")

    # compute chunk proposal
    dtype = np.dtype(meta['dtype'])
    chunk_tuple = compute_optimal_chunks(shape=darr.shape,
                                         dtype=dtype,
                                         channels=meta['channels'],
                                         total_mem_bytes=total_usable,
                                         n_workers=n_workers,
                                         xy_align=(256 if force_divisible_by_256 else 16),
                                         tile_boundary=None,
                                         prefer_z_chunks=1)

    print("Proposed chunk shape:", chunk_tuple)
    # apply rechunk
    print("Rechunking Dask array to proposed chunk sizes (this generates a dask graph)...")
    darr_rechunked = darr.rechunk(chunk_tuple)

    # Build multiscale downsample factors
    ds_factors = [2] * downscale_levels

    # write pyramid to zarr (this will compute)
    print("Starting to write OME-Zarr multiscale ...")
    # Use zarr default compressor if desired (None => default)
    compressor = None

    # Use to_zarr with storage options: we create a directory group
    # We'll write multiscales manually; make_multiscale uses .to_zarr on groups.
    make_multiscale(darr_rechunked, out_path=zarr_out, downscale_factors=ds_factors,
                    compressor=compressor, chunks=chunk_tuple, overwrite=overwrite)

    print("Completed writing multiscale zarr.")
    client.close()
    return zarr_out


# If running as script, provide an example usage:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create optimized NGFF/OME-Zarr from TIFF folder.")
    parser.add_argument("--levels", type=int, default=4, help="Number of downsample levels (default 4).")
    parser.add_argument("--align", type=int, default=256, help="Align XY chunks to this multiple (default 256).")
    parser.add_argument("--force-256", action="store_true", help="Force XY chunks divisible by 256 for cloud optimization.")
    args = parser.parse_args()
    tiff_folder = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK37/preps/C1/full_aligned'
    zarr_path = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK37/www/neuroglancer_data/C1.zarr'

    write_ome_zarr_multiscale(tiff_folder, zarr_path, downscale_levels=args.levels,
                              align_xy=args.align, force_divisible_by_256=True, overwrite=True)
