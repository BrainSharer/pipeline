import argparse

import numpy as np

def compute_mipmaps(base_resolution, max_voxel_size=512.0):
    num_mips=100
    base_resolution = np.array(base_resolution, dtype=float)
    base_chunk_size=(64, 64, 64)
    scales = [(2,2,1)]

    current_res = base_resolution.copy()
    chunks = []

    for mip in range(1, num_mips):
        # --- Continuous anisotropy correction ---
        # Normalize by smallest voxel dimension
        min_res = np.min(current_res)
        ratios = current_res / min_res

        # Smooth scaling: more aggressive for finer axes
        # Use inverse ratio to bias scaling
        inv_ratios = 1.0 / ratios

        # Normalize to [1, 2] range
        scale = 1.0 + inv_ratios
        scale = np.clip(scale, 1.0, 2.0)

        # Round to nearest integer (Neuroglancer prefers ints)
        scale = np.round(scale).astype(int)

        # Ensure at least 1x scaling
        scale = np.maximum(scale, 1)

        # --- Apply scaling ---
        new_res = current_res * scale

        # --- Stop if exceeding max voxel size ---
        if np.any(new_res > max_voxel_size):
            print(f"Stopping at mip {mip}: exceeded max voxel size")
            break

        # --- Compute chunk size (world-space balanced) ---
        # Keep chunk size ~constant in microns        
        world_chunk = np.array(base_chunk_size) * base_resolution
        chunk = np.round(world_chunk / new_res).astype(int)
        # Clamp chunk sizes to reasonable bounds
        chunk = np.clip(chunk, 16, 256)
        chunks.append(chunk.tolist())

        # Store results
        scales.append(scale)
        #resolutions.append(new_res.tolist())

        current_res = new_res

    scales = [tuple(int(x) for x in s) for s in scales]
    resolutions = []
    x,y,z = base_resolution
    for mip, scale in enumerate(zip(scales)):
        if mip == 0:
            x = x * scales[mip][0]
            y = y * scales[mip][1]
            z = z * scales[mip][2]
            resolution = [x, y, z]
        else:
            x = resolutions[mip-1][0] * scales[mip][0]
            y = resolutions[mip-1][1] * scales[mip][1]
            z = resolutions[mip-1][2] * scales[mip][2]

        resolution = [float(x), float(y), float(z)]
        resolutions.append(resolution)

    return scales, resolutions, chunks



def print_pyramid(resolutions, scales, chunks):
    print(f"{'Mip':<4} {'Resolution (µm)':<30} {'Scale':<30} {'Chunk'}")
    print("-" * 90)
    for i, (r, s, c) in enumerate(zip(resolutions, scales, chunks)):
        print(f"{i:<4} {str(r):<30} {str(s):<30} {c}")

if __name__ == "__main__":
    # Example usage:
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--xy", help="xy resolution", required=True, type=float)
    parser.add_argument("--z", help="z resolution", required=True, type=float)

    args = parser.parse_args()
    x = args.xy
    y = args.xy
    z = args.z

    resolution = [x, y, z]  # microns

    #scales, resolutions, chunks = compute_mipmaps(resolution)
    #print(f'resolution: {resolutions}')
    #print(f'scales: {scales}')
    #print(f'chunks: {chunks}')

    #for mip, (scale, resolution, chunk) in enumerate(zip(scales, resolutions, chunks)):
    #    print(f'{mip=} Scale factor: {scale} Resolution: {resolution} Chunk: {chunk}')

    resolutions, scales, chunks = compute_neuroglancer_pyramid(
        resolution,
        n_mips=8,
        max_voxel_size=512,
        base_chunk=(64, 64, 64),
        anisotropy_alpha=0.5
    )

    print_pyramid(resolutions, scales, chunks)