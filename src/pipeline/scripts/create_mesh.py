import json
import shutil
import os
import glob
import numpy as np
import tifffile as tiff

from cloudvolume import CloudVolume
import igneous.task_creation as tc

from taskqueue.taskqueue import LocalTaskQueue


############################################
# CONFIG
############################################

TIFF_DIR = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/X/preps/C1/downsampled_10"
OUT_VOL = "file:///net/birdstore/Active_Atlas_Data/data_root/pipeline_data/X/www/neuroglancer_data/C1"

if os.path.exists(OUT_VOL.replace("file://", "")):
    print(f"Removing and creating output directory: {OUT_VOL}")
    shutil.rmtree(OUT_VOL.replace("file://", ""))
os.makedirs(OUT_VOL.replace("file://", ""))

um2nanom = 1000.0
VOXEL_SIZE = (int(10 * um2nanom), int(10 * um2nanom), int(10 * um2nanom))  # in nanometers
CHUNK_SIZE = (64, 64, 64)
DOWNSAMPLE_LEVELS = 3

############################################
# LOAD TIFF STACK (LAZY)
############################################
print("Creating output at ", OUT_VOL)

############################################
# CREATE CLOUDVOLUME
############################################


# -----------------------------
# USER CONFIG
# -----------------------------
NUM_MIPS = 5               # resolution pyramid depth
SEG_DTYPE = np.uint32

# -----------------------------
# CREATE NEUROGLANCER VOLUME
# -----------------------------
def create_volume():

    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type="segmentation",  # 'image' or 'segmentation'
        data_type=SEG_DTYPE,  #
        encoding='raw',  # other options: 'jpeg', 'compressed_segmentation' (req. uint32 or uint64)
        resolution=VOXEL_SIZE,  # Size of X,Y,Z pixels in nanometers,
        voxel_offset = [0,0,0],  # values X,Y,Z values in voxels
        chunk_size = CHUNK_SIZE,  # rechunk of image X,Y,Z in voxels
        volume_size = [1037,789,1332],  # X,Y,Z size in voxels
    )
    vol = CloudVolume(OUT_VOL, info=info, compress=False, progress=True)
    vol.commit_info()
    return vol



    vol = CloudVolume(
        OUT_VOL,
        info={
            "layer_type": "segmentation",
            "data_type": "uint32",
            "num_channels": 1,
            "scales": [{
                "encoding": "raw",
                "chunk_sizes": [list(CHUNK_SIZE)],
                "resolution": [r * 1000 for r in VOXEL_SIZE_UM],  # nm
                "voxel_offset": [0, 0, 0],
                "volume_size": [1037,789,1332],
            }],
        },
        compress=False,
        progress=True
    )
    vol.commit_info()
    return vol


# -----------------------------
# STREAM TIFF STACK INTO VOLUME
# -----------------------------
def ingest_tiffs(vol):
    files = sorted(glob.glob(os.path.join(TIFF_DIR, "*.tif")))
    assert len(files) == vol.info['scales'][0]['size'][2], "Number of TIFF files does not match volume Z size"

    z = 0
    block = []

    for f in files:
        img = tiff.imread(f).astype(SEG_DTYPE)
        block.append(img)

        if len(block) == CHUNK_SIZE[2]:
            block_np = np.stack(block, axis=0)
            vol[:, :, z:z+len(block), 0] = block_np.transpose(2, 1, 0)
            z += len(block)
            block.clear()

    if block:
        block_np = np.stack(block, axis=0)
        vol[:, :, z:z+len(block), 0] = block_np.transpose(2, 1, 0)


# -----------------------------
# DOWNSAMPLE SEGMENTATION PYRAMID
# -----------------------------
def build_mips():
    tq = LocalTaskQueue(parallel=8)

    tasks = tc.create_downsampling_tasks(
        layer_path=OUT_VOL,
        num_mips=NUM_MIPS
    )

    tq.insert(tasks)
    tq.execute()


# -----------------------------
# BUILD MULTI-RES MESH PYRAMID
# -----------------------------
def build_meshes():
    tq = LocalTaskQueue(parallel=8)

    tasks = tc.create_meshing_tasks(
        layer_path=OUT_VOL,
        mip=0,
        max_simplification_error=40,  # microns, tune for vessels
        mesh_dir="meshes"
    )
    tq.insert(tasks)
    tq.execute()

    tasks = tc.create_mesh_manifest_tasks(OUT_VOL, mesh_dir="meshes") # Second Pass
    tq.insert(tasks)    
    tq.execute()

    tasks = tc.create_unsharded_multires_mesh_tasks(OUT_VOL, num_lod=3, mesh_dir="meshes")
    tq.insert(tasks)    
    tq.execute()

    cloud_volume = CloudVolume(OUT_VOL, 0)
    cloud_volume.info['segment_properties'] = 'names'
    cloud_volume.commit_info()
    segment_properties = {str(id): str(id) for id in [0,1]}


    segment_properties_path = os.path.join(cloud_volume.layerpath.replace('file://', ''), 'names')
    os.makedirs(segment_properties_path, exist_ok=True)
    info = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": [str(number) for number, _ in segment_properties.items()],
            "properties": [{
                "id": "label",
                "type": "label",
                "values": [str(label) for _, label in segment_properties.items()]
            }]
        }
    }
    with open(os.path.join(segment_properties_path, 'info'), 'w') as file:
        json.dump(info, file, indent=2)





# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    vol = create_volume()
    ingest_tiffs(vol)
    build_mips()
    build_meshes()

    print("✔ Neuroglancer multi-resolution segmentation mesh complete")
