import os
from pathlib import Path
import cv2
import numpy as np
import tifffile
import argparse
from aicspylibczi import CziFile
from tqdm import tqdm

DOWNSCALE = 32
TIFF_COMPRESSION = "zlib"

def equalize_tiff(img_np):

    # Perform histogram equalization
    # Note: cv2.equalizeHist expects uint8 (8-bit)
    if img_np.dtype != np.uint8:
        img_np = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    equalized = cv2.equalizeHist(img_np)
    return equalized


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def downsample_image(img: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample image using area interpolation.

    AREA interpolation is best for microscopy decimation.
    """

    new_w = max(1, img.shape[1] // factor)
    new_h = max(1, img.shape[0] // factor)

    return cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )


def normalize_axes(arr: np.ndarray) -> np.ndarray:
    """
    Remove singleton dimensions from aicspylibczi output.
    """

    return np.squeeze(arr)


def extract_regions(czi_path: str, output_dir: str, channel:int=0):
    czi = CziFile(czi_path)

    scene_boxes = czi.get_all_scene_bounding_boxes()
    scene_data = []
    for s in range(len(scene_boxes)):
        bbox = scene_boxes[s]
        x = bbox.x
        y = bbox.y
        w = bbox.w
        h = bbox.h


        img = czi.read_mosaic(
            region=(x, y, w, h),
            scale_factor=1.0 / DOWNSCALE,
            C=channel,
        )

        img = normalize_axes(img)

        # -------------------------------------------------
        # Ensure 2D grayscale
        # -------------------------------------------------

        if img.ndim > 2:
            img = np.squeeze(img)

        img = equalize_tiff(img)

        # -------------------------------------------------
        # Save TIFF
        # -------------------------------------------------
        scene_data.append((s, img, x//DOWNSCALE, y//DOWNSCALE))
    # Normalize coordinates
    min_x = min(d[2] for d in scene_data)
    min_y = min(d[3] for d in scene_data)

    aligned = []
    for s, img, x, y in scene_data:
        aligned.append((s, img, int(x - min_x), int(y - min_y)))

    # Determine canvas size
    #max_x = max(x + img.shape[-1] for _, img, x, _ in aligned)
    #max_y = max(y + img.shape[-2] for _, img, _, y in aligned)
    max_x = max(0 + img.shape[-1] for _, img, x, _ in aligned)
    max_y = max(0 + img.shape[-2] for _, img, _, y in aligned)
    print(f"Canvas size: {max_x} x {max_y}")
    

    for s, img, x, y in aligned:
        canvas = np.zeros((max_y, max_x), dtype=img.dtype)
        h, w = img.shape[-2:]
        #canvas[y:y+h, x:x+w] = img
        canvas[0:y+h, 0:x+w] = img
        outfile = os.path.join(output_dir, f"{Path(czi_path).stem}_scene_{s:03d}_channel_{channel:03d}_scale_1_{DOWNSCALE}.tif")
        tifffile.imwrite(
            outfile,
            canvas,
            compression=TIFF_COMPRESSION,
            bigtiff=True,
        )


###################################


def extract_czi_scene_to_tiff(
    czi_path,
    output_dir,
    scene_region,
    channel_index=0,
    scale=1.0,
    compression="zlib",
    tile_size=(512, 512),
):

    czi_path = Path(czi_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    czi = CziFile(str(czi_path))

    # Read mosaic for one scene + one channel
    # This automatically stitches tiles into the correct mosaic arrangement
    mosaic = czi.read_mosaic(
        C=channel_index,
        region=scene_region,
        scale_factor=1.0/DOWNSCALE
    )

    # Remove singleton dimensions
    mosaic = np.squeeze(mosaic)


    # Ensure contiguous memory
    mosaic = np.ascontiguousarray(mosaic)
    mosaic = equalize_tiff(mosaic)

    output_name = (
        f"{czi_path.stem}"
        f"_ch{channel_index:02d}"
        f"_scale{scale:.2f}.tif"
    )

    output_path = output_dir / output_name


    tifffile.imwrite(
        output_path,
        mosaic,
        bigtiff=True,
        compression=compression,
        tile=tile_size,
        photometric="minisblack",
    )



def batch_extract_czi_scenes(
    input_dir,
    output_dir,
    channel_index=0,
    scale=1.0,
):
    """
    Batch extract all scenes from all CZI files.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    czi_files = sorted(input_dir.glob("*.czi"))


    for czi_file in tqdm(czi_files, desc="Processing CZI files"):


        czi = CziFile(str(czi_file))
        scene_boxes = czi.get_all_scene_bounding_boxes()
        for s in range(len(scene_boxes)):
            bbox = scene_boxes[s]
            x = bbox.x
            y = bbox.y
            w = bbox.w
            h = bbox.h

        for scene_index in range(len(scene_boxes)):

            extract_czi_scene_to_tiff(
                czi_path=czi_file,
                output_dir=output_dir,
                scene_region=(x, y, w, h),
                channel_index=channel_index,
                scale=scale,
            )



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract TIFFs from CZI files")
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)
    parser.add_argument("--scene_index", type=int, default=None, help="Scene index to extract (default: all)")
    parser.add_argument("--channel_index", type=int, default=0, help="Channel index to extract (default: 0)")
    parser.add_argument("--scale_level", type=int, default=32, help="Pyramid level (default: 32)")
    parser.add_argument("--align_scenes", action='store_true', help="Align scenes using stage coordinates")

    args = parser.parse_args()

    animal = args.animal

    base_dir = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{args.animal}'
    input_dir = os.path.join(base_dir, 'czi')
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    output_dir = os.path.join(base_dir, 'preps', 'extracted_tiffs')
    os.makedirs(output_dir, exist_ok=True)

    batch_extract_czi_scenes(
        input_dir=input_dir,
        output_dir=output_dir,
        channel_index=0,
        scale=1.0/DOWNSCALE,
    )

    exit(0)



    czi_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".czi")])

    for czi_name in tqdm(czi_files[0:1], desc="Processing CZI files"):
        czi_path = os.path.join(input_dir, czi_name)
        extract_regions(czi_path, output_dir)
