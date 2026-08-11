import os
import cv2
import numpy as np
from skimage import exposure
from tifffile import imread, imwrite

def equalize_without_bright_spots(tif_path, output_path=None, clip_percentile=89.5):
    """
    Perform histogram equalization while suppressing very bright regions.
    
    Parameters
    ----------
    tif_path : str
        Path to the input TIF image.
    output_path : str, optional
        Path to save the processed image (if None, not saved).
    clip_percentile : float
        Upper percentile used to clip high intensities (default: 99.5%).
    """
    
    # Read image
    img = imread(tif_path).astype(np.float32)
    
    # Handle multichannel
    if img.ndim == 3 and img.shape[-1] in (3, 4):  # RGB or RGBA
        img_gray = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    
    # Compute intensity clipping threshold
    upper_clip = np.percentile(img_gray, clip_percentile)
    
    # Clip bright regions (flatten outliers)
    img_clipped = np.clip(img_gray, 0, upper_clip)
    
    # Normalize to [0,1] for stability
    img_clipped /= upper_clip
    
    # Apply adaptive histogram equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply((img_clipped * 255).astype(np.uint8))
    
    # Optionally save
    if output_path:
        imwrite(output_path, img_eq.astype(np.uint8))
    
    return img_eq


# Example usage:
if __name__ == "__main__":
    data_path = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/CTB015/preps/C2/thumbnail_aligned"
    infile = os.path.join(data_path, "000.tif")
    outfile = os.path.join(data_path, "000_equalized.tif")
    eq = equalize_without_bright_spots(infile, outfile)
