import os
import cv2
import numpy as np

def create_mask_from_largest_contour(tif_path, min_area_ratio=0.001):
    """
    Create a binary mask from the largest contour in an sRGB TIF image.
    The function assumes the background is mostly white and the border is gray/darker.

    Args:
        tif_path (str): Path to the input TIF file.
        output_mask_path (str, optional): Path to save the binary mask (as .tif or .png).
        min_area_ratio (float): Minimum area ratio (to image area) to filter small noise.
    Returns:
        mask (np.ndarray): Binary mask with 1 inside contour and 0 outside.
    """
    # --- Load the image ---
    img = cv2.imread(tif_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {tif_path}")

    # --- Convert to grayscale ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Normalize and invert (since background is white) ---
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    inv = cv2.bitwise_not(gray)

    # --- Apply Gaussian blur to smooth edges ---
    blur = cv2.GaussianBlur(inv, (5, 5), 0)

    # --- Threshold ---
    # Use Otsu's method to separate foreground (border/object) from background
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Find contours ---
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in image.")

    # --- Select the largest contour (by area) ---
    img_area = img.shape[0] * img.shape[1]
    contours = [c for c in contours if cv2.contourArea(c) > img_area * min_area_ratio]
    if not contours:
        raise ValueError("No large contours found; try lowering min_area_ratio.")

    largest_contour = max(contours, key=cv2.contourArea)

    # --- Create empty mask and draw the largest contour ---
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, color=255, thickness=-1)

    # --- Convert to binary mask (0, 1) ---
    mask = (mask > 0).astype(np.uint8)


    # --- Optional: Smooth mask edges ---
    #mask = cv2.GaussianBlur(mask, (5, 5), 0)
    #mask = (mask > 128).astype(np.uint8) * 255

    # --- Apply mask to original image ---
    masked_rgb = cv2.bitwise_and(img, img, mask=mask)

    # --- Convert masked region to grayscale ---
    gray_masked = cv2.cvtColor(masked_rgb, cv2.COLOR_BGR2GRAY)

    return gray_masked

# Example usage:
if __name__ == "__main__":
    animal = "MD585"
    input_dir = f"/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps/C1/thumbnail_aligned"
    output_dir = f"/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps/C1/thumbnail_masked"
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.tif')])
    for f in files:
        input_tif = os.path.join(input_dir, f)
        output_tif = os.path.join(output_dir, f)
        masked = create_mask_from_largest_contour(input_tif)
        cv2.imwrite(output_tif, masked)
        print(f"Processed and saved mask for {f}")