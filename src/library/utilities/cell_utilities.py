import os, sys
import cv2
import imageio
import numpy as np
import subprocess
from pathlib import Path
from library.controller.sql_controller import SqlController
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from library.utilities.utilities_process import get_cpus


def load_image(file: str):
    if os.path.exists(file):
        return imageio.imread(file)
    else:
        print(f'ERROR: {file} not found')
        sys.exit(1)


def subtract_blurred_image(image: np.ndarray, gaussian_blur_standard_deviation_sigmaX: int, kernel_size_pixels: tuple, make_smaller: bool = True, debug: bool = False):
    '''PART OF STEP 2. Identify cell candidates: average the image by subtracting gaussian blurred mean
    Gaussian blur applied to smaller image using Gaussian kernel of 41x41 pixels and standard deviation (s.d.) of the Gaussian distribution is 10
    Larger kernel size produces stronger blur effect
    Higher s.d. leads to more blurring

    Args:
        image: Input image (converted to float32 if not already)
        gaussian_blur_standard_deviation_sigmaX: Standard deviation for Gaussian blur
        kernel_size_pixels: Kernel size for Gaussian blur
        make_smaller: If True, process at lower resolution for speed/memory
        debug: Print debug information
        scale_factor: Downscaling factor when make_smaller is True (default 0.1)
    '''
    scale_factor = 0.1

    if image.dtype != np.float32:
        image = image.astype(np.float32, copy=False)
    
    if make_smaller:
        # Calculate target size once instead of using resize with fx/fy
        h, w = image.shape[:2]
        small_size = (max(1, int(w * scale_factor)), max(1, int(h * scale_factor))) # Ensure size is at least 1x1 to avoid errors

        # Resize and blur
        small = cv2.resize(image, small_size, interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(small, ksize=kernel_size_pixels, 
                                 sigmaX=gaussian_blur_standard_deviation_sigmaX)
        
        # Resize back using linear interpolation (faster than INTER_AREA for upscaling)
        relarge = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
        
        difference = image - relarge
    else:
        # Apply blur directly and subtract in-place
        blurred = cv2.GaussianBlur(image, ksize=kernel_size_pixels, 
                                 sigmaX=gaussian_blur_standard_deviation_sigmaX)
        
        difference = image - blurred
        
    #3-JUL-2025 Song-Mao change
    difference = np.clip(difference, 0, None)  # Ensure no negative

    if debug:
        print(f'DEBUG: -subtract_blurred_image- detail:')
        print(f"DEBUG: Original image shape: {image.shape}")
        print(f'DEBUG: Gaussian blur detail:')
        print(f"DEBUG: {kernel_size_pixels=}")
        print(f"DEBUG: {gaussian_blur_standard_deviation_sigmaX=}")
    
    return difference


def find_connected_segments(image, segmentation_threshold) -> tuple:
    '''PART OF STEP 2. Identify cell candidates'''

    #CREATE BINARY MASK FROM PIXEL INTENSITY > THRESHOLD (segment_location is centroid of each segment)
    n_segments, segment_masks, segment_stats, segment_location = cv2.connectedComponentsWithStats(np.int8(image > segmentation_threshold))

    if segment_location.size > 0:
        # Replace NaN/inf with 0 and ensure finite values
        segment_location = np.nan_to_num(segment_location, nan=0, posinf=0, neginf=0)
        
        # Clip to integer range to prevent overflow
        segment_location = np.clip(segment_location, 
                                np.iinfo(np.int32).min, 
                                np.iinfo(np.int32).max)
        
        # Convert to int32 safely
        segment_location = segment_location.astype(np.int32)
    
    # Flip coordinates (y,x) -> (x,y) if array is not empty
    if segment_location.size > 0:
        segment_location = np.flip(segment_location, 1)

    return (n_segments, segment_masks, segment_stats, segment_location)


def filter_cell_candidates(
    animal: str,
    section_number: int,
    connected_segments,
    segment_size_min,
    segment_size_max,
    cell_radius,
    x_window,
    y_window,
    absolute_coordinates,
    difference_ch1,
    difference_ch3,
    task: str = None,
    pruning_info: dict = None,
    super_annotation_dict: dict = None,
    debug: bool = False
):
    '''PART OF STEP 2. Identify cell candidates:  Area is for the object, where pixel values are not zero,
    Segments are filtered to remove those that are too large or too small
    
    NOTE 1: May have additional pruning filters if argument 'self.run_pruning' == True:
    'prune_x_range' = The [min, max] x-axis coordinate (in absolute units)
    'prune_y_range' = The [min, max] y-axis coordinate (in absolute units)
    'prune_area_min' = The minimum area (pixel^2)
    'prune_area_max' = The maximum area (pixel^2)
    'prune_annotation_ids' = volume id(s) from brainsharer (filters if point is located inside volume)
    'prune_combine_method' = Method to combine overlapping segments (e.g., 'union', 'intersection')
    '''
  
    n_segments, segment_masks, segment_stats, segment_location = connected_segments
    label_of_interest = []
    cell_candidates = []
    img_counterstain = []

    for segmenti in range(n_segments):
        _, _, width, height, object_area = segment_stats[segmenti, :]

        if object_area > segment_size_max or object_area < segment_size_min:
            continue

        segment_row, segment_col = segment_location[segmenti, :]
        row_start = int(segment_row - cell_radius)
        col_start = int(segment_col - cell_radius)
        if row_start < 0 or col_start < 0:
            continue

        row_end = int(segment_row + cell_radius)
        col_end = int(segment_col + cell_radius)
        if row_end > x_window or col_end > y_window:  # row evaluates with x-axis (width), col evaluates with y-axis (height)
            continue

        segment_mask = (segment_masks[row_start:row_end, col_start:col_end] == segmenti)
        label_of_interest = difference_ch3[row_start:row_end, col_start:col_end].T
        absolute_coordinates_YX = (absolute_coordinates[2] + segment_col,
                                 absolute_coordinates[0] + segment_row)

        if pruning_info and pruning_info.get('run_pruning'):
            try:
                prune_x = pruning_info.get("prune_x_range")
                if prune_x and not (prune_x[0] <= segment_col <= prune_x[1]):
                    continue
            
                prune_y = pruning_info.get("prune_y_range")
                if prune_y and not (prune_y[0] <= segment_row <= prune_y[1]):
                    continue
            
                if (object_area < pruning_info.get("prune_amin", 0) or 
                    object_area > pruning_info.get("prune_amax", float('inf'))):
                    continue
                
                annotation_ids = pruning_info.get("prune_annotation_ids")
                if annotation_ids:
                    result_in_vol = point_in_volume_pruning(animal, section_number, absolute_coordinates_YX, annotation_ids, super_annotation_dict, pruning_info.get("prune_combine_method")) #returns bool

                    if not result_in_vol:
                        continue
                
            except Exception as e:
                print(f"Error occurred while applying pruning filters: {e}")

        #FINAL SANITY CHECK [IF NOT 'segment' TASK]
        if task != 'segment':
            img_counterstain = difference_ch1[row_start:row_end, col_start:col_end].T
            if img_counterstain.shape != label_of_interest.shape or img_counterstain.shape != segment_mask.shape:
                if debug:
                    print(f"ERROR: Image shapes do not match. Skipping this segment.")
                    print(f'img_counterstain: {img_counterstain.shape}')
                    print(f'img_label_of_interest: {label_of_interest.shape}')
                    print(f'segment_mask: {segment_mask.shape}')
                continue

        cell = {
            "animal": animal,
            "section": section_number,
            "area": object_area,
            "absolute_coordinates_YX": absolute_coordinates_YX,
            "cell_shape_XY": (height, width),
            "image_CH3": label_of_interest,
            "image_CH1": img_counterstain,
            "mask": segment_mask.T,
        }                                        
        cell_candidates.append(cell)
    return cell_candidates


def equalize_array_size_by_trimming(array1, array2):
    '''PART OF STEP 3. CALCULATE CELL FEATURES; array1 and array 2 the same size
    Note: CPU
    '''
    size0 = min(array1.shape[0], array2.shape[0])
    size1 = min(array1.shape[1], array2.shape[1])
    array1 = trim_array_to_size(array1, size0, size1)
    array2 = trim_array_to_size(array2, size0, size1)
    return array1, array2    


def trim_array_to_size(arr, size0, size2):
    '''PART OF STEP 3. CALCULATE CELL FEATURES
    Note: CPU
    '''
    if(arr.shape[0] > size0):
        size_difference = int((arr.shape[0]-size0)/2)
        arr = arr[size_difference:size_difference+size0, :]
    if(arr.shape[1] > size2):
        size_difference = int((arr.shape[1]-size2)/2)
        arr = arr[:, size_difference:size_difference+size2]
    return arr

def calc_moments_of_mask(mask):   
    '''
    calculate moments (how many) and Hu Moments (7)
    Moments(
            double m00,
            double m10,
            double m01,
            double m20,
            double m11,
            double m02,
            double m30,
            double m21,
            double m12,
            double m03
            );
    Hu Moments are described in this paper: 
    https://www.researchgate.net/publication/224146066_Analysis_of_Hu's_moment_invariants_on_image_scaling_and_rotation

    NOTE: image moments (weighted average of pixel intensities) are used to calculate centroid of arbritary shapes in opencv library
    '''
    mask = mask.astype(np.float32)
    moments = cv2.moments(mask)

    huMoments = cv2.HuMoments(moments)
    moments = {key + "_mask": value for key, value in moments.items()} #append_string_to_every_key
    return (moments, {'h%d'%i+f'_mask':huMoments[i,0]  for i in range(7)}) #return first 7 Hu moments e.g. h1_mask


def features_using_center_connected_components(cell_candidate_data: dict, debug: bool = False):
    '''Part of step 3. calculate cell features'''
    def mask_mean(mask, image): #ORG CODE FROM Kui github (FeatureFinder)
        mean_in = np.mean(image[mask == 1])
        mean_all = np.mean(image.flatten())
        
        numerator = mean_in - mean_all
        denominator = mean_in + mean_all
        if numerator == 0 and denominator == 0:
            return 1
        else:
            return numerator / denominator 

    image1 = cell_candidate_data['image_CH1']
    image3 = cell_candidate_data['image_CH3']
    mask = cell_candidate_data['mask']

    if debug:
        if mask.shape != cell_candidate_data['image_CH1'].shape or mask.shape != cell_candidate_data['image_CH3'].shape:
            print(f"ERROR: mask shape does not match image shape")
            print(f"mask shape: {mask.shape}")
            print(f"image_CH1 shape: {cell_candidate_data['image_CH1'].shape}")
            print(f"image_CH3 shape: {cell_candidate_data['image_CH3'].shape}")
            sys.exit(1)

    # Add input validation
    if mask.max() == 0:
        return 0.0, 0.0, calc_moments_of_mask(mask)
    moments_data = calc_moments_of_mask(mask)

    ch1_contrast = mask_mean(mask, image1)
    ch3_contrast = mask_mean(mask, image3)
    
    return ch1_contrast, ch3_contrast, moments_data


def find_available_backup_filename(file_path):
    """
    Recursively search for available backup filename.
    
    Args:
        file_path (str): Path to the original file.
    
    Returns:
        str: Path to the available backup file.
    """
    # Initialize the backup extension counter
    i = 1
    
    # Loop until we find an available filename
    while True:
        # Construct the backup filename
        backup_filename = f"{file_path}.bak.{i}"
        
        # Check if the backup filename is available
        if not os.path.exists(backup_filename):
            return backup_filename
        
        # Increment the backup extension counter
        i += 1


def calculate_correlation_and_energy(avg_cell_img, cell_candidate_img):
    '''
    Calculates correlation and energy features between cell images

    part of step 3. 
    calculate cell features; calculate correlation [between cell_candidate_img 
    and avg_cell_img] and and energy for cell canididate
    NOTE: avg_cell_img and cell_candidate_img contain respective channels prior to passing in arguments
    '''

    cell_candidate_img, avg_cell_img = equalize_array_size_by_trimming(
        cell_candidate_img, 
        avg_cell_img
    )

    avg_x, avg_y = sobel(avg_cell_img)
    candidate_x, candidate_y = sobel(cell_candidate_img)
    dot_prod = (avg_x * candidate_x) + (avg_y * candidate_y)
    corr = np.mean(dot_prod)
    mag = np.sqrt(candidate_x**2 + candidate_y**2)
    energy = np.mean(mag * avg_cell_img)
    
    return corr, energy


def sobel(img):
    '''PART OF STEP 3. CALCULATE CELL FEATURES; Compute the normalized sobel edge magnitudes'''

    if img is None or img.size == 0:
        raise ValueError("Error: The input image is empty or not loaded properly.")

    #DEFINE SOBEL KERNELS
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    
    #NORMALIZATION
    _mean = (np.mean(sobel_x) + np.mean(sobel_y))/2.
    _std = np.sqrt((np.var(sobel_x) + np.var(sobel_y))/2)

    eps = 1e-8 #If _std is zero or contains NaN/Inf values, the normalization step will fail. 
    sobel_x = (sobel_x - _mean) / (_std + eps)
    sobel_y = (sobel_y - _mean) / (_std + eps)
    return sobel_x, sobel_y


def copy_with_rclone(src_dir: str, dest_dir: str) -> None:
    """
    Copies all files from src_dir to dest_dir using rclone.
    
    Args:
        src_dir (str): Source directory path.
        dest_dir (str): Destination directory path.
    """
    # Ensure paths are absolute and properly formatted
    src_dir = Path(src_dir).resolve()
    dest_dir = Path(dest_dir).resolve()

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)

    num_cores = get_cpus()
    transfers = min(num_cores[0], max(num_cores[1], 1))

    # Construct the rclone command
    rclone_cmd = [
        "rclone",
        "copy",
        "--progress",  # Show progress (optional)
        "--transfers", str(transfers), #num of parallel transfers
        str(src_dir) + "/",  # Ensure trailing slash for directory copy
        str(dest_dir) + "/",
    ]

    try:
        print(f"Copying files from {src_dir} to {dest_dir} using rclone...")
        subprocess.run(rclone_cmd, check=True)
        print("Copy completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during rclone copy: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def point_in_volume_pruning(animal: str, section_number: int, point_YX: tuple[int, int], prune_annotation_ids: list[int] | int | None = None, super_annotation_dict: dict | None = None, prune_combine_method: str = "union") -> bool:
    """
    Check if a point is within one or more volume annotations, combining results according to specified method.
    
    Args:
        animal: Animal ID
        point: (y, x) coordinates to check
        prune_annotation_ids: Single ID or list of annotation IDs to check against
        prune_combine_method: How to combine results ('union' or 'intersection')
        
    Returns:
        bool: True if point meets the inclusion criteria based on combine method
    """
    
    if not prune_annotation_ids:
        return False
    
    print(super_annotation_dict)
    return True

    
        
    #     polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    #     for section, points in vol_polygons.items():
    #         for x, y in points:
    #             if section == section_number:
    #                 if (y, x) == point_YX:
    #                     return True

    #     if not vol_polygons:
    #         print(f"Warning: No volume polygons found for annotation ID {annotation_id} in {animal}")
    #         if prune_combine_method == "intersection":
    #             return False  # Early exit for intersection mode
    #         continue
        
    #     # Check if point is in any of the polygons for this annotation
    #     in_volume = any(polygon.contains_point(point) for polygon in vol_polygons)
    #     results.append(in_volume)
        
    #     # Early exit optimization for union mode
    #     if prune_combine_method == "union" and in_volume:
    #         return True
    
    # # Determine final result based on combine method
    # if prune_combine_method == "union":
    #     return any(results)
    # elif prune_combine_method == "intersection":
    #     return all(results)
    # else:
    #     raise ValueError(f"Unknown combine method: {prune_combine_method}. Use 'union' or 'intersection'")
    

    
     