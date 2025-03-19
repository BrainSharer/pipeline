import os
import sys
import cv2
import imageio
import numpy as np


def load_image(file: str):
    if os.path.exists(file):
        return imageio.imread(file)
    else:
        print(f'ERROR: {file} not found')
        sys.exit(1)


def subtract_blurred_image(image, make_smaller=True):
    '''PART OF STEP 2. Identify cell candidates: average the image by subtracting gaussian blurred mean'''
    image = np.float32(image)
    if make_smaller:
        small = cv2.resize(image, (0, 0), fx=0.05, fy=0.05, interpolation=cv2.INTER_AREA)
    else:
        small = image.copy()
    blurred = cv2.GaussianBlur(small, ksize=(21, 21), sigmaX=10) # Blur the resized image
    relarge = cv2.resize(blurred, image.T.shape, interpolation=cv2.INTER_AREA) # Resize the blurred image back to the original size
    difference = image - relarge # Calculate the difference between the original and resized-blurred images
    return difference


def find_connected_segments(image, segmentation_threshold) -> tuple:
    '''PART OF STEP 2. Identify cell candidates'''
    n_segments, segment_masks, segment_stats, segment_location = cv2.connectedComponentsWithStats(np.int8(image > segmentation_threshold))
    segment_location = np.int32(segment_location)
    segment_location = np.flip(segment_location, 1) 
    return (n_segments, segment_masks, segment_stats, segment_location)


def filter_cell_candidates(
    animal,
    section_number,
    connected_segments,
    max_segment_size,
    cell_radius,
    x_window,
    y_window,
    absolute_coordinates,
    difference_ch1,
    difference_ch3,
):
    """PART OF STEP 2. Identify cell candidates:  Area is for the object, where pixel values are not zero,
    Segments are filtered to remove those that are too large or too small"""
    n_segments, segment_masks, segment_stats, segment_location = (connected_segments)
    cell_candidates = []
    for segmenti in range(n_segments):
        _, _, width, height, object_area = segment_stats[segmenti, :]
        if object_area > max_segment_size:
            continue
        segment_row, segment_col = segment_location[segmenti, :]

        row_start = int(segment_row - cell_radius)
        col_start = int(segment_col - cell_radius)
        if row_start < 0 or col_start < 0:
            continue
        row_end = int(segment_row + cell_radius)
        col_end = int(segment_col + cell_radius)
        if (
            row_end > x_window or col_end > y_window
        ):  # row evaluates with x-axis (width), col evaluates with y-axis (height)
            continue
        segment_mask = (segment_masks[row_start:row_end, col_start:col_end] == segmenti)

        #FINAL SANITY CHECK
        img_CH1 = difference_ch1[row_start:row_end, col_start:col_end].T
        img_CH3 = difference_ch3[row_start:row_end, col_start:col_end].T
        if img_CH1.shape != img_CH3.shape or img_CH1.shape != segment_mask.shape:
            print(f"ERROR: Image shapes do not match. Skipping this segment.")
            print(f'img_CH1: {img_CH1.shape}')
            print(f'img_CH3: {img_CH3.shape}')
            print(f'segment_mask: {segment_mask.shape}')
            continue

        cell = {
            "animal": animal,
            "section": section_number,
            "area": object_area,
            "absolute_coordinates_YX": (
                absolute_coordinates[2] + segment_col,
                absolute_coordinates[0] + segment_row,
            ),
            "cell_shape_XY": (height, width),
            "image_CH3": img_CH3,
            "image_CH1": img_CH1,
            "mask": segment_mask.T,
        }                                        
        cell_candidates.append(cell)
    return cell_candidates


def calculate_correlation_and_energy(avg_cell_img, cell_candidate_img):  
    '''part of step 3. 
    calculate cell features; calculate correlation [between cell_candidate_img 
    and avg_cell_img] and and energy for cell canididate
    NOTE: avg_cell_img and cell_candidate_img contain respective channels prior to passing in arguments
    '''
    
    if avg_cell_img is None or avg_cell_img.size == 0:
        print(f'DEBUG: avg_cell_img={avg_cell_img.size}, cell_candidate_img={cell_candidate_img.size}')
        raise ValueError(f"Error: 'avg_cell_img' is empty or not loaded properly.")
    
    # Ensure image arrays to same size
    cell_candidate_img, avg_cell_img = equalize_array_size_by_trimming(cell_candidate_img, avg_cell_img)
    # print(f'DEBUG2: {avg_cell_img.size}')
    # print(f'DEBUGA2: {cell_candidate_img.size}')
    # print('*'*40)
    if avg_cell_img is None or avg_cell_img.size == 0:
        raise ValueError(f"Error2: 'avg_cell_img' is empty or not loaded properly.")

    # Compute normalized sobel edge magnitudes using gradients of candidate image vs. gradients of the example image
    avg_cell_img_x, avg_cell_img_y = sobel(avg_cell_img)
    cell_candidate_img_x, cell_candidate_img_y = sobel(cell_candidate_img)

    # corr = the mean correlation between the dot products at each pixel location
    dot_prod = (avg_cell_img_x * cell_candidate_img_x) + (avg_cell_img_y * cell_candidate_img_y)
    corr = np.mean(dot_prod.flatten())      

    # energy: the mean of the norm of the image gradients at each pixel location
    mag = np.sqrt(cell_candidate_img_x **2 + cell_candidate_img_y **2)
    energy = np.mean((mag * avg_cell_img).flatten())  
    return corr, energy


def equalize_array_size_by_trimming(array1, array2):
    '''PART OF STEP 3. CALCULATE CELL FEATURES; array1 and array 2 the same size'''
    size0 = min(array1.shape[0], array2.shape[0])
    size1 = min(array1.shape[1], array2.shape[1])
    array1 = trim_array_to_size(array1, size0, size1)
    array2 = trim_array_to_size(array2, size0, size1)
    return array1, array2    


def trim_array_to_size(arr, size0, size2):
    '''PART OF STEP 3. CALCULATE CELL FEATURES'''
    if(arr.shape[0] > size0):
        size_difference = int((arr.shape[0]-size0)/2)
        arr = arr[size_difference:size_difference+size0, :]
    if(arr.shape[1] > size2):
        size_difference = int((arr.shape[1]-size2)/2)
        arr = arr[:, size_difference:size_difference+size2]
    return arr


def sobel(img):
    '''PART OF STEP 3. CALCULATE CELL FEATURES; Compute the normalized sobel edge magnitudes'''

    if img is None or img.size == 0:
        raise ValueError("Error: The input image is empty or not loaded properly.")

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    _mean = (np.mean(sobel_x) + np.mean(sobel_y))/2.
    _std = np.sqrt((np.var(sobel_x) + np.var(sobel_y))/2)

    eps = 1e-8 #If _std is zero or contains NaN/Inf values, the normalization step will fail. 
    sobel_x = (sobel_x - _mean) / (_std + eps)
    sobel_y = (sobel_y - _mean) / (_std + eps)
    return sobel_x, sobel_y


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