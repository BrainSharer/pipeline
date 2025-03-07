import os
import sys
import cv2
import imageio
import numpy as np

def calculate_correlation_and_energy(avg_cell_img, cell_candidate_img):  
    '''part of step 3. 
    calculate cell features; calculate correlation [between cell_candidate_img 
    and avg_cell_img] and and energy for cell canididate
    NOTE: avg_cell_img and cell_candidate_img contain respective channels prior to passing in arguments
    '''

    # Ensure image arrays to same size
    cell_candidate_img, avg_cell_img = equalize_array_size_by_trimming(cell_candidate_img, avg_cell_img)

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
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    _mean = (np.mean(sobel_x) + np.mean(sobel_y))/2.
    _std = np.sqrt((np.var(sobel_x) + np.var(sobel_y))/2)
    sobel_x = (sobel_x - _mean) / _std
    sobel_y = (sobel_y - _mean) / _std
    return sobel_x, sobel_y

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
        cell = {
            "animal": animal,
            "section": section_number,
            "area": object_area,
            "absolute_coordinates_YX": (
                absolute_coordinates[2] + segment_col,
                absolute_coordinates[0] + segment_row,
            ),
            "cell_shape_XY": (height, width),
            "image_CH3": difference_ch3[row_start:row_end, col_start:col_end].T,
            "image_CH1": difference_ch1[row_start:row_end, col_start:col_end].T,
            "mask": segment_mask.T,
        }                                        
        cell_candidates.append(cell)
    return cell_candidates

def append_string_to_every_key(dictionary, post_fix): 
    return dict(zip([keyi + post_fix for keyi in dictionary.keys()],dictionary.values()))


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
    moments = append_string_to_every_key(moments, f'_mask')
    return (moments, {'h%d'%i+f'_mask':huMoments[i,0]  for i in range(7)}) #return first 7 Hu moments e.g. h1_mask

def features_using_center_connected_components(cell_candidate_data):   
    '''Part of step 3. calculate cell features'''
    def mask_mean(mask,image):
        mean_in=np.mean(image[mask==1])
        mean_all=np.mean(image.flatten())
        return (mean_in-mean_all)/(mean_in+mean_all)    # calculate the contrast: mean

    mask = cell_candidate_data['mask']  

    moments_data = calc_moments_of_mask(mask)

    # Calculate constrasts relative to mask
    ch1_contrast = mask_mean(mask, cell_candidate_data['image_CH1'])
    ch3_constrast = mask_mean(mask, cell_candidate_data['image_CH3'])

    return ch1_contrast, ch3_constrast, moments_data
