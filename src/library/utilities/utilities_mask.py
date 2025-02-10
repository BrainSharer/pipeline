"""Simple methods to help in manipulating images.
"""

import os
import sys
import cv2
import numpy as np
from skimage.exposure import rescale_intensity
from library.database_model.scan_run import FULL_MASK, FULL_MASK_NO_CROP
from skimage import color
from scipy.ndimage import binary_fill_holes
from skimage import exposure
from tqdm import tqdm

from library.utilities.utilities_process import read_image, write_image


def rotate_image(img, file: str, rotation: int):
    """Rotate the image by the number of rotation(s)

    Rotate the image by the number of rotation
    :param img: image to work on
    :param file: file name and path
    :param rotation: number of rotations, 1 = 90degrees clockwise
    :return: rotated image
    """

    try:
        img = np.rot90(img, rotation, axes=(1,0))
    except:
        print('Could not rotate', file)
    return img



def place_image(file_key: tuple, bgcolor: int = 0):
    infile, outfile, max_width, max_height, bgcolor = file_key
    img = read_image(infile)
    dtype = img.dtype

    zmidr = max_height // 2
    zmidc = max_width // 2

    startr = zmidr - (img.shape[0] // 2)
    endr = startr + img.shape[0]
    startc = zmidc - (img.shape[1] // 2)
    endc = startc + img.shape[1]

    if startr < 0:
        print(f'Error placing {infile} start row {startr=} < 0')
        sys.exit()

    if startc < 0:
        print(f'Error placing {infile} start column {startc=} < 0')
        sys.exit()
    


    if img.ndim == 2:  # Grayscale
        placed_img = np.full((max_height, max_width), bgcolor, dtype=dtype)
        try:
            #placed_img[startr:endr, startc:endc] = img[:endr-startr, :endc-startc] I think this is wrong
            placed_img[startr:endr, startc:endc] = img            
        except Exception as e:
            print(f"Error placing grayscale {infile}: {e}")
            print(f"img shape {img.shape} placed_img shape {placed_img.shape} img ndim {img.ndim}")
            print(f"startr {startr} endr {endr} startc {startc} endc {endc}")
            sys.exit()

    elif (img.ndim == 3 and img.shape[2] == 3):  # Color (RGB)
        r, g, b = (np.full((max_height, max_width), bg, dtype=dtype) for bg in bgcolor)
        try:
            r[startr:endr, startc:endc] = img[:endr-startr, :endc-startc, 0]
            g[startr:endr, startc:endc] = img[:endr-startr, :endc-startc, 1]
            b[startr:endr, startc:endc] = img[:endr-startr, :endc-startc, 2]
            placed_img = cv2.merge((b, g, r))
        except Exception as e:
            print(f"Error placing {infile}: {e}")
            print(f"img shape {img.shape} placed_img shape {placed_img.shape} img ndim {img.ndim}")
            print(f"startr {startr} endr {endr} startc {startc} endc {endc}")
            sys.exit()
    elif (img.ndim == 3 and img.shape[2] == 1): # Grayscale with img.shape[2] == 1
        img = img.squeeze(axis=2)
        placed_img = np.full((max_height, max_width), bgcolor, dtype=dtype)
        try:
            placed_img[startr:endr, startc:endc] = img[:endr-startr, :endc-startc]
        except Exception as e:
            print(f"Error placing {infile}: {e}")
            print(f"img shape {img.shape} placed_img shape {placed_img.shape} img ndim {img.ndim}")
            print(f"startr {startr} endr {endr} startc {startc} endc {endc}")
            sys.exit()
    else:
        print(f"Img {infile} unsupported image shape: {img.shape} ndim={img.ndim} dtype={img.dtype}")
        sys.exit()


    write_image(outfile, placed_img.astype(dtype))




def normalize_image(img):
    """This is a simple opencv image normalization for 16 bit images.

    :param img: the numpy array of the 16bit image
    :return img: the normalized image
    """
    max = 2 ** 16 - 1 # 16bit
    cv2.normalize(img, img, 0, max, cv2.NORM_MINMAX)
    return img


def scaled(img, scale=32000):
    """Stretch values out to scale
    Used to be 45000, but changing it down to 32000 as of 7 Aug 2024
    """
    dtype = img.dtype
    if dtype == np.uint16:
        scale = 32000
    else:
        scale = 250

    epsilon = 0.99    
    _max = np.quantile(img[img>0], epsilon)
    scaled = (img * (scale / _max)).astype(dtype) # scale the image from original values to a broader range of values
    del img
    return scaled

def rescaler(img):
    # Contrast stretching
    lower = 0
    upper = 99.9
    plower, pupper = np.percentile(img, (lower, upper))
    img_rescale = exposure.rescale_intensity(img, in_range=(plower, pupper))
    return img_rescale



def mask_with_background(img, mask):
    """
    Masks the image with the given mask and replaces the masked region with the background color.

    Args:
        img (numpy.ndarray): The input image.
        mask (numpy.ndarray): The mask to be applied on the image.

    Returns:
        numpy.ndarray: The masked image with the background color.

    """
    white = np.where(mask==255)
    whiterows = white[0]
    firstrow = whiterows[1]
    bgcolor = (np.max(img[firstrow]))
    img[mask == 0] = bgcolor
    return img


def mask_with_contours(img):

    new_img = color.rgb2gray(img)
    new_img *= 255  # or any coefficient
    new_img = new_img.astype(np.uint8)
    new_img[(new_img > 200)] = 0
    lowerbound = 0
    upperbound = 255
    # all pixels value above lowerbound will  be set to upperbound
    _, thresh = cv2.threshold(new_img.copy(), lowerbound, upperbound, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    smoothed = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel)
    inverted_thresh = cv2.bitwise_not(smoothed)
    filled_thresh = binary_fill_holes(inverted_thresh).astype(np.uint8)
    return cv2.bitwise_and(img, img, mask=filled_thresh)
    # return cv2.bitwise_not(img, filled_thresh)


def equalized(fixed, cliplimit=5):
    """Takes an image that has already been scaled and uses opencv adaptive histogram
    equalization. This cases uses 5 as the clip limit and splits the image into rows
    and columns. A higher cliplimit will make the image brighter. A cliplimit of 1 will
    do nothing. 

    :param fixed: image we are working on
    :return: a better looking image
    """
    cliplimit = 5
    if fixed.ndim == 3:
        fixed = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
        fixed = scaled(fixed, scale=200)
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(8, 32))
    fixed = clahe.apply(fixed)
    return fixed

def normalize8(img):
    mn = img.min()
    mx = img.max()
    mx -= mn
    img = ((img - mn)/mx) * 2**8 - 1
    return np.round(img).astype(np.uint8) 

def normalize16(img):
    if img.dtype == np.uint32:
        print('image dtype is 32bit')
        return img.astype(np.uint16)
    else:
        mn = img.min()
        mx = img.max()
        mx -= mn
        img = ((img - mn)/mx) * 2**16 - 1
        return np.round(img).astype(np.uint16) 


def clean_and_rotate_image(file_key: tuple[str, str, str, int, str, bool, int, int, bool]) -> None:
    """The main function that uses the user edited mask to crop out the tissue from 
    surrounding debris. It also rotates the image to
    a usual orientation (where the olfactory bulb is facing left and the cerebellum is facing right.
    The hippocampus is facing up and the brainstem is facing down)
    Normalization needs adjusting, for section 064 of DK101, the cells get an uwanted
    outline that is far too bright. This happens with the scaled method.
    An affected area on 064.tif full resolution is top left corner of 32180x19665, and
    bottom right corner at: 33500x20400 on the full cleaned version
    For the regular tif, look at 15812x43685, 16816x44463

    :param file_key: is a tuple of the following:

    - infile file path of image to read
    - outpath file path of image to write
    - mask binary mask image of the image
    - rotation number of 90 degree rotations
    - flip either flip or flop
    - max_width width of image
    - max_height height of image
    - scale used in scaling. Gotten from the histogram

    :return: nothing. we write the image to disk
    """

    infile, outfile, maskfile, rotation, flip, mask_image, bgcolor, channel, debug = file_key

    img = read_image(infile)

    mask = read_image(maskfile)
    # Ensure mask is binary and uint8
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    # Handle different image types
    if img.ndim == 2:  # Grayscale
        img = img.astype(np.uint16)  # Retain original dtype
    elif img.ndim == 3:  # Color
        img = img.astype(np.uint16)  # Retain original dtype
        if mask.ndim == 2:
            mask = cv2.merge([mask] * 3)
    try:
        cleaned = cv2.bitwise_and(img, img, mask=mask)
    except:
        # May as well exit as something is very wrong.
        print(f"Error in masking {infile} with mask shape {mask.shape} img shape {img.shape}")
        fix = False
        if fix:
            print("Resizing mask to fix")
            try:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            except:
                print("Could not resize mask to fit image")
                print(f"Mask shape {mask.shape} Image shape {img.shape}")
                sys.exit()
            try:
                cleaned = cv2.bitwise_and(img, img, mask=mask)
            except:
                print("Could not clean image with this mask")
                print(f"Mask shape {mask.shape} Image shape {img.shape}")
                sys.exit()
        else:
            print("Image size does not match mask size, please fix")
            sys.exit()

    if cleaned.dtype == np.uint8 and cleaned.ndim == 3:
        #b, g, r = cv2.split(cleaned) # this is an expensive function, using numpy is faster
        r = cleaned[:,:,0]
        g = cleaned[:,:,1]
        b = cleaned[:,:,2]
        r[r == 0] = bgcolor[0]
        g[g == 0] = bgcolor[1]
        b[b == 0] = bgcolor[2]
        cleaned = cv2.merge((b,g,r)) # put them back in the correct order for cv2

    if channel == 1:    
        cleaned = rescaler(cleaned)

    if mask_image == FULL_MASK:
        cleaned = crop_image(cleaned, mask)
        del img
        del mask

    if rotation > 0:
        cleaned = rotate_image(cleaned, infile, rotation)
    # flip = switch top to bottom
    # flop = switch left to right
    if flip == "flip":
        cleaned = np.flip(cleaned, axis=0)
    if flip == "flop":
        cleaned = np.flip(cleaned, axis=1)

    message = f'Error in saving {outfile} with shape {cleaned.shape} img type {cleaned.dtype}'
    write_image(outfile, cleaned, message=message)


def crop_image(img, mask):
    """Crop image to remove parts of image not in mask

    :param img: numpy array of image
    :param mask: numpy array of mask
    :return: numpy array of cropped image
    """

    x1, y1, x2, y2 = get_image_box(mask)
    img = np.ascontiguousarray(img, dtype=img.dtype)
    cropped = img[y1:y2, x1:x2]
    return cropped


def get_image_box(img):
    """
    Computes the bounding box coordinates of the non-zero regions in a binary image.

    Args:
        img (numpy.ndarray): Input binary image where non-zero pixels represent the object.

    Returns:
        tuple: A tuple (x1, y1, x2, y2) representing the coordinates of the bounding box 
               where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    """

    mask = img.copy()
    if mask.dtype == np.uint16:
        mask = (mask / 256).astype(np.uint8)
    mask[mask > 0] = 255
    _, thresh = cv2.threshold(mask, 200, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 100:
            xmin = int(round(x))
            ymin = int(round(y))
            xmax = int(round(x + w))
            ymax = int(round(y + h))
            boxes.append([xmin, ymin, xmax, ymax])
    x1 = min(x[0] for x in boxes)
    y1 = min(x[1] for x in boxes)
    x2 = max(x[2] for x in boxes)
    y2 = max(x[3] for x in boxes)
    x1, y1, x2, y2 = [0 if i < 0 else i for i in [x1, y1, x2, y2]]
    return x1, y1, x2, y2

def get_box_corners(arr):
    areaArray = []  
    _, thresh = cv2.threshold(arr, 200, 250, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)

    #first sort the array by area
    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
    #find the nth largest contour [n-1][1], in this case 2
    secondlargestcontour = sorteddata[1][1]    

    x,y,w,h = cv2.boundingRect(secondlargestcontour)
    p1x = x
    p1y = y
    
    p2x = x+w
    p2y = y
    
    p3x = x
    p3y = y+h
    
    p4x = x+w
    p4y = y+h
    """
    moving_file = os.path.join(self.input, f"{moving_index}.tif")
    moving_point_file = os.path.join(self.registration_output, f'{moving_index}_points.txt')
    if not os.path.exists(moving_point_file):
        moving_arr = read_image(moving_file)
        moving_arr = normalize8(moving_arr)
        p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y = get_box_corners(moving_arr)
        with open(moving_point_file, 'w') as f:
            f.write('point\n')
            f.write('4\n')
            f.write(f'{p1x} {p1y}\n')
            f.write(f'{p2x} {p2y}\n')
            f.write(f'{p3x} {p3y}\n')
            f.write(f'{p4x} {p4y}\n')
    """
    return p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y


def merge_mask(image, mask):
    """Merge image with mask [so user can edit]
    stack 3 channels on single image (black background, image, then mask)

    :param image: numpy array of the image
    :param mask: numpy array of the mask
    :return: merged numpy array
    """
    b = mask

    if image.ndim == 3:     
        g = image[:,:,1]
        r = np.zeros_like(image[:,:,0]).astype(np.uint8)
    else:
        g = image
        r = np.zeros_like(image).astype(np.uint8)
    merged = np.stack([r, g, b], axis=2)
    return merged


def combine_dims(a):
    """Combines dimensions of a numpy array

    :param a: numpy array
    :return: numpy array
    """
    
    if a.shape[0] > 0:
        a1 = a[0,:,:]
        a2 = a[1,:,:]
        a3 = np.add(a1,a2)
    else:
        a3 = np.zeros([a.shape[1], a.shape[2]]) + 255
    return a3

def smooth_image(gray):
    # threshold
    thresh = cv2.threshold(gray, 32, 255, cv2.THRESH_BINARY)[1]
    # blur threshold image
    blur = cv2.GaussianBlur(thresh, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
    # stretch so that 255 -> 255 and 127.5 -> 0
    stretch = rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255)).astype(np.uint8)
    # threshold again
    thresh2 = cv2.threshold(stretch, 0, 255, cv2.THRESH_BINARY)[1]
    # get external contour
    contours = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
    # draw white filled contour on black background as mas
    contour = np.zeros_like(gray)
    cv2.drawContours(contour, [big_contour], 0, 255, -1)
    # dilate mask for dark border
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12,12))
    dilate = cv2.morphologyEx(contour, cv2.MORPH_CLOSE, kernel)
    # apply morphology erode
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
    dilate = cv2.morphologyEx(dilate, cv2.MORPH_ERODE, kernel)
    # blur dilate image
    blur2 = cv2.GaussianBlur(dilate, (3,3), sigmaX=0, sigmaY=0, borderType = cv2.BORDER_DEFAULT)
    # stretch so that 255 -> 255 and 127.5 -> 0
    mask = rescale_intensity(blur2, in_range=(127.5,255), out_range=(0,255))
    #return cv2.bitwise_and(gray, gray, mask=mask.astype(np.uint8))
    return cv2.bitwise_and(gray, mask.astype(np.uint8), mask=None)

def match_histogramsXXX(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram matches that of a target image

    Arguments:
    source -- a grayscale image which histogram will be modified
    template -- a grayscale image which histogram will be used as a reference

    Returns:
    a grayscale image with the same size as source
    """
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def match_histograms(cleaned, reference):
    #basepath = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration'
    #allenpath = os.path.join(basepath, 'Allen_25um_sagittal_mid.tif')
    #referencepath = os.path.join(basepath, 'out.tif')
    #referencepath = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK161/preps/out.tif'
    #reference = read_image(referencepath)
    #shapeto = cleaned.shape
    #reference = midallenarr[midallenarr > 0]
    #reference = reference.flatten()
    #target = cleaned.flatten()
    img = exposure.match_histograms(cleaned, reference)
    #img = img.reshape(shapeto)
    data = img / np.max(img) # normalize the data to 0 - 1
    del img
    data = 65535 * data # Now scale by bits
    return data.astype(np.uint16)    

def create_mask(image):
    if image.dtype == np.uint16:
        image = (image/256).astype(np.uint8)

    ret,thresh = cv2.threshold(image, 0, 200, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areaArray = []
    for contour in contours:
        area = cv2.contourArea(contour)
        print(area)
        areaArray.append(area)
    # first sort the array by area
    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
    largest_contour = sorteddata[0][1]
    output = cv2.fillPoly(image, pts =[largest_contour], color=255)
    return output


def compare_directories(dir1: str, dir2: str) -> None:
    """
    Compares the contents of two directories to ensure they have the same files and that the images
    within those files have the same dimensions.
    Args:
        dir1 (str): The path to the first directory.
        dir2 (str): The path to the second directory.
    Raises:
        AssertionError: If the number of files in the directories are not equal or if any directory is empty.
        SystemExit: If there are any mismatches in file names or image dimensions, the function prints the errors and exits the program.
    """
    error = ""
    files1 = sorted(os.listdir(dir1))
    files2 = sorted(os.listdir(dir2))
    assert len(files1) == len(files2), f"Length of {dir1} {len(files1)} != {dir2} {len(files2)}"
    assert len(files1) > 0, f"Empty directory {dir1}"
    desc = f"Comparing {os.path.basename(dir1)} and {os.path.basename(dir2)}"
    for file1, file2 in tqdm(zip(files1, files2), total=len(files1), desc=desc):
        img1 = read_image(os.path.join(dir1, file1))
        img2 = read_image(os.path.join(dir2, file2))
        if file1 != file2: error += f"{file1} != {file2}\n"
        if img1.shape[0] != img2.shape[0]: error += f"{file1} rows {img1.shape[0]} != {file2} {img2.shape[0]}\n"
        if img1.shape[1] != img2.shape[1]: error += f"{file1} cols {img1.shape[1]} != {file2} {img2.shape[1]}\n"
    
    if len(error) > 0:
        print(error)
        print(f"Error {desc}")
        sys.exit()