"""Simple methods to help in manipulating images.
"""

import sys
import cv2
import numpy as np
from skimage.exposure import rescale_intensity
from library.database_model.scan_run import FULL_MASK

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


def place_image(img, file: str, max_width, max_height, bgcolor=None):
    """Places the image in a padded one size container with the correct background

    :param img: image we are working on.
    :param file: file name and path location
    :param max_width: width to pad
    :param max_height: height to pad
    :param bgcolor: background color of image, 0 for NTB, white for thionin
    :return: placed image centered in the correct size.
    """

    zmidr = max_height // 2
    zmidc = max_width // 2
    startr = zmidr - (img.shape[0] // 2)
    endr = startr + img.shape[0]
    startc = zmidc - (img.shape[1] // 2)
    endc = startc + img.shape[1]
    dt = img.dtype
    if bgcolor == None:
        start_bottom = img.shape[0] - 5
        bottom_rows = img[start_bottom:img.shape[0], :]
        avg = np.mean(bottom_rows)
        bgcolor = int(round(avg))
    new_img = np.zeros([max_height, max_width]).astype(dt) + bgcolor
    #print(f'Resizing {file} from {img.shape} to {new_img.shape}')
    if img.ndim == 2:
        try:
            new_img[startr:endr, startc:endc] = img
        except:
            ###mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            #img = cv2.resize(img, (new_img.shape[1], new_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            print(f'Could not place {file} with shape:{img.shape} in {max_height}x{max_width}')
    if img.ndim == 3:
        try:
            new_img = np.zeros([max_height, max_width, 3]) + bgcolor
            new_img[startr:endr, startc:endc,0] = img[:,:,0]
            new_img[startr:endr, startc:endc,1] = img[:,:,1]
            new_img[startr:endr, startc:endc,2] = img[:,:,2]
        except:
            print(f'Could not place 3DIM {file} with width:{img.shape[1]}, height:{img.shape[0]} in {max_width}x{max_height}')
    del img
    return new_img.astype(dt)


def normalize_image(img):
    """This is a simple opencv image normalization for 16 bit images.

    :param img: the numpy array of the 16bit image
    :return img: the normalized image
    """
    max = 2 ** 16 - 1 # 16bit
    cv2.normalize(img, img, 0, max, cv2.NORM_MINMAX)
    return img


def scaled(img, scale=20000):
    """First we find really high values, which are the bright spots and turn them down
    """
    dtype = img.dtype
    epsilon = 0.99    
    _max = np.quantile(img[img>0], epsilon)
    scaled = (img * (scale / _max)).astype(dtype) # scale the image from original values to a broader range of values
    del img
    return scaled



def equalized(fixed, cliplimit=5):
    """Takes an image that has already been scaled and uses opencv adaptive histogram
    equalization. This cases uses 5 as the clip limit and splits the image into rows
    and columns. A higher cliplimit will make the image brighter. A cliplimit of 1 will
    do nothing. 

    :param fixed: image we are working on
    :return: a better looking image
    """
    
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(8, 8))
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

def clean_and_rotate_image(file_key):
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

    infile, outpath, maskfile, rotation, flip, max_width, max_height = file_key

    img = read_image(infile)
    mask = read_image(maskfile)
    cleaned = apply_mask(img, mask, infile)
    cleaned = scaled(cleaned)

    # Cropping is not working 100% of the time
    #if mask_image == FULL_MASK:
    #    cleaned = crop_image(cleaned, mask)
    del img
    del mask
    if rotation > 0:
        cleaned = rotate_image(cleaned, infile, rotation)
    if flip == "flip":
        cleaned = np.flip(cleaned)
    if flip == "flop":
        cleaned = np.flip(cleaned, axis=1)
    cleaned = place_image(cleaned, infile, max_width, max_height, bgcolor=0)

    message = f'Error in saving {outpath} with shape {cleaned.shape} img type {cleaned.dtype}'
    write_image(outpath, cleaned, message=message)
        
    return


def apply_mask(img, mask, infile):
    """Apply image mask to image.

    :param img: numpy array of image
    :param mask: numpy array of mask
    :param infile: path to file
    :return: numpy array of cleaned image
    """

    try:
        cleaned = cv2.bitwise_and(img, img, mask=mask)
    except:
        print(f"Error in masking {infile} with mask shape {mask.shape} img shape {img.shape}")
        print("Are the shapes exactly the same?")
        print("Unexpected error:", sys.exc_info()[0])
        raise
    return cleaned


def crop_image(img, mask):
    """Crop image to remove parts of image not in mask

    :param img: numpy array of image
    :param mask: numpy array of mask
    :return: numpy array of cropped image
    """

    x1, y1, x2, y2 = get_image_box(mask)
    img = np.ascontiguousarray(img, dtype=np.uint16)
    cropped = img[y1:y2, x1:x2]
    return cropped


def get_image_box(mask):
    """Find new max width and height

    :param img: numpy array of image
    :param mask: numpy array of mask
    :return: numpy array of cropped image
    """

    BUFFER = 10
    mask = np.array(mask)
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
    x1 = min(x[0] for x in boxes) - BUFFER
    y1 = min(x[1] for x in boxes) - BUFFER
    x2 = max(x[2] for x in boxes) + BUFFER
    y2 = max(x[3] for x in boxes) + BUFFER
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
