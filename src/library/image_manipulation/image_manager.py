import os
import sys
import numpy as np
import tifffile
from skimage import io
import cv2
import glob

from library.utilities.utilities_mask import rescaler
from library.utilities.utilities_process import read_image

class ImageManager:
    """
    A class for managing image files and performing image manipulation operations.

    Args:
        directory (str): The directory path where the image files are located.
        filetype (str, optional): The file extension of the image files. Defaults to 'tif'.

    Attributes:
        files (list): A sorted list of image file paths in the specified directory.
        len_files (int): The number of image files in the directory.
        midpoint (int): The index of the middle image file in the list.
        midfile (str): The path of the middle image file.
        img (ndarray): The image data of the middle image file.
        dtype (str): The data type of the image.
        ndim (int): The number of dimensions of the image.
        shape (tuple): The shape of the image.
        width (int): The width of the image.
        height (int): The height of the image.
        center (ndarray): The center coordinates of the image.
        volume_size (tuple): The size of the image volume (width, height, number of files).
        num_channels (int): The number of color channels in the image.
        size (tuple): The size of the image (width, height).
        DK37 024 full res size = 1,842,750,000
        DK37 024 downsampled size = 1,555,500
        YH10 full res size = 35,490,384



    Methods:
        get_bgcolor(maskpath): Returns the background color of the image based on a sample image

    """

    def __init__(self, directory, filetype='tif'):
        self.files = sorted(glob.glob( os.path.join(directory, f'*.{filetype}') ))
        self.len_files = len(self.files)
        if self.len_files == 0:
            print(f'No image files found in: {directory}')
            sys.exit(1)
        self.midpoint = (self.len_files // 2) # works with odd number of files
        self.midfile = self.files[self.midpoint] 
        midfilepath = os.path.join(directory, self.midfile)
        self.img = read_image(midfilepath)
        self.img = np.squeeze(self.img)  # Remove single-dimensional entries from the shape of the array
        self.dtype = self.img.dtype
        self.ndim = self.img.ndim
        self.shape = self.img.shape

        if self.ndim == 4:
            self.width = self.shape[-1]
            self.height = self.shape[-2]
            self.num_channels = self.shape[-1]
        elif self.ndim == 3:
            self.width = self.shape[1]
            self.height = self.shape[0]
            self.num_channels = self.shape[-1]
        else:
            self.width = self.shape[1]
            self.height = self.shape[0]
            self.num_channels = 1
        self.center = np.array([self.width, self.height]) / 2
        self.volume_size = (self.width, self.height, self.len_files)
        self.size = self.img.size

    def get_bgcolor(self):
        """align needs either an integer or a tuple of integers for the fill color
        Get the background color of the image based on the the 10th row and 10th column of the image.
        This is usually the background color in the image.
        """
        debug = False

        if self.img.ndim == 2:
            # If the image is grayscale, return the background color as an integer
            bgcolor = int(self.img[10, 10])
            return bgcolor

        test_image = self.img[self.img != 255]  # Get the pixel value at (10, 10)

        unique_values, counts = np.unique(test_image, return_counts=True)
        max_count_index = np.argmax(counts)
        bgcolor = unique_values[max_count_index]

        if self.img.ndim == 3 and self.img.shape[-1] > 1:
            # If the image has multiple channels, return the background color as a tuple
            bgcolor = (bgcolor,) * self.img.shape[-1]
        elif self.img.ndim == 3 and self.img.shape[-1] == 1:
            # If the image has a single channel, return the background color as an integer
            bgcolor = int(bgcolor)
        else:
            # If the image is grayscale, return the background color as an integer
            bgcolor = int(bgcolor)

        return bgcolor

    def get_reference_image(self, maskpath):
        """Get the reference image for alignment

        Args:
            reference_file (str): The file path of the reference image.

        Returns:
            ndarray: The reference image data.
        """
        self.masks = sorted(os.listdir(maskpath))
        midmaskfile = self.masks[self.midpoint]
        midmaskpath = os.path.join(maskpath, midmaskfile)

        self.mask = tifffile.imread(midmaskpath)
        reference_file = self.files[self.midpoint]
        reference_image = read_image(reference_file)
        cleaned = cv2.bitwise_and(reference_image, reference_image, mask=self.mask)                                   
        rescaled = rescaler(cleaned)
        return rescaled
