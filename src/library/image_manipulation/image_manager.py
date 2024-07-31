import os
import sys
import numpy as np
import tifffile
from skimage import io
import glob

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

    Methods:
        get_bgcolor(maskpath): Returns the background color of the image based on the provided mask.

    """

    def __init__(self, directory, filetype='tif'):
        self.files = sorted(glob.glob( os.path.join(directory, f'*.{filetype}') ))
        self.len_files = len(self.files)
        if self.len_files == 0:
            print(f'No image files found in: {directory}')
            sys.exit(1)
        self.midpoint = self.len_files // 2
        self.midfile = self.files[self.midpoint]
        midfilepath = os.path.join(directory, self.midfile)
        self.img = io.imread(midfilepath)
        self.dtype = self.img.dtype
        self.ndim = self.img.ndim
        self.shape = self.img.shape
        self.width = self.shape[1]
        self.height = self.shape[0]
        self.center = np.array([self.width, self.height]) / 2
        self.volume_size = (self.width, self.height, self.len_files)
        self.num_channels = self.img.shape[2] if len(self.img.shape) > 2 else 1



    def get_bgcolor(self, maskpath):
        """align needs either an integer or a tuple of integers for the fill color
        """

        self.masks = sorted(os.listdir(maskpath))
        if len(self.masks) != self.len_files:
            print('Warning: no masks are available for this image')
            if self.ndim == 3:
                return (255,255,255)
            else:
                return 0
        midmaskfile = self.masks[self.midpoint]
        midmaskpath = os.path.join(maskpath, midmaskfile)

        self.mask = tifffile.imread(midmaskpath)

        if self.mask is None:
            print('Warning: no mask is availabe for this image')
            return 0
        
        white = np.where(self.mask==255)
        whiterows = white[0]
        whitecols = white[1]
        firstrow = whiterows[0]
        firstcol = whitecols[1]
        bgcolor = self.img[firstrow, firstcol]
        if isinstance(bgcolor, (list, np.ndarray)):
            bgcolor = tuple(bgcolor)
        else:
            bgcolor = int(bgcolor)
        return bgcolor

