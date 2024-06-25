import os
import numpy as np
import tifffile
from skimage import io


class ImageManager:
    def __init__(self, directory):
        self.files = sorted(os.listdir(directory))
        self.len_files = len(self.files)
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
        """align needs either an integer or a tuple of integers for the fill color"""

        self.masks = sorted(os.listdir(maskpath))
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

