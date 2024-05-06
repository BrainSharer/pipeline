import os
import numpy as np
import tifffile

class ImageManager:
    def __init__(self, directory):
        self.files = sorted(os.listdir(directory))
        len_files = len(self.files)
        self.midpoint = len_files // 2
        midfile = self.files[self.midpoint]
        midfilepath = os.path.join(directory, midfile)
        self.img = tifffile.imread(midfilepath)
        self.dtype = self.img.dtype
        self.ndim = self.img.ndim
        self.shape = self.img.shape
        self.width = self.shape[1]
        self.height = self.shape[0]
        self.center = np.array([self.width, self.height]) / 2



    def get_bgcolor(self, maskpath):
        self.masks = sorted(os.listdir(maskpath))
        midmaskfile = self.masks[self.midpoint]
        midmaskpath = os.path.join(maskpath, midmaskfile)

        self.mask = tifffile.imread(midmaskpath)

        if self.mask is None:
            print('Warning: no mask is availabe for this image')
            return 0
        
        white = np.where(self.mask==255)
        whiterows = white[0]
        firstrow = whiterows[0]
        bgcolor = (np.max(self.img[firstrow]))
        return bgcolor

