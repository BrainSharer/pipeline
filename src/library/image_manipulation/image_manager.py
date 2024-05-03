import os
import numpy as np
import tifffile

class ImageManager:
    def __init__(self, directory, maskpath=None):
        self.files = sorted(os.listdir(directory))
        self.masks = sorted(os.listdir(maskpath))
        len_files = len(self.files)
        assert len_files == len(self.masks), 'Number of files and masks do not match.'
        midpoint = len_files // 2
        midfile = self.files[midpoint]
        midmaskfile = self.masks[midpoint]
        midfilepath = os.path.join(directory, midfile)
        midmaskpath = os.path.join(maskpath, midmaskfile)
        self.img = tifffile.imread(midfilepath)
        self.dtype = self.img.dtype
        self.ndim = self.img.ndim
        self.shape = self.img.shape
        self.mask = tifffile.imread(midmaskpath)


    def get_bgcolor(self):

        if self.mask is None:
            print('Warning: no mask is availabe for this image')
            return 0
        
        white = np.where(self.mask==255)
        whiterows = white[0]
        firstrow = whiterows[0]
        bgcolor = (np.max(self.img[firstrow]))
        return bgcolor

