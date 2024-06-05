import os
import numpy as np

from library.utilities.utilities_mask import equalized, scaled
from library.utilities.utilities_process import read_image, write_image

class Normalizer:
    """Single method to normlize images
    """


    def create_normalized_image(self):
        """Normalize the downsampled images with QC applied"""
        if self.downsample:
            self.input = self.fileLocationManager.get_thumbnail(self.channel)
            self.output = self.fileLocationManager.get_normalized(self.channel)
            self.logevent(f"self.input FOLDER: {self.input}")
            files = sorted(os.listdir(self.input))
            self.logevent(f"CURRENT FILE COUNT: {len(files)}")
            self.logevent(f"Output FOLDER: {self.output}")
            os.makedirs(self.output, exist_ok=True)

            for file in files:
                infile = os.path.join(self.input, file)
                outfile = os.path.join(self.output, file)
                if os.path.exists(outfile):
                    continue
                
                img = read_image(infile)
                if self.debug:
                    print(f'{file} dtype={img.dtype} shape={img.shape} ndim={img.ndim}')

                if img.dtype == np.uint16:
                    img = (img / 256).astype(np.uint8)

                if img.ndim == 2:
                    img = equalized(img)
                write_image(outfile, img.astype(np.uint8))
