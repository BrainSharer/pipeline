import os
import numpy as np

from library.utilities.utilities_mask import equalized
from library.utilities.utilities_process import read_image, write_image

class Normalizer:
    """Single method to normlize images
    """


    def create_normalized_image(self):
        """Normalize the downsampled images with QC applied"""
        if self.downsample:
            INPUT = self.fileLocationManager.get_thumbnail(self.channel)
            OUTPUT = self.fileLocationManager.get_normalized(self.channel)
            self.logevent(f"INPUT FOLDER: {INPUT}")
            files = sorted(os.listdir(INPUT))
            self.logevent(f"CURRENT FILE COUNT: {len(files)}")
            self.logevent(f"OUTPUT FOLDER: {OUTPUT}")
            os.makedirs(OUTPUT, exist_ok=True)

            for file in files:
                infile = os.path.join(INPUT, file)
                outfile = os.path.join(OUTPUT, file)
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
