import os
import numpy as np

from library.utilities.utilities_mask import equalized, scaled
from library.utilities.utilities_process import read_image, write_image

class Normalizer:
    """Single method to normlize images
    """


    def create_normalized_image(self):
        """Normalize the downsampled images with QC applied
        Note, normalized images must be of type unit8. We use pillow and torchvision to create
        the masks and 16bit images do not work.
        """
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
                dtype = img.dtype
                if self.debug:
                    print(f'{file} dtype={img.dtype} shape={img.shape} ndim={img.ndim}')

                if dtype == np.uint16:
                    img = (img / 256).astype(np.uint8)

                scale = 250
                img = scaled(img, scale=scale)
                write_image(outfile, img.astype(np.uint8))
