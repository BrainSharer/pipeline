import os
import numpy as np

from library.utilities.utilities_mask import compare_directories, equalized, scaled
from library.utilities.utilities_process import read_image, write_image

class Normalizer:
    """Single method to normlize images
    """


    def create_normalized_image(self):
        """Normalize the downsampled images with QC applied
        Note, normalized images must be of type unit8. We use pillow and torchvision to create
        the masks and 16bit images do not work.
        Converting from 16bit sRGB to 8bit grayscale uses: 
        https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
        """
        self.input = self.fileLocationManager.get_thumbnail(self.channel)
        self.output = self.fileLocationManager.get_normalized(self.channel)
        self.fileLogger.logevent(f"self.input FOLDER: {self.input}")
        files = sorted(os.listdir(self.input))
        self.fileLogger.logevent(f"CURRENT FILE COUNT: {len(files)}")
        self.fileLogger.logevent(f"Output FOLDER: {self.output}")
        os.makedirs(self.output, exist_ok=True)

        if self.debug:
            print(f'NORMALIZING: {self.input} -> {self.output}')

        for file in files:
            infile = os.path.join(self.input, file)
            outfile = os.path.join(self.output, file)
            if os.path.exists(outfile):
                continue
            
            img = read_image(infile)
            if img.ndim == 3:
                img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
                img = img.astype(np.uint8)

            if self.debug:
                print(f'{file} dtype={img.dtype} shape={img.shape} ndim={img.ndim}')

            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)

            scale = 250
            img = scaled(img, scale=scale)
            write_image(outfile, img.astype(np.uint8))

        compare_directories(self.input, self.output)
