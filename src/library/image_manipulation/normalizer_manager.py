import os
import cv2
import sys
import numpy as np
from skimage import io

from library.utilities.utilities_mask import equalized

class Normalizer:
    """Single method to normlize images
    """


    def create_normalized_image(self):
        """Normalize the downsampled images with QC applied"""
        if self.downsample:
            INPUT = self.fileLocationManager.get_thumbnail(self.active_channel)
            OUTPUT = self.fileLocationManager.get_normalized(self.active_channel)
            self.logevent(f"INPUT FOLDER: {INPUT}")
            files = sorted(os.listdir(INPUT))
            self.logevent(f"CURRENT FILE COUNT: {len(files)}")
            self.logevent(f"OUTPUT FOLDER: {OUTPUT}")
            os.makedirs(OUTPUT, exist_ok=True)

            for file in files:
                infile = os.path.join(INPUT, file)
                outpath = os.path.join(OUTPUT, file)
                if os.path.exists(outpath):
                    continue
                try:
                    img = io.imread(infile)
                except IOError as errno:
                    print(f"I/O error {errno}")
                except ValueError:
                    print("Could not convert data to an integer.")
                except:
                    print(f' Could not open {infile}')
                    print("Unexpected error:", sys.exc_info()[0])
                    sys.exit()


                if img.dtype == np.uint16:
                    img = (img / 256).astype(np.uint8)
                img = equalized(img)
                cv2.imwrite(outpath, img.astype(np.uint8))
