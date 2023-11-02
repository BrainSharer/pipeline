"""This module takes clear of cleaning up the junk from outside
the brain area by using masks.
"""
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from library.utilities.utilities_mask import clean_and_rotate_image, get_image_box
from library.utilities.utilities_process import SCALING_FACTOR, read_image, test_dir


class ImageCleaner:
    """Methods for cleaning images [and rotation, if necessary].  'Cleaning' means 
    applying user-verified masks (QC step) to
    downsampled or full-resolution images
    """


    def create_cleaned_images(self):
        """This method applies the image masks that has been edited by the user to 
        extract the tissue image from the surrounding
        debris
        """

        if self.downsample:
            self.crop_all_images()
            self.create_cleaned_images_thumbnail(channel=self.channel)            
        else:
            self.create_cleaned_images_full_resolution()

    def crop_all_images(self):
        MASKS = self.fileLocationManager.get_thumbnail_masked(channel='C1') # usually channel=1, except for step 6
        maskfiles = sorted(os.listdir(MASKS))
        widths = []
        heights = []
        for maskfile in maskfiles:
            maskpath = os.path.join(MASKS, maskfile)
            mask = read_image(maskpath)
            x1, y1, x2, y2 = get_image_box(mask)
            width = x2 - x1
            height = y2 - y1
            widths.append(width)
            heights.append(height)
        max_width = max(widths)
        max_height = max(heights)
        if self.debug:
            print(f'Updating {self.animal} width={max_width} height={max_height}')
        self.sqlController.update_cropped_data(self.sqlController.scan_run.id, max_width, max_height)
        

    def create_cleaned_images_thumbnail(self, channel):
        """Clean the image using the masks for the downsampled version
        """
        
        CLEANED = self.fileLocationManager.get_thumbnail_cleaned(self.channel)
        INPUT = self.fileLocationManager.get_thumbnail(self.channel)
        MASKS = self.fileLocationManager.get_thumbnail_masked(channel='C1') # usually channel=1, except for step 6
        self.logevent(f"INPUT FOLDER: {INPUT}")
        starting_files = os.listdir(INPUT)
        self.logevent(f"FILE COUNT: {len(starting_files)}")
        self.logevent(f"MASK FOLDER: {MASKS}")
        starting_files = os.listdir(INPUT)
        self.logevent(f"FILE COUNT: {len(starting_files)}")
        self.logevent(f"OUTPUT FOLDER: {CLEANED}")
        os.makedirs(CLEANED, exist_ok=True)
        self.parallel_create_cleaned(INPUT, CLEANED, MASKS)

    def create_cleaned_images_full_resolution(self, channel):
        """Clean the image using the masks for the full resolution image
        """
        
        CLEANED = self.fileLocationManager.get_full_cleaned(self.channel)
        os.makedirs(CLEANED, exist_ok=True)
        INPUT = self.fileLocationManager.get_full(self.channel)
        MASKS = self.fileLocationManager.get_full_masked(channel='C1') #usually channel=1, except for step 6
        self.logevent(f"INPUT FOLDER: {INPUT}")
        starting_files = os.listdir(INPUT)
        self.logevent(f"FILE COUNT: {len(starting_files)}")
        self.logevent(f"MASK FOLDER: {MASKS}")
        starting_files = os.listdir(INPUT)
        self.logevent(f"FILE COUNT: {len(starting_files)}")
        self.logevent(f"OUTPUT FOLDER: {CLEANED}")
        self.parallel_create_cleaned(INPUT, CLEANED, MASKS)

    def parallel_create_cleaned(self, INPUT, CLEANED, MASKS):
        """Do the image cleaning in parallel

        :param INPUT: str of file location input
        :param CLEANED: str of file location output
        :param MASKS: str of file location of masks
        """

        max_width = self.sqlController.scan_run.width
        max_height = self.sqlController.scan_run.height
        if self.downsample:
            max_width = int(max_width / SCALING_FACTOR)
            max_height = int(max_height / SCALING_FACTOR)

        rotation = self.sqlController.scan_run.rotation
        flip = self.sqlController.scan_run.flip
        test_dir(
            self.animal, INPUT, self.section_count, self.downsample, same_size=False
        )

        files = sorted(os.listdir(INPUT))

        file_keys = []
        for file in files:
            infile = os.path.join(INPUT, file)
            outfile = os.path.join(CLEANED, file)  # regular-birdstore
            if os.path.exists(outfile):
                continue
            maskfile = os.path.join(MASKS, file)
            file_keys.append(
                [
                    infile,
                    outfile,
                    maskfile,
                    rotation,
                    flip,
                    max_width,
                    max_height,
                    self.channel,
                ]
            )

        # Cleaning images takes up around 20-25GB per full resolution image
        # so we cut the workers in half here
        workers = self.get_nworkers() // 2
        self.run_commands_concurrently(clean_and_rotate_image, file_keys, workers)

