"""This module takes clear of cleaning up the junk from outside
the brain area by using masks.
"""
import os
import sys
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from library.image_manipulation.image_manager import ImageManager
from library.utilities.utilities_mask import clean_and_rotate_image, get_image_box, place_image
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
        1. Set up the mask, input and output directories
        2. clean images
        3. Crop images if mask is set to FULL_MASK
        4. Get biggest box size from all contours from all files and update DB with that info
        5. Place images in new cropped image size with correct background color
        """

        if self.downsample:
            self.output = self.fileLocationManager.get_thumbnail_cleaned(self.channel)
            self.input = self.fileLocationManager.get_thumbnail(self.channel)
            MASKS = self.fileLocationManager.get_thumbnail_masked(channel=1) # usually channel=1, except for step 6
        else:
            self.output = self.fileLocationManager.get_full_cleaned(self.channel)
            self.input = self.fileLocationManager.get_full(self.channel)
            MASKS = self.fileLocationManager.get_full_masked(channel=1) #usually channel=1, except for step 6

        starting_files = os.listdir(self.input)
        self.logevent(f"image_cleaner::create_cleaned_images Input FOLDER: {self.input} FILE COUNT: {len(starting_files)} MASK FOLDER: {MASKS}")
        os.makedirs(self.output, exist_ok=True)

        self.setup_parallel_create_cleaned(MASKS)
        # Update the scan run with the cropped width and height. The images are also rotated and/or flipped at this point. 
        if self.debug:
            print(f'Updating scan run.')
        self.set_crop_size()
        self.setup_parallel_place_images()
        

    def setup_parallel_create_cleaned(self, MASKS):
        """Do the image cleaning in parallel
        :param MASKS: str of file location of masks
        """

        rotation = self.sqlController.scan_run.rotation
        flip = self.sqlController.scan_run.flip
        test_dir(self.animal, self.input, self.section_count, self.downsample, same_size=False)
        files = sorted(os.listdir(self.input))

        file_keys = []
        for file in files:
            infile = os.path.join(self.input, file)
            outfile = os.path.join(self.output, file)
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
                    self.mask_image,
                    self.downsample
                ]
            )

        # Cleaning images takes up around 20-25GB per full resolution image
        # so we cut the workers in half here
        # The method below will clean and crop. It will also rotate and flip if necessary
        # It then writes the files to the clean dir. They are not padded at this point.
        workers = self.get_nworkers() // 2
        self.run_commands_concurrently(clean_and_rotate_image, file_keys, workers)



    def setup_parallel_place_images(self):
        """Do the image placing in parallel. Cleaning and cropping has already taken place.
        We first need to get all the correct image sizes and then update the DB.
        """
        if self.downsample:
            self.input = self.fileLocationManager.get_thumbnail_cleaned(self.channel)
            self.output = self.fileLocationManager.get_thumbnail_cropped(self.channel)
        else:
            self.input = self.fileLocationManager.get_full_cleaned(self.channel)
            self.output = self.fileLocationManager.get_full_cropped(self.channel)

        os.makedirs(self.output, exist_ok=True)
        max_width = self.sqlController.scan_run.width
        max_height = self.sqlController.scan_run.height
        if self.downsample:
            max_width = int(max_width / SCALING_FACTOR)
            max_height = int(max_height / SCALING_FACTOR)

        if max_width == 0 or max_height == 0:
            print(f'Error: width or height is 0. width={max_width} height={max_height}')
            sys.exit()

        test_dir(self.animal, self.input, self.section_count, self.downsample, same_size=False)
        files = sorted(os.listdir(self.input))

        file_keys = []
        for file in files:
            infile = os.path.join(self.input, file)
            outfile = os.path.join(self.output, file)
            if os.path.exists(outfile):
                continue
            file_keys.append([infile, outfile, max_width, max_height])

        if self.debug:
            print(f'len of file keys in place={len(file_keys)}')
        workers = self.get_nworkers() // 2
        self.run_commands_concurrently(place_image, file_keys, workers)

    def set_crop_size(self):
        MASKS = self.fileLocationManager.get_thumbnail_masked(channel=1) # usually channel=1, except for step 6
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
            print(f'Updating {self.animal} scan_run with width={max_width} height={max_height}')
        self.sqlController.update_width_height(self.sqlController.scan_run.id, max_width, max_height)

    def update_bg_color(self):
            """
            Updates the background color of the image.

            This method retrieves the background color of the image using the ImageManager class,
            and then updates the corresponding field in the scan run table using the SQLController class.

            Parameters:
                None

            Returns:
                None
            """
            MASKS = self.fileLocationManager.get_thumbnail_masked(channel=1)
            self.input = self.fileLocationManager.get_thumbnail_cleaned(self.channel)

            image_manager = ImageManager(self.input, MASKS)
            update_dict = {'bgcolor': image_manager.get_bgcolor() }
            self.sqlController.update_scan_run(self.sqlController.scan_run.id, update_dict)
