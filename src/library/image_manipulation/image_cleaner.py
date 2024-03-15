"""This module takes clear of cleaning up the junk from outside
the brain area by using masks.
"""
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from library.database_model.scan_run import FULL_MASK
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
        """

        if self.downsample:
            CLEANED = self.fileLocationManager.get_thumbnail_cleaned(self.channel)
            CROPPED = self.fileLocationManager.get_thumbnail_cropped(self.channel)
            INPUT = self.fileLocationManager.get_thumbnail(self.channel)
            MASKS = self.fileLocationManager.get_thumbnail_masked(channel=1) # usually channel=1, except for step 6
        else:
            CLEANED = self.fileLocationManager.get_full_cleaned(self.channel)
            CROPPED = self.fileLocationManager.get_full_cropped(self.channel)
            INPUT = self.fileLocationManager.get_full(self.channel)
            MASKS = self.fileLocationManager.get_full_masked(channel=1) #usually channel=1, except for step 6

        starting_files = os.listdir(INPUT)
        self.logevent(f"INPUT FOLDER: {INPUT} FILE COUNT: {len(starting_files)} MASK FOLDER: {MASKS}")
        os.makedirs(CLEANED, exist_ok=True)

        self.setup_parallel_create_cleaned(INPUT, CLEANED, MASKS)
        if self.mask_image == FULL_MASK:
            print(f'Updating scan run again')
            self.update_scanrun(self.fileLocationManager.get_thumbnail_cleaned(channel=1))

        self.setup_parallel_place_images(CLEANED)
        

    def setup_parallel_create_cleaned(self, INPUT, CLEANED, MASKS):
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
        test_dir(self.animal, INPUT, self.section_count, self.downsample, same_size=False)
        files = sorted(os.listdir(INPUT))

        file_keys = []
        for file in files:
            infile = os.path.join(INPUT, file)
            outfile = os.path.join(CLEANED, file)
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
                    self.mask_image
                    
                ]
            )

        # Cleaning images takes up around 20-25GB per full resolution image
        # so we cut the workers in half here
        # First clean the image.
        # Crop images and save in cleaned dir
        # Get new width and height from stack of cleaned images
        # Update the DB with the new width and height
        # Place images into the new width and height and save into cropped dir
        workers = self.get_nworkers() // 2
        self.run_commands_concurrently(clean_and_rotate_image, file_keys, workers)


    def setup_parallel_place_images(self, CLEANED):
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

        test_dir(self.animal, CLEANED, self.section_count, self.downsample, same_size=False)
        files = sorted(os.listdir(CLEANED))
        print(f'len cleaned={len(files)}')

        file_keys = []
        for file in files:
            infile = os.path.join(CLEANED, file)
            #outfile = os.path.join(CROPPED, file)
            outfile = infile
            file_keys.append(
                [
                    infile,
                    max_width,
                    max_height,
                ]
            )

        workers = self.get_nworkers() // 2
        self.run_commands_concurrently(place_image, file_keys, workers)

    def get_crop_size(self):
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
            print(f'Updating {self.animal} width={max_width} height={max_height}')
        self.sqlController.update_width_height(self.sqlController.scan_run.id, max_width, max_height)

