"""This module takes clear of cleaning up the junk from outside
the brain area by using masks.
"""
import os
import sys
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from library.image_manipulation.image_manager import ImageManager
from library.utilities.utilities_mask import clean_and_rotate_image, clean_rotate_and_place_image, get_image_box, place_image
from library.utilities.utilities_process import delete_in_background, SCALING_FACTOR, read_image, test_dir, get_scratch_dir, get_directory_size

#FOR BACKGROUND WRITING IMAGE & SCRATCH
import shutil
import subprocess



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
        
        #CLEANUP NESTED IF STATEMENTS
        if self.downsample:
            self.input = self.fileLocationManager.get_thumbnail(self.channel)
            input_aggregate_size_in_bytes  = get_directory_size(self.input)
            final_output = self.fileLocationManager.get_thumbnail_cleaned(self.channel)
            staging_output = final_output
            if self.use_scratch:
                scratch_tmp = get_scratch_dir()
                _, _, available_space_in_bytes = shutil.disk_usage(scratch_tmp)
                if input_aggregate_size_in_bytes*3 < available_space_in_bytes: #THERE IS ENOUGH ROOM ON SCRATCH FOR CLEAN,PLACE IMAGES
                    SCRATCH = os.path.join(scratch_tmp, 'pipeline', self.animal, 'masks')
                    cleaned_folder_name = os.path.basename(final_output)
                    staging_output = os.path.join(SCRATCH, cleaned_folder_name)
                else:
                    self.use_scratch = False
            self.maskpath = self.fileLocationManager.get_thumbnail_masked(channel=1) # usually channel=1, except for step 6
            self.cropped_output = self.fileLocationManager.get_thumbnail_cropped(self.channel)
        else:
            self.input = self.fileLocationManager.get_full(self.channel)
            input_aggregate_size_in_bytes  = get_directory_size(self.input)
            final_output = self.fileLocationManager.get_full_cleaned(self.channel)
            staging_output = final_output
            if self.use_scratch:
                scratch_tmp = get_scratch_dir()
                _, _, available_space_in_bytes = shutil.disk_usage(scratch_tmp)
                if input_aggregate_size_in_bytes*3 < available_space_in_bytes: #THERE IS ENOUGH ROOM ON SCRATCH FOR CLEAN,PLACE IMAGES
                    SCRATCH = os.path.join(scratch_tmp, 'pipeline', self.animal, 'masks')
                    cleaned_folder_name = os.path.basename(final_output)
                    staging_output = os.path.join(SCRATCH, cleaned_folder_name)
                else:
                    self.use_scratch = False
            self.maskpath = self.fileLocationManager.get_full_masked(channel=1) #usually channel=1, except for step 6
            self.cropped_output = self.fileLocationManager.get_full_cropped(self.channel)

        #######################################################
        # PARAMETER SUMMARY
        print('*'*50, '\nPARAMETER SUMMARY')
        print(f'SRC IMG LOCATION: {self.input}')
        print(f'IMG MASK [INPUT] LOCATION: {self.maskpath}')
        print(f'FINAL CLEANED OUTPUT DIR={final_output}')
        print(f'FINAL CROPPED OUTPUT DIR={self.cropped_output}')
        print(f'CLEANED STAGING DIR={staging_output}')
        print(f'USING SCRATCH: {self.use_scratch}')
        #######################################################
        
        try:
            starting_files = os.listdir(self.input)
        except OSError:
            print(f"Error: Could not find the input directory: {self.input}")
            return
        self.fileLogger.logevent(f"image_cleaner::create_cleaned_images Input FOLDER: {self.input} FILE COUNT: {len(starting_files)} MASK FOLDER: {self.maskpath}")

        os.makedirs(staging_output, exist_ok=True)
        image_manager = ImageManager(self.input)
        if self.mask_image > 0: 
            self.bgcolor = image_manager.get_bgcolor()
        else:
            self.bgcolor = 0
        self.setup_parallel_create_cleaned(staging_output, final_output)
        
        # Update the scan run with the cropped width and height. The images are also rotated and/or flipped at this point. 
        if self.mask_image > 0 and self.channel == 1 and self.downsample:
            self.set_crop_size()
            if self.debug:
                print(f'Updating scan run. and set bgcolor to {self.bgcolor}')

            self.setup_parallel_place_images(staging_output) #ONLY APPLIES PAST 1ST RUN
        

    def setup_parallel_create_cleaned(self, staging_output, final_output):
        """Do the image cleaning in parallel
        """

        rotation = self.sqlController.scan_run.rotation
        flip = self.sqlController.scan_run.flip
        files, _, max_width, max_height = test_dir(self.animal, self.input, self.section_count, self.downsample, same_size=False)
        
        file_keys = []
        for file in files:
            infile = os.path.join(self.input, file)
            outfile = os.path.join(staging_output, file)
            cropped_staging_output = os.path.join(self.cropped_output, file)
            cropped_final_output = os.path.dirname(cropped_staging_output)

            if self.use_scratch:
                cleaned_path = os.path.dirname(staging_output)
                dir_path = os.path.dirname(cropped_staging_output)
                cropped_folder_name = os.path.basename(dir_path)
                cropped_staging_output = os.path.join(cleaned_path, cropped_folder_name)

            #######################################################
            if self.debug:
                print(f'CROPPED STAGING DIR={cropped_staging_output}')
                print('*'*50, '\n')
            #######################################################

            if os.path.exists(outfile): #SKIP IF FILE EXISTS IN STAGING DIR
                continue
            if os.path.exists(os.path.join(final_output, os.path.basename(outfile))): #SKIP IF FILE EXISTS IN FINAL OUTPUT DIR
                continue
            maskfile = os.path.join(self.maskpath, file)

            file_keys.append(
                [
                    infile,
                    outfile,
                    maskfile,
                    rotation,
                    flip,
                    self.mask_image,
                    self.bgcolor,
                    self.channel,
                    self.debug,
                    max_width,
                    max_height,
                    cropped_staging_output
                ]
            )
        
        # Cleaning images takes up around 20-25GB per full resolution image
        # so we cut the workers in half here
        # The method below will clean and crop. It will also rotate and flip if necessary
        # It then writes the files to the clean dir. They are not padded at this point.
        workers = self.get_nworkers() // 2
        self.run_commands_concurrently(clean_and_rotate_image, file_keys, workers)

        if self.use_scratch:
            print(f'MOVE STAGING FILES [IN BACKGROUND] TO FINAL OUTPUT DIR (cleaned): {staging_output} -> {final_output}')
            print(f'MOVE STAGING FILES [IN BACKGROUND] TO FINAL OUTPUT DIR (cropped): {cropped_staging_output} -> {cropped_final_output}')

            if not os.path.exists(final_output):#MOVE 'cleaned' DIR TO FINAL OUTPUT (IF SCRATCH USED & NOT EXISTS)
                print(f'Moving {staging_output} to {final_output}')
                os.makedirs(final_output, exist_ok=True)
                # Use rclone to move the directory
                subprocess.run(["rclone", "move", staging_output, final_output], check=True)
            else:# FOLDER EXISTS, BUT CHECK INDIVIDUAL FILES (STRAGLERS)
                for filename in os.listdir(staging_output):
                    source_file = os.path.join(staging_output, filename)
                    destination_file = os.path.join(final_output, filename)
                    if not os.path.exists(destination_file):
                        print(f"Copying {filename} to destination...")
                        shutil.copy2(source_file, destination_file)

            if not os.path.exists(cropped_final_output):#COPY 'cropped' DIR TO FINAL OUTPUT (IF SCRATCH USED & NOT EXISTS)
                print(f'Moving {cropped_staging_output} to {cropped_final_output}')
                os.makedirs(cropped_final_output, exist_ok=True)
                # Use rclone to move the directory
                subprocess.run(["rclone", "copy", cropped_staging_output, cropped_final_output], check=True)
            else:# FOLDER EXISTS, BUT CHECK INDIVIDUAL FILES (STRAGLERS)
                for filename in os.listdir(cropped_staging_output):
                    source_file = os.path.join(cropped_staging_output, filename)
                    destination_file = os.path.join(cropped_final_output, filename)
                    if not os.path.exists(destination_file):
                        print(f"Copying {filename} to destination...")
                        shutil.copy2(source_file, destination_file)
            
    
    def setup_parallel_place_images(self, staging_output):
        """Do the image placing in parallel. Cleaning and cropping has already taken place.
        We first need to get all the correct image sizes and then update the DB.
        """
        if self.downsample:
            #self.input = self.fileLocationManager.get_thumbnail_cleaned(self.channel)
            self.input = staging_output
            self.output = self.fileLocationManager.get_thumbnail_cropped(self.channel)
        else:
            #self.input = self.fileLocationManager.get_full_cleaned(self.channel)
            self.input = staging_output
            self.output = self.fileLocationManager.get_full_cropped(self.channel)

        os.makedirs(self.output, exist_ok=True)
        max_width = self.sqlController.scan_run.width
        max_height = self.sqlController.scan_run.height
        if self.downsample:
            max_width = int(max_width / SCALING_FACTOR)
            max_height = int(max_height / SCALING_FACTOR)

        if max_width == 0 or max_height == 0:
            print(f'Error in setup parallel place images: width or height is 0. width={max_width} height={max_height}')
            sys.exit()

        files, cnt_files, max_width, max_height = test_dir(self.animal, self.input, self.section_count, self.downsample, same_size=False)
        #files = sorted(os.listdir(self.input))

        file_keys = []
        for file in files:
            outfile = os.path.join(self.output, file)
            if not os.path.exists(outfile):
                infile = os.path.join(self.input, file)
                file_keys.append((infile, outfile, max_width, max_height, self.bgcolor))

        if self.debug:
            print(f'len of file keys in place={len(file_keys)}')
        workers = self.get_nworkers() // 2
        self.run_commands_concurrently(place_image, tuple(file_keys), workers)

    def set_crop_size(self):
        self.maskpath = self.fileLocationManager.get_thumbnail_masked(channel=1) # usually channel=1, except for step 6
        maskfiles = sorted(os.listdir(self.maskpath))
        widths = []
        heights = []
        for maskfile in maskfiles:
            maskpath = os.path.join(self.maskpath, maskfile)
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
            self.maskpath = self.fileLocationManager.get_thumbnail_masked(channel=1)
            self.input = self.fileLocationManager.get_thumbnail_cleaned(self.channel)

            image_manager = ImageManager(self.input, self.maskpath)
            update_dict = {'bgcolor': image_manager.get_bgcolor() }
            self.sqlController.update_scan_run(self.sqlController.scan_run.id, update_dict)
