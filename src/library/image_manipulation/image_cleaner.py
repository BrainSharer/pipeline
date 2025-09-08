"""This module takes clear of cleaning up the junk from outside
the brain area by using masks.
"""
import os
import inspect
import shutil
import sys
from PIL import Image
from cloudvolume import CloudVolume

Image.MAX_IMAGE_PIXELS = None
import numpy as np
from tqdm import tqdm
from skimage.filters import gaussian
import cv2

from library.image_manipulation.neuroglancer_manager import NumpyToNeuroglancer
from library.image_manipulation.filelocation_manager import ALIGNED_DIR, CLEANED_DIR
from library.image_manipulation.image_manager import ImageManager
from library.utilities.utilities_mask import clean_and_rotate_image, compare_directories, get_image_box, mask_with_contours, place_image, rotate_image
from library.utilities.utilities_process import SCALING_FACTOR, read_image, test_dir, write_image


class ImageCleaner:
    """Methods for cleaning images [and rotation, if necessary].  'Cleaning' means 
    applying user-verified masks (QC step) to
    downsampled or full-resolution images
    """

    def set_max_width_and_height(self):
        width = self.sqlController.scan_run.width
        height = self.sqlController.scan_run.height
        print(f'Width and height before update width={width} height={height}')
        
        inputpath = self.fileLocationManager.get_thumbnail(channel=1)
        _, _, max_width, max_height = test_dir(self.animal, inputpath, self.section_count, downsample=True, same_size=False)

        self.sqlController.update_width_height(self.sqlController.scan_run.id, max_width, max_height, self.scaling_factor)
        width = self.sqlController.scan_run.width
        height = self.sqlController.scan_run.height
        print(f'Width and height after update width={width} height={height}')



    def create_cleaned_images(self):
        """This method applies the image masks that has been edited by the user to 
        extract the tissue image from the surrounding
        debris
        1. Set up the mask, input and output directories
        2. clean images
        3. Get biggest box size from all contours from all files and update DB with that info
        4. Place images in image size with correct background color
        """

        if self.downsample:
            INPUT = self.fileLocationManager.get_thumbnail(self.channel)
            MASKS = self.fileLocationManager.get_thumbnail_masked(channel=1)
        else:
            INPUT = self.fileLocationManager.get_full(self.channel)
            MASKS = self.fileLocationManager.get_full_masked(channel=1)

        CLEANED = self.fileLocationManager.get_directory(self.channel, self.downsample, inpath=CLEANED_DIR)

        compare_directories(INPUT, MASKS)

        try:
            starting_files = os.listdir(INPUT)
        except OSError:
            print(f"Error: Could not find the input directory: {INPUT}")
            return

        current_function_name = inspect.currentframe().f_code.co_name
        self.fileLogger.logevent(f"{self.__class__.__name__}::{current_function_name} Input FOLDER: {INPUT} FILE COUNT: {len(starting_files)} MASK FOLDER: {MASKS}")
        if self.downsample and os.path.exists(CLEANED):
            print(f'Removing {CLEANED}')
            shutil.rmtree(CLEANED)        
        os.makedirs(CLEANED, exist_ok=True)
        self.parallel_create_cleaned(INPUT, CLEANED, MASKS)


    def parallel_create_cleaned(self, INPUT, CLEANED, MASKS):
        """Do the image cleaning in parallel

        :param INPUT: str of file location input
        :param CLEANED: str of file location output
        :param MASKS: str of file location of masks
        """
        image_manager = ImageManager(INPUT)
        self.bgcolor = image_manager.get_bgcolor()
        print(f'Background color for cleaning is {self.bgcolor} for animal {self.animal} channel {self.channel} downsample={self.downsample}')

        max_width = self.sqlController.scan_run.width
        max_height = self.sqlController.scan_run.height
        if self.downsample:
            max_width = int(np.round(max_width / self.scaling_factor))
            max_height = int(np.round(max_height / self.scaling_factor))

        rotation = self.sqlController.scan_run.rotation
        flip = self.sqlController.scan_run.flip
        test_dir(self.animal, INPUT, self.section_count, self.downsample, same_size=False)
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
                    self.bgcolor
                ]
            )

        # Cleaning images takes up around 20-25GB per full resolution image
        # so we cut the workers in half here
        workers = self.get_nworkers() // 2
        if self.debug:
            print(f'len of file keys in parallel create cleaned={len(file_keys)}')
            for file_key in file_keys:
                clean_and_rotate_image(file_key)
        else:
            self.run_commands_concurrently(clean_and_rotate_image, file_keys, workers)


    def create_rotated_aligned_masks(self):

        def cleanup(dirs):
            for output_dir in output_dirs:
                test_dir = self.fileLocationManager.get_directory(self.channel, self.downsample, inpath=output_dir)
                if os.path.exists(test_dir):
                    print(f'Removing {test_dir}')
                    shutil.rmtree(test_dir)
        self.maskpath = self.fileLocationManager.get_thumbnail_masked(channel=1) # usually channel=1, except for step 6
        maskfiles = sorted(os.listdir(self.maskpath))
        rotation = self.sqlController.scan_run.rotation
        flip = self.sqlController.scan_run.flip
        max_width = self.sqlController.scan_run.width
        max_height = self.sqlController.scan_run.height
        max_width = int(max_width / SCALING_FACTOR)
        max_height = int(max_height / SCALING_FACTOR)
        bgcolor = 0

        # Clean up
        output_dirs = ['mask_placed', 'mask_placed_aligned_0', 'mask_placed_aligned']
        cleanup(output_dirs)

        self.output = self.fileLocationManager.get_directory(self.channel, self.downsample, inpath='placed_mask')
        os.makedirs(self.output, exist_ok=True)

        for maskfile in maskfiles:
            maskpath = os.path.join(self.maskpath, maskfile)
            outfile = os.path.join(self.output, maskfile)
            if os.path.exists(outfile):
                continue
            mask = read_image(maskpath)
            if rotation > 0:
                cleaned = rotate_image(mask, maskpath, rotation)
            else:
                cleaned = mask
            # flip = switch top to bottom
            # flop = switch left to right
            if flip == "flip":
                cleaned = np.flip(cleaned, axis=0)
            if flip == "flop":
                cleaned = np.flip(cleaned, axis=1)
            del mask

            placed_img = place_image(cleaned, maskpath, max_width, max_height, bgcolor)
            del cleaned

            message = f'Error in saving {outfile} with shape {placed_img.shape} img type {placed_img.dtype}'
            write_image(outfile, placed_img, message=message)
        ##### now align images iteration 0
        self.input = self.output
        self.files = os.listdir(self.input)
        self.output = self.fileLocationManager.get_directory(self.channel, self.downsample, inpath='mask_placed_aligned_0')
        os.makedirs(self.output, exist_ok=True)
        self.iteration = 0
        self.start_image_alignment()
        #compare_directories(self.fileLocationManager.get_directory(self.channel, self.downsample, inpath=ALIGNED_DIR), self.output)
        ##### now align images iteration 1
        self.input = self.output
        self.files = os.listdir(self.input)
        self.output = self.fileLocationManager.get_directory(self.channel, self.downsample, inpath='mask_placed_aligned')
        os.makedirs(self.output, exist_ok=True)
        self.iteration = 1
        self.start_image_alignment()

        # Test the placed_aligned images
        # Cleanup
        output_dirs = ['placed', 'mask_placed_aligned_0']
        cleanup(output_dirs)
        

    def create_shell_from_mask(self):
        self.create_rotated_aligned_masks()
        ##### now align images
        self.input = self.output
        self.files = os.listdir(self.input)
        self.output = self.fileLocationManager.get_directory(self.channel, self.downsample, inpath='placed_aligned')
        os.makedirs(self.output, exist_ok=True)
        self.iteration = 0
        self.start_image_alignment()
        self.files = sorted(os.listdir(self.output))
        file_list = []
        for file in tqdm(self.files):
            filepath = os.path.join(self.output, file)
            farr = read_image(filepath)
            file_list.append(farr)
        volume = np.stack(file_list, axis = 0)
        volume = np.swapaxes(volume, 0, 2) # put it in x,y,z format
        volume = gaussian(volume, 1)  # this is a float array
        volume[volume > 0] = 255
        volume = volume.astype(np.uint8)
        ids, counts = np.unique(volume, return_counts=True)
        data_type = volume.dtype
        xy = self.sqlController.scan_run.resolution * 1000 * 1000 / self.scaling_factor
        z = self.sqlController.scan_run.zresolution * 1000
        scales = (int(xy), int(xy), int(z))
        chunks = [64, 64, 64]
        
        print(f'Volume shape={volume.shape} dtype={volume.dtype} chunks at {chunks} and scales with {scales}')
        print(f'IDS={ids}')
        print(f'counts={counts}')
        
        
        ng = NumpyToNeuroglancer(self.animal, volume, scales, layer_type='segmentation', 
            data_type=data_type, chunk_size=chunks)
        self.mesh_dir = os.path.join(self.fileLocationManager.neuroglancer_data, 'shell')
        self.layer_path = f'file://{self.mesh_dir}'

        ng.init_volume(self.mesh_dir)
        
        # This calls the igneous create_transfer_tasks
        #ng.add_rechunking(MESH_DIR, chunks=chunks, mip=0, skip_downsamples=True)

        #tq = LocalTaskQueue(parallel=4)
        cloudpath2 = f'file://{self.mesh_dir}'
        #ng.add_downsampled_volumes(chunk_size = chunks, num_mips = 1)

        ##### add segment properties
        print('Adding segment properties')
        cv2 = CloudVolume(cloudpath2, 0)
        segment_properties = {str(id): str(id) for id in ids}
        ng.add_segment_properties(cv2, segment_properties)

        ##### first mesh task, create meshing tasks
        print(f'Creating meshing tasks on volume from {cloudpath2}')
        ##### first mesh task, create meshing tasks
        ng.add_segmentation_mesh(cv2.layer_cloudpath, mip=0)

    def mask_aligned_imageNOTUSED(self, img, file):
        #from scipy.ndimage import binary_fill_holes

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray[gray > 0] = 255
        #return gray
        # If the pixel value is smaller than or 
        # equal to the threshold, it is set to 0, otherwise it is set to a maximum value        
        
        #thresh = cv2.threshold(gray, threshold, maxval, cv2.THRESH_BINARY)[1]
        #ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY )
        #return thresh
        # blur threshold image
        #new_img = new_img.astype(np.uint8)
        #new_img[(new_img > 200)] = 0
        #lowerbound = 0
        #upperbound = 255
        # all pixels value above lowerbound will  be set to upperbound
        #_, thresh = cv2.threshold(new_img.copy(), lowerbound, upperbound, cv2.THRESH_BINARY_INV)
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        #thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        #smoothed = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel)
        #inverted_thresh = cv2.bitwise_not(smoothed)
        #filled_thresh = binary_fill_holes(thresh).astype(np.uint8)
        #return thresh
        #return cv2.bitwise_and(img, img, mask=filled_thresh)



        #blur = cv2.GaussianBlur(thresh, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
        #return blur.astype(np.uint8)
        # stretch so that 255 -> 255 and 127.5 -> 0
        # threshold again
        #thresh2 = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY)[1]
        #ret, thresh = cv2.threshold(gray, 1, 255, 0)
        # get external contour
        #contours, xxx = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours = np.array(xxx)
        #im = np.copy(img)
        #cv2.drawContours(thresh, contours, -1, 255, 8)
        #cv2.fillPoly(thresh, pts = [contours.astype(np.int32)], color = 0)

        #points = contours.astype(np.int32)
        #print(f'points shape={points.shape} type={points.dtype}')
        #cv2.fillPoly(thresh, pts = [points], color = 255)
        res = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = res[-2] # for cv2 v3 and v4+ compatibility
        gray = np.zeros_like(gray)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area > 3000:
                cv2.putText(gray, str(int(area)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
                #print(f'file={file} filling area={area} type {type(area)}')
                cv2.polylines(gray, [contour], isClosed=True, color=255, thickness=10)            
                cv2.fillPoly(gray, pts=[contour], color=255)   


        return gray.astype(np.uint8)

    def create_shell(self):
        WHITE = 255
        WRITE_MASKS = False
        iteration = self.get_alignment_status()
        if iteration is None:
            print('No alignment iterations found.  Please run the alignment steps first.')
            return
        input = self.fileLocationManager.get_directory(channel=self.channel, downsample=True, inpath=ALIGNED_DIR)
          

        if WRITE_MASKS:
            output = self.fileLocationManager.get_directory(self.channel, self.downsample, inpath='masked_aligned')
            os.makedirs(output, exist_ok=True)
        print('with input =', input)
        files = sorted(os.listdir(input))
        file_list = []
        for file in tqdm(files, disable=False):
            filepath = os.path.join(input, file)
            if WRITE_MASKS:
                outpath = os.path.join(self.output, file)
                if os.path.exists(outpath):
                    continue
            img = read_image(filepath)
            if img.ndim == 3:
                img = mask_with_contours(img)
                img = self.mask_aligned_image(img, file)
                if WRITE_MASKS:
                    write_image(outpath, img)
            else:
                img[img > 0] = WHITE
            file_list.append(img)
        volume = np.stack(file_list, axis = 0)
        volume = np.swapaxes(volume, 0, 2) # put it in x,y,z format
        volume = gaussian(volume, 1)  # this is a float array
        volume[volume > 0] = WHITE
        volume = volume.astype(np.uint8)
        ids = list(np.unique(volume, return_counts=False))
        data_type = volume.dtype
        xy = self.sqlController.scan_run.resolution * 1000 * 1000 / self.scaling_factor
        z = self.sqlController.scan_run.zresolution * 1000
        scales = (int(xy), int(xy), int(z))
        chunks = [64, 64, 64]
        
        print(f'Volume shape={volume.shape} dtype={volume.dtype} chunks at {chunks} and scales with {scales}nm')
        
        
        ng = NumpyToNeuroglancer(self.animal, volume, scales, layer_type='segmentation', 
            data_type=data_type, chunk_size=chunks)
        self.mesh_dir = os.path.join(self.fileLocationManager.neuroglancer_data, 'shell')
        self.layer_path = f'file://{self.mesh_dir}'

        ng.init_volume(self.mesh_dir)
        
        # This calls the igneous create_transfer_tasks
        #ng.add_rechunking(MESH_DIR, chunks=chunks, mip=0, skip_downsamples=True)

        #tq = LocalTaskQueue(parallel=4)
        cloudpath2 = f'file://{self.mesh_dir}'
        #ng.add_downsampled_volumes(chunk_size = chunks, num_mips = 1)

        ##### add segment properties
        print('Adding segment properties')
        cv2 = CloudVolume(cloudpath2, 0)
        segment_properties = {str(id): str(id) for id in ids}
        ng.add_segment_properties(cv2, segment_properties)

        ##### first mesh task, create meshing tasks
        print(f'Creating meshing tasks on volume from {cloudpath2}')
        ##### first mesh task, create meshing tasks
        ng.add_segmentation_mesh(cv2.layer_cloudpath, mip=0)
