"""This module takes clear of cleaning up the junk from outside
the brain area by using masks.
"""
import os
import inspect
import shutil
from PIL import Image
from cloudvolume import CloudVolume
from pathlib import Path

import torch
import torchvision

Image.MAX_IMAGE_PIXELS = None
import numpy as np
from tqdm import tqdm
from skimage.filters import gaussian
import cv2

from library.image_manipulation.neuroglancer_manager import NumpyToNeuroglancer
from library.image_manipulation.filelocation_manager import ALIGNED_DIR, CLEANED_DIR
from library.image_manipulation.image_manager import ImageManager
from library.utilities.utilities_mask import clean_and_rotate_image, compare_directories, merge_mask, place_image, rotate_image
from library.utilities.utilities_process import SCALING_FACTOR, read_image, test_dir, write_image

class ImageCleaner:
    """Methods for cleaning images [and rotation, if necessary].  'Cleaning' means 
    applying user-verified masks (QC step) to
    downsampled or full-resolution images
    """

    def set_max_width_and_height(self):
        current_width = self.sqlController.scan_run.width
        current_height = self.sqlController.scan_run.height

        if current_width == 0 or current_height == 0:
            inputpath = self.fileLocationManager.get_thumbnail(channel=1)
            _, _, max_width, max_height = test_dir(self.animal, inputpath, self.section_count, downsample=True, same_size=False)
            self.sqlController.update_width_height(self.sqlController.scan_run.id, max_width, max_height, self.scaling_factor)
            width = self.sqlController.scan_run.width
            height = self.sqlController.scan_run.height
            print(f'Width and height after update width={width} height={height}')
        else:
            print(f'Width and height already set width={current_width} height={current_height}')


    def create_cleaned_images(self):
        """This method applies the image masks that has been edited by the user to 
        extract the tissue image from the surrounding
        debris
        1. Set up the mask, input and output directories
        2. clean images
        3. Get biggest box size from all contours from all files and update DB with that info
        4. Place images in image size with correct background color
        """

        if self.debug:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} START")

        if self.downsample:
            INPUT = self.fileLocationManager.get_thumbnail(self.channel)
            MASKS = self.fileLocationManager.get_thumbnail_masked(channel=1)
        else:
            INPUT = self.fileLocationManager.get_full(self.channel)
            MASKS = self.fileLocationManager.get_full_masked(channel=1)

        CLEANED = self.fileLocationManager.get_directory(self.channel, self.downsample, inpath=CLEANED_DIR)
        if self.downsample:
            print(f'Cleaning: {CLEANED}')
            shutil.rmtree(CLEANED, ignore_errors=True)
        os.makedirs(CLEANED, exist_ok=True)

        #15-SEP-2025 testing - Duane
        compare_directories(INPUT, MASKS)

        try:
            starting_files = os.listdir(INPUT)
        except OSError:
            print(f"Error: Could not find the input directory: {INPUT}")
            return

        self.fileLogger.logevent(f"INPUT FOLDER: {INPUT}, QTY FILES: {len(starting_files)}")
        self.fileLogger.logevent(f"MASK FOLDER: {MASKS}")
        self.fileLogger.logevent(f"CLEANED [OUTPUT] FOLDER: {CLEANED}")
                        
        self.parallel_create_cleaned(INPUT, CLEANED, MASKS)


    def parallel_create_cleaned(self, input_path: str, cleaned_path: Path, masks_path: str):
        """Do the image cleaning in parallel

        :param INPUT: str of file location input
        :param CLEANED: str of file location output
        :param MASKS: str of file location of masks
        """
        if self.debug:
            current_function_name = inspect.currentframe().f_code.co_name
            print(f"DEBUG: {self.__class__.__name__}::{current_function_name} START")

        image_manager = ImageManager(input_path)
        self.bgcolor = image_manager.get_bgcolor()
        print(f'Background color for cleaning is {self.bgcolor} for animal {self.animal} channel {self.channel} downsample={self.downsample}')

        max_width = self.sqlController.scan_run.width
        max_height = self.sqlController.scan_run.height
        if self.downsample:
            max_width = int(np.round(max_width / self.scaling_factor))
            max_height = int(np.round(max_height / self.scaling_factor))
        print(f'Using max width={max_width} and max height={max_height} for placing cleaned images')

        rotation = self.sqlController.scan_run.rotation
        flip = self.sqlController.scan_run.flip
        test_dir(self.animal, input_path, self.section_count, self.downsample, same_size=False)
        files = sorted(os.listdir(input_path ))
        
        file_keys = []
        for file in files:
            infile = Path(input_path, file)
            outfile = Path(cleaned_path, file)  # regular-birdstore
            if os.path.exists(outfile):
                continue
            maskfile = Path(masks_path, file)
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

    def create_hollow_shell(self, image_path):
        transform = torchvision.transforms.ToTensor()

        threshold = 0.975
        arr = read_image(image_path)
        if arr.dtype == np.uint8:
            # 8-bit max value is 255
            img_float = arr.astype(np.float32) / 255.0
        elif arr.dtype in [np.uint16, np.int16]:
            # 16-bit max value is 65535
            img_float = arr.astype(np.float32) / 65535.0
            arr = arr.astype(np.uint8)  # Convert to 8-bit for OpenCV operations
        else:
            # If already float or another type, handle accordingly
            img_float = arr.astype(np.float32)
            
        img = Image.fromarray(img_float)
        
        torch_input = transform(img)
        torch_input = torch_input.unsqueeze(0)
        self.loaded_model.eval()
        with torch.no_grad():
            pred = self.loaded_model(torch_input)

        raw_masks = pred[0]["masks"]  # Shape is usually (N, 1, H, W)

        if raw_masks.shape[0] == 0:
            print(f'No masks detected for {image_path}. using alterate method.')
            _, mask = cv2.threshold(arr, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Squeeze only the channel dimension (dim 1), keeping the object count
            # Then take the first detected object [0]
            binary_masks = (raw_masks > threshold).squeeze(1) # Shape: (N, H, W)
            mask = binary_masks[0].detach().cpu().numpy()     # Shape: (H, W)

        binary = mask.astype(np.uint8)
        binary[binary > 0] = 255
        if self.debug:
            print(f'Binary mask shape={binary.shape} dtype={binary.dtype} unique values={np.unique(binary)}')
            return binary
        # 1. Define a structuring element (kernel)
        # A larger size (e.g., 5x5 or 7x7) increases the effect
        kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        # 2. Erode the mask (shrinks the white shape, removes small noise)
        # Increase iterations to erode more aggressively
        eroded_mask = cv2.erode(binary, kernel, iterations=2)

        # 3. Smooth the main mask
        # Using MORPH_CLOSE fills in small holes/gaps inside the main shape
        # Using MORPH_OPEN removes ragged edges on the outside
        binary = cv2.morphologyEx(eroded_mask, cv2.MORPH_CLOSE, kernel)
        min_area = 100
        ratio_threshold = 0.1

        all_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)
        top_contours = sorted_contours[:4]
        valid_contours = [c for c in top_contours if cv2.contourArea(c) >= min_area]
        largest_area = cv2.contourArea(top_contours[0])
        filtered_contours = []
        
        for c in valid_contours:
            current_area = cv2.contourArea(c)
            # Compare current contour size to the largest contour size
            if (current_area / largest_area) >= ratio_threshold:
                filtered_contours.append(c)
            else:
                # Since they are sorted, if one fails the ratio check, the remaining smaller ones will too
                break
            
        # Create a completely black background of the same size
        output_mask = np.zeros_like(binary)
        # Draw the white border
        # -1 draws all found contours (or you can select the largest one if there's noise)
        # thickness=2 sets the line width of the white border; adjust as needed
        cv2.drawContours(output_mask, filtered_contours, -1, (255), thickness=8)
        return output_mask


    def create_hollow_shellXXX(self, image_path):


        # 1. Load the sagittal histology TIF image
        # Read as grayscale since we only need the structure for segmentation
        # This does not work with the MD brains.
        min_area = 100
        ratio_threshold = 0.1
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        binary = img.astype(np.uint8)
        _, binary = cv2.threshold(binary, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, big_kernel)  # Fills small holes
        all_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)
        top_contours = sorted_contours[:4]
        valid_contours = [c for c in top_contours if cv2.contourArea(c) >= min_area]
        largest_area = cv2.contourArea(top_contours[0])
        filtered_contours = []
        
        for c in valid_contours:
            current_area = cv2.contourArea(c)
            # Compare current contour size to the largest contour size
            if (current_area / largest_area) >= ratio_threshold:
                filtered_contours.append(c)
            else:
                # Since they are sorted, if one fails the ratio check, the remaining smaller ones will too
                break
            
        # Create a completely black background of the same size
        output_mask = np.zeros_like(img)
        # Draw the white border
        # -1 draws all found contours (or you can select the largest one if there's noise)
        # thickness=2 sets the line width of the white border; adjust as needed
        cv2.drawContours(output_mask, filtered_contours, -1, (255), thickness=8)
        blur_size = 7  # Must be an odd number. Higher = smoother/more rounded
        blurred = cv2.GaussianBlur(output_mask, (blur_size,blur_size), sigmaX=blur_size, sigmaY=blur_size)

        # 2. Threshold again to snap the soft blur back into a sharp, solid white line
        _, smoothed_blur = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)        

        # 6. Save the final image
        return smoothed_blur
    


    def create_aligned_masks(self):

        iteration = self.get_alignment_status()
        if iteration is None:
            print('No alignment iterations found.  Please run the alignment steps first.')
            return
        input_dir = self.fileLocationManager.get_directory(channel=self.channel, downsample=True, inpath=ALIGNED_DIR)
          
        output_dir = self.fileLocationManager.get_directory(self.channel, self.downsample, inpath='masked_aligned')
        os.makedirs(output_dir, exist_ok=True)
        print('with input =', input_dir)
        print('with output =', output_dir)
        files = sorted(os.listdir(input_dir))
        ### setup model
        self.load_machine_learning_model()

        ### done with model setup
        for file in tqdm(files, disable=self.debug):
            input_filename = os.path.join(input_dir, file)
            output_filename = os.path.join(output_dir, file)
            if os.path.exists(output_filename):
                continue
            border = self.create_hollow_shell(input_filename)
            if border is not None:
                write_image(output_filename, border)


    def create_shell(self):

        self.create_aligned_masks()
        input_dir = self.fileLocationManager.get_directory(self.channel, self.downsample, inpath='masked_aligned')
        if not os.path.exists(input_dir):
            print(f'Missing: {input_dir}')
            return
          
        files = sorted(os.listdir(input_dir))
        file_list = []
        for file in tqdm(files, disable=self.debug):
            filepath = os.path.join(input_dir, file)
            border = read_image(filepath)
            file_list.append(border)

        volume = np.stack(file_list, axis = 0)
        volume = np.swapaxes(volume, 0, 2) # put it in x,y,z format
        volume = volume.astype(np.uint8)
        #ids = list(np.unique(volume, return_counts=False))
        data_type = volume.dtype

        if self.debug:
            # hard coding to DK55
            xy = 10.4 * 1000
            z = 20 * 1000
        else: 
            xy = (self.sqlController.scan_run.resolution * self.scaling_factor) * 1000
            z = self.sqlController.scan_run.zresolution * 1000
        scales = xy, xy, int(z)
        chunks = [64, 64, 64]
        scales = (int(round(xy)), int(round(xy)), int(z))        
        print(f'Volume shape={volume.shape} dtype={volume.dtype} chunks at {chunks} and scales after rounding with {scales}nm')
        
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
        segment_properties = {'255':'shell'}
        ng.add_segment_properties(cv2, segment_properties)

        ##### first mesh task, create meshing tasks
        print(f'Creating meshing tasks on volume from {cloudpath2}')
        ##### first mesh task, create meshing tasks
        ng.add_segmentation_mesh(cv2.layer_cloudpath, mip=0)

