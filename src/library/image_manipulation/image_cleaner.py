"""This module takes clear of cleaning up the junk from outside
the brain area by using masks.
"""
import os
import shutil
import sys
from PIL import Image
from taskqueue.taskqueue import LocalTaskQueue
import igneous.task_creation as tc
from cloudvolume import CloudVolume
Image.MAX_IMAGE_PIXELS = None
import numpy as np

from library.image_manipulation.neuroglancer_manager import MESHDTYPE, NumpyToNeuroglancer
from library.database_model.scan_run import FULL_MASK_NO_CROP
from library.image_manipulation.filelocation_manager import CLEANED_DIR, CROPPED_DIR
from library.image_manipulation.image_manager import ImageManager
from library.utilities.utilities_mask import clean_and_rotate_image, get_image_box, place_image, rotate_image
from library.utilities.utilities_process import SCALING_FACTOR, get_cpus, read_image, test_dir, write_image


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
            self.input = self.fileLocationManager.get_thumbnail(self.channel)
            self.maskpath = self.fileLocationManager.get_thumbnail_masked(self.channel)
        else:
            self.input = self.fileLocationManager.get_full(self.channel)
            self.maskpath = self.fileLocationManager.get_full_masked(self.channel)

        self.output = self.fileLocationManager.get_directory(self.channel, self.downsample, inpath=CLEANED_DIR)

        try:
            starting_files = os.listdir(self.input)
        except OSError:
            print(f"Error: Could not find the input directory: {self.input}")
            return

        self.fileLogger.logevent(f"image_cleaner::create_cleaned_images Input FOLDER: {self.input} FILE COUNT: {len(starting_files)} MASK FOLDER: {self.maskpath}")
        os.makedirs(self.output, exist_ok=True)
        image_manager = ImageManager(self.input)
        if self.mask_image > 0: 
            self.bgcolor = image_manager.get_bgcolor()
        else:
            self.bgcolor = 0
        self.setup_parallel_create_cleaned()
        # Update the scan run with the cropped width and height. The images are also rotated and/or flipped at this point. 
        if self.mask_image > 0 and self.channel == 1 and self.downsample and self.mask_image != FULL_MASK_NO_CROP:
            self.set_crop_size()
            if self.debug:
                print(f'Updating scan run. and set bgcolor to {self.bgcolor}')
        self.setup_parallel_place_images()
        

    def setup_parallel_create_cleaned(self):
        """Do the image cleaning in parallel
        If we are working on the downsampled files, we delete the output directory to make 
        sure there are no stale files
        """

        rotation = self.sqlController.scan_run.rotation
        flip = self.sqlController.scan_run.flip
        files, *_ = test_dir(self.animal, self.input, self.section_count, self.downsample, same_size=False)

        if self.downsample and os.path.exists(self.output):
            print(f'Removing {self.output}')
            shutil.rmtree(self.output)

        os.makedirs(self.output, exist_ok=True)

        file_keys = []
        for file in files:
            infile = os.path.join(self.input, file)
            outfile = os.path.join(self.output, file)
            if os.path.exists(outfile):
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
                    self.debug
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
        
        self.input = self.fileLocationManager.get_directory(self.channel, self.downsample, inpath=CLEANED_DIR)
        self.output = self.fileLocationManager.get_directory(self.channel, self.downsample, inpath=CROPPED_DIR)
        if self.downsample and os.path.exists(self.output):
            print(f'Removing {self.output}')
            shutil.rmtree(self.output)

        os.makedirs(self.output, exist_ok=True)
        max_width = self.sqlController.scan_run.width
        max_height = self.sqlController.scan_run.height
        if self.downsample:
            max_width = int(max_width / SCALING_FACTOR)
            max_height = int(max_height / SCALING_FACTOR)

        if max_width == 0 or max_height == 0:
            print(f'Error in setup parallel place images: width or height is 0. width={max_width} height={max_height}')
            sys.exit()

        files, _, max_width, max_height = test_dir(self.animal, self.input, self.section_count, self.downsample, same_size=False)

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


    def create_shell(self):
        CLEAN = False
        self.maskpath = self.fileLocationManager.get_thumbnail_masked(channel=1) # usually channel=1, except for step 6
        maskfiles = sorted(os.listdir(self.maskpath))
        rotation = self.sqlController.scan_run.rotation
        flip = self.sqlController.scan_run.flip
        max_width = self.sqlController.scan_run.width
        max_height = self.sqlController.scan_run.height
        max_width = int(max_width / SCALING_FACTOR)
        max_height = int(max_height / SCALING_FACTOR)
        bgcolor = 0

        self.output = self.fileLocationManager.get_directory(self.channel, self.downsample, inpath='placed_mask')
        if os.path.exists(self.output) and CLEAN:
            print(f'Removing {self.output}')
            shutil.rmtree(self.output)
        os.makedirs(self.output, exist_ok=True)

        for maskfile in maskfiles:
            maskpath = os.path.join(self.maskpath, maskfile)
            outfile = os.path.join(self.output, maskfile)
            if os.path.exists(outfile):
                continue
            mask = read_image(maskpath)
            if rotation > 0:
                cleaned = rotate_image(mask, maskpath, rotation)
            # flip = switch top to bottom
            # flop = switch left to right
            if flip == "flip":
                cleaned = np.flip(cleaned, axis=0)
            if flip == "flop":
                cleaned = np.flip(cleaned, axis=1)
            del mask
            zmidr = max_height // 2
            zmidc = max_width // 2
            startr = max(0, zmidr - (cleaned.shape[0] // 2))
            endr = min(max_height, startr + cleaned.shape[0])
            startc = max(0, zmidc - (cleaned.shape[1] // 2))
            endc = min(max_width, startc + cleaned.shape[1])



            placed_img = np.full((max_height, max_width), bgcolor, dtype=cleaned.dtype)
            try:
                placed_img[startr:endr, startc:endc] = cleaned[:endr-startr, :endc-startc]
            except Exception as e:
                print(f"Error placing {maskfile}: {e}")



            message = f'Error in saving {outfile} with shape {placed_img.shape} img type {placed_img.dtype}'
            write_image(outfile, placed_img, message=message)
        ##### now align images
        self.input = self.output
        self.files = os.listdir(self.input)
        len_files = len(self.files)
        self.output = self.fileLocationManager.get_directory(self.channel, self.downsample, inpath='placed_aligned')
        os.makedirs(self.output, exist_ok=True)
        self.iteration = 0
        self.start_image_alignment()
        ##### now start mesh creation
        self.input = self.output
        self.output = None
        self.progress_dir = os.path.join(self.fileLocationManager.neuroglancer_data, 'progress', 'shell')
        self.mesh_input_dir = os.path.join(self.fileLocationManager.neuroglancer_data, 'mesh_input_shell')
        self.mesh_dir = os.path.join(self.fileLocationManager.neuroglancer_data, 'shell')
        self.layer_path = f'file://{self.mesh_dir}'

        xy = self.sqlController.scan_run.resolution * 1000 * 1000 / self.scaling_factor
        z = self.sqlController.scan_run.zresolution * 1000
        self.scales = (int(xy), int(xy), int(z))
        print(f'scales {self.scales} self.scaling_factor={self.scaling_factor} and resolution={self.sqlController.scan_run.resolution}')
        self.xs = self.scales[0]
        self.ys = self.scales[1]
        self.zs = self.scales[2]
        scale_dir = "_".join([str(self.xs), str(self.ys), str(self.zs)])
        self.transfered_path = os.path.join(self.mesh_dir, scale_dir)

        self.chunks = [64,64,1]
        ng = NumpyToNeuroglancer(self.animal, None, self.scales, layer_type='segmentation', 
            data_type=MESHDTYPE, chunk_size=self.chunks)
        self.volume_size = (max_width, max_height, len_files)
        ng.init_precomputed(self.mesh_input_dir, self.volume_size)

        for index, file in enumerate(sorted(self.files)):
            infile = os.path.join(self.input, file)
            filekey = (index, infile, self.progress_dir)
            ng.process_image_shell(filekey)

        chunks = [64, 64, 64]
        self.mesh_mip = 0

        _, cpus = get_cpus()
        tq = LocalTaskQueue(parallel=cpus)
        os.makedirs(self.mesh_dir, exist_ok=True)
        if not os.path.exists(self.transfered_path):
            tasks = tc.create_image_shard_transfer_tasks(ng.precomputed_vol.layer_cloudpath, 
                                                            self.layer_path, mip=0, 
                                                            chunk_size=chunks)

            print(f'Creating transfer tasks in {self.transfered_path} with shards and chunks={chunks}')
            tq.insert(tasks)
            tq.execute()
        else:
            print(f'Already created transfer tasks in {self.transfered_path} with shards and chunks={chunks}')



        ##### add segment properties
        cloudpath = CloudVolume(self.layer_path, self.mesh_mip)
        downsample_path = cloudpath.meta.info['scales'][self.mesh_mip]['key']
        print(f'Creating mesh from {downsample_path}', end=" ")
        self.ids = [0, 255]
        segment_properties = {str(id): str(id) for id in self.ids}

        print('and creating segment properties', end=" ")
        ng.add_segment_properties(cloudpath, segment_properties)

        sharded = False
        print(f'and mesh at mip={self.mesh_mip} with shards={str(sharded)}')
        tasks = tc.create_meshing_tasks(self.layer_path, mip=self.mesh_mip, 
                                        compress=True, 
                                        sharded=sharded) # The first phase of creating mesh
        tq.insert(tasks)
        tq.execute()

        # for apache to serve shards, this command: curl -I --head --header "Range: bytes=50-60" https://activebrainatlas.ucsd.edu/index.html
        # must return HTTP/1.1 206 Partial Content
        # a magnitude < 3 is more suitable for local mesh creation. Bigger values are for horizontal scaling in the cloud.

        print(f'Creating meshing manifest tasks with {cpus} CPUs')
        tasks = tc.create_mesh_manifest_tasks(self.layer_path) # The second phase of creating mesh
        tq.insert(tasks)
        tq.execute()







