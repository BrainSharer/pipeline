import os
import numpy as np
from PIL import Image
from utilities.utilities_mask import clean_and_rotate_image
from utilities.utilities_process import SCALING_FACTOR, test_dir
from database_model.slide import SlideCziTif
from database_model.slide import Slide
from database_model.slide import Section
Image.MAX_IMAGE_PIXELS = None


class ImageCleaner:
    '''
    Methods for cleaning images [and rotation, if necessary].  "Cleaning" means applying user-verified masks (QC step) to
    downsampled or full-resolution images


     Methods
    -------
    create_cleaned_images()
    create_cleaned_images_thumbnail()
    create_cleaned_images_full_resolution()
    get_section_rotation()
    parallel_create_cleaned()

    '''


    def create_cleaned_images(self):
        """
        This method applies the image masks that has been edited by the user to extract the tissue image from the surrounding
        debris
        """
        if self.channel == 1:
            self.sqlController.set_task(
                self.animal, self.progress_lookup.CLEAN_CHANNEL_1_THUMBNAIL_WITH_MASK
            )
        if self.downsample:
            self.create_cleaned_images_thumbnail()
        else:
            self.create_cleaned_images_full_resolution()

    def create_cleaned_images_thumbnail(self):
        """Clean the image using the masks for the downsampled version"""
        CLEANED = self.fileLocationManager.get_thumbnail_cleaned(self.channel)
        INPUT = self.fileLocationManager.get_thumbnail(self.channel)
        MASKS = self.fileLocationManager.thumbnail_masked
        self.fileLogger.logevent(f"INPUT FOLDER: {INPUT}")
        starting_files = os.listdir(INPUT)
        self.fileLogger.logevent(f"FILE COUNT: {len(starting_files)}")
        self.fileLogger.logevent(f"MASK FOLDER: {MASKS}")
        starting_files = os.listdir(INPUT)
        self.fileLogger.logevent(f"FILE COUNT: {len(starting_files)}")
        self.fileLogger.logevent(f"OUTPUT FOLDER: {CLEANED}")
        os.makedirs(CLEANED, exist_ok=True)
        self.parallel_create_cleaned(INPUT, CLEANED, MASKS)

    def create_cleaned_images_full_resolution(self):
        """Clean the image using the masks for the full resolution image"""
        CLEANED = self.fileLocationManager.get_full_cleaned(self.channel)
        os.makedirs(CLEANED, exist_ok=True)
        INPUT = self.fileLocationManager.get_full(self.channel)
        MASKS = self.fileLocationManager.full_masked
        self.fileLogger.logevent(f"INPUT FOLDER: {INPUT}")
        starting_files = os.listdir(INPUT)
        self.fileLogger.logevent(f"FILE COUNT: {len(starting_files)}")
        self.fileLogger.logevent(f"MASK FOLDER: {MASKS}")
        starting_files = os.listdir(INPUT)
        self.fileLogger.logevent(f"FILE COUNT: {len(starting_files)}")
        self.fileLogger.logevent(f"OUTPUT FOLDER: {CLEANED}")
        self.parallel_create_cleaned(INPUT, CLEANED, MASKS)

    def get_section_rotation(self, section: Section):
        sections = self.sqlController.session.query(SlideCziTif).filter(
            SlideCziTif.FK_slide_id == section.FK_slide_id
        )
        indices = np.sort(np.unique([i.scene_index for i in sections]))
        scene = np.where(indices == section.scene_index)[0][0] + 1
        slide = self.sqlController.session.query(Slide).get(section.FK_slide_id)
        return getattr(slide, f"scene_rotation_{scene}")

    def parallel_create_cleaned(self, INPUT, CLEANED, MASKS):
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

        progress_id = self.sqlController.get_progress_id(
            self.downsample, self.channel, "CLEAN"
        )
        self.sqlController.set_task(self.animal, progress_id)
        file_keys = []
        for file in files:
            infile = os.path.join(INPUT, file)
            outpath = os.path.join(CLEANED, file)  # regular-birdstore
            if os.path.exists(outpath):
                continue
            maskfile = os.path.join(MASKS, file)
            file_keys.append(
                [
                    infile,
                    outpath,
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

