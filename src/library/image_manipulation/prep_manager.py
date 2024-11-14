import os
import sys
from PIL import Image
from library.controller.scan_run_controller import ScanRunController
Image.MAX_IMAGE_PIXELS = None

from library.utilities.utilities_process import get_image_size, read_image


class PrepCreater:
    """Contains methods related to generating low-resolution images from image stack 
    [so user can review for abnormalities
    e.g. missing tissue, poor scan, etc.] and applying this quality control analysis 
    to image stack
    """

    def update_scanrun(self, INPUT):
        """This is where the scan run table gets updated so the width and 
        height are correct.
        """
        if self.channel == 1 and self.downsample:
            files = sorted(os.listdir(INPUT))
            widths = []
            heights = []
            for file in files:
                filepath = os.path.join(INPUT, file)
                img = read_image(filepath)
                widths.append(img.shape[1])
                heights.append(img.shape[0])
            max_width = max(widths)
            max_height = max(heights)
            self.sqlController.update_width_height(self.sqlController.scan_run.id, max_width, max_height)


    def apply_QC(self):
        """Applies the inclusion and replacement results defined by the user on the Django admin portal for the Quality Control step
        to the full resolution images.  The result is stored in the animal_folder/preps/CHX/full directory
        Note: We don't want the image size when we are downsampling, only at full resolution.
        """

        if self.downsample:
            self.input = self.fileLocationManager.thumbnail_original
            self.output = self.fileLocationManager.get_thumbnail(self.channel)
        else:
            self.input = self.fileLocationManager.tif
            self.output = self.fileLocationManager.get_full(self.channel)
        
        if not os.path.exists(self.input):
            """This checks for the thumbnail_original dir. This might not be available with the original brains
            The data will then be in the regular thumbnail dir
            """
            print(f'This dir does not exist. {self.input}')
            print(f'Checking the regular thumbnail dir')
            self.input = self.output
            if not os.path.exists(self.input):
                return

        try:
            starting_files = os.listdir(self.input)
        except OSError:
            print(f"Error: Could not find the input directory: {self.input}")
            return
            
        self.fileLogger.logevent(f"Input FOLDER: {self.input}")
        self.fileLogger.logevent(f"INPUT FOLDER FILE COUNT: {len(starting_files)}")
        self.fileLogger.logevent(f"OUTPUT FOLDER: {self.output}")
        os.makedirs(self.output, exist_ok=True)
        try:
            sections = self.sqlController.get_sections(self.animal, self.channel, self.debug)
        except:
            raise Exception('Could not get sections from database')
        
        self.fileLogger.logevent(f"DB SECTIONS [EXPECTED OUTPUT FOLDER FILE COUNT]: {len(sections)}")

        for section_number, section in enumerate(sections):
            infile = os.path.basename(section.file_name)
            input_path = os.path.join(self.input, infile)
            output_path = os.path.join(self.output, str(section_number).zfill(3) + ".tif")
            
            if not os.path.exists(input_path):
                print(f"MISSING SRC FILE: {section_number=}; {input_path}: SKIPPING SYMBOLIC LINK (CHECK DB OR RENAME FILE)")
                continue

            if os.path.exists(output_path):
                continue

            if not self.downsample:
                width, height = get_image_size(input_path)
                self.sqlController.update_tif(section.id, width, height)

            if self.debug:
                print(f'Creating symlink to {output_path}')

            try:    
                relative_input_path = os.path.relpath(input_path, os.path.dirname(output_path))
                os.symlink(relative_input_path, output_path)
            except Exception as e:
                print(f"CANNOT CREATE SYMBOLIC LINK: {output_path} {e}")
                
