import os
from PIL import Image
from library.controller.scan_run_controller import ScanRunController
Image.MAX_IMAGE_PIXELS = None

from library.utilities.utilities_process import get_image_size


class PrepCreater:
    """Contains methods related to generating low-resolution images from image stack 
    [so user can review for abnormalities
    e.g. missing tissue, poor scan, etc.] and applying this quality control analysis 
    to image stack
    """

    def update_scanrun(self):
        """This is where the scan run table gets updated so the width and 
        height are correct.
        """
        if self.channel == 1 and self.downsample:
            scanrunController = ScanRunController(self.session)
            scanrunController.update_scanrun(self.sqlController.scan_run.id)

    def apply_QC(self):
        """Applies the inclusion and replacement results defined by the user on the Django admin portal for the Quality Control step
        to the full resolution images.  The result is stored in the animal_folder/preps/CHX/full directory
        Note: We don't want the image size when we are downsampling, only at full resolution.
        """

        if self.downsample:
            INPUT = self.fileLocationManager.thumbnail_original
            OUTPUT = self.fileLocationManager.get_thumbnail(self.channel)
        else:
            INPUT = self.fileLocationManager.tif
            OUTPUT = self.fileLocationManager.get_full(self.channel)
            
        starting_files = os.listdir(INPUT)
        self.logevent(f"INPUT FOLDER: {INPUT}")
        self.logevent(f"CURRENT FILE COUNT: {len(starting_files)}")
        self.logevent(f"OUTPUT FOLDER: {OUTPUT}")
        os.makedirs(OUTPUT, exist_ok=True)
        sections = self.sqlController.get_sections(self.animal, self.channel, self.rescan_number)
        for section_number, section in enumerate(sections):
            infile = os.path.basename(section.file_name)
            input_path = os.path.join(INPUT, infile)
            output_path = os.path.join(OUTPUT, str(section_number).zfill(3) + ".tif")
            if not os.path.exists(input_path):
                continue
            if os.path.exists(output_path):
                continue
            relative_input_path = os.path.relpath(input_path, os.path.dirname(output_path))
            width, height = get_image_size(input_path)
            if not self.downsample:
                self.sqlController.update_tif(section.id, width, height)

            try:    
                os.symlink(relative_input_path, output_path)
            except Exception as e:
                print(f"CANNOT CREATE SYMBOLIC LINK (ALREADY EXISTS): {output_path} {e}")
                
