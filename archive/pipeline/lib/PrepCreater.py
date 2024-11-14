import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from concurrent.futures.process import ProcessPoolExecutor
from utilities.utilities_process import test_dir, create_downsample, get_image_size


class PrepCreater:
    '''
    Contains methods related to generating low-resolution images from image stack [so user can review for abnormalities
    e.g. missing tissue, poor scan, etc.] and applying this quality control analysis to image stack

    Methods
    -------
    set_task_preps()
    apply_QC()
    make_low_resolution()

    '''


    def set_task_preps(self):
        if self.channel == 1:
            self.sqlController.update_scanrun(self.sqlController.scan_run.id)
        progress_id = self.sqlController.get_progress_id(True, self.channel, "TIF")
        self.sqlController.set_task(self.animal, progress_id)
        progress_id = self.sqlController.get_progress_id(False, self.channel, "TIF")
        self.sqlController.set_task(self.animal, progress_id)

    def apply_QC(self):
        """
        Applies the inclusion and replacement results defined by the user on the Django admin portal for the Quality Control step
        to the full resolution images.  The result is stored in the animal_folder/preps/full directory
        """
        if self.downsample:
            INPUT = self.fileLocationManager.thumbnail_original
            OUTPUT = self.fileLocationManager.get_thumbnail(self.channel)
        else:
            INPUT = self.fileLocationManager.tif
            OUTPUT = self.fileLocationManager.get_full(self.channel)
            
        starting_files = os.listdir(INPUT)
        self.fileLogger.logevent(f"INPUT FOLDER: {INPUT}")
        self.fileLogger.logevent(f"CURRENT FILE COUNT: {len(starting_files)}")
        self.fileLogger.logevent(f"OUTPUT FOLDER: {OUTPUT}")
        os.makedirs(OUTPUT, exist_ok=True)
        sections = self.sqlController.get_sections(self.animal, self.channel)
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
            if self.downsample:
                self.sqlController.update_tif(section.id, width, height)

            try:    
                os.symlink(relative_input_path, output_path)
            except Exception as e:
                print(f"CANNOT CREATE SYMBOLIC LINK (ALREADY EXISTS): {output_path} {e}")
                

    def make_low_resolution(self):
        """
        Making low resolution version of the full resolution images with QC applied.  These images are used to
        create image masks and within stack alignments
        """
        file_keys = []
        INPUT = self.fileLocationManager.get_full(self.channel)
        OUTPUT = self.fileLocationManager.get_thumbnail(self.channel)
        os.makedirs(OUTPUT, exist_ok=True)
        test_dir(
            self.animal, INPUT, self.section_count, downsample=False, same_size=False
        )
        files = sorted(os.listdir(INPUT))
        for file in files:
            infile = os.path.join(INPUT, file)
            outpath = os.path.join(OUTPUT, file)
            if os.path.exists(outpath):
                continue
            file_keys.append([infile, outpath])
            
        workers = self.get_nworkers()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            executor.map(create_downsample, sorted(file_keys))

