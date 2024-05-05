"""
This class is used to run the entire preprocessing pipeline - 
from CZI files to a pyramid of tiles that can be viewed in neuroglancer.

Args are animal, self.channel, and downsample. With animal being
the only required argument.
All imports are listed by the order in which they are used in the 
"""

import os
import sys

from library.image_manipulation.filelocation_manager import FileLocationManager
from library.image_manipulation.meta_manager import MetaUtilities
from library.image_manipulation.prep_manager import PrepCreater
from library.image_manipulation.precomputed_manager import NgPrecomputedMaker
from library.image_manipulation.tiff_extractor_manager import TiffExtractor
from library.image_manipulation.file_logger import FileLogger
from library.image_manipulation.parallel_manager import ParallelManager
from library.image_manipulation.normalizer_manager import Normalizer
from library.image_manipulation.mask_manager import MaskManager
from library.image_manipulation.image_cleaner import ImageCleaner
from library.image_manipulation.histogram_maker import HistogramMaker
from library.image_manipulation.elastix_manager import ElastixManager
from library.cell_labeling.cell_manager import CellMaker
from library.image_manipulation.omezarr_manager import OmeZarrManager
from library.controller.sql_controller import SqlController
from library.utilities.utilities_process import get_hostname, SCALING_FACTOR
from library.database_model.scan_run import IMAGE_MASK

try:
    from settings import data_path, host, schema
except ImportError:
    print('Missing settings using defaults')
    data_path = "/net/birdstore/Active_Atlas_Data/data_root"
    host = "db.dk.ucsd.edu"
    schema = "brainsharer"


class Pipeline(
    MetaUtilities,
    TiffExtractor,
    PrepCreater,
    ParallelManager,
    Normalizer,
    MaskManager,
    ImageCleaner,
    HistogramMaker,
    ElastixManager,
    NgPrecomputedMaker,
    FileLogger,
    CellMaker,
    OmeZarrManager
):
    """
    This is the main class that handles the preprocessing pipeline responsible for converting Zeiss microscopy images (.czi) into neuroglancer
    viewable formats.  The Zeiss module can be swapped out to make the pipeline compatible with other microscopy setups
    """
    TASK_EXTRACT = "Extracting TIFFs and meta-data"
    TASK_MASK = "Creating masks"
    TASK_CLEAN = "Applying masks"
    TASK_HISTOGRAM =  "Making histogram"
    TASK_ALIGN = "Creating elastix transform"
    TASK_EXTRA_CHANNEL = "Creating separate channel"
    TASK_NEUROGLANCER = "Neuroglancer"
    TASK_CELL_LABELS = "Creating centroids for cells"
    TASK_OMEZARR = "Creating multiscaled ome zarr"

    def __init__(self, animal, rescan_number=0, channel='C1', downsample=False, scaling_factor=SCALING_FACTOR,  
                 task='status', debug=False):
        """Setting up the pipeline and the processing configurations
        Here is how the Class is instantiated:
            pipeline = Pipeline(animal, self.channel, downsample, data_path, tg, debug)

           The pipeline performst the following steps:
           1. extracting the images from the microscopy formats (eg czi) to tiff format
           2. Prepare thumbnails of images for quality control
           3. clean the images
           4. align the images
           5. convert to Seung lab neuroglancer cloudvolume format

           step 3 and 4 are first performed on downsampled images, and the image masks(for step 3) and the within stack alignments(for step 4) are
           upsampled for use in the full resolution images

        Args:
            animal (str): Animal Id
            self.channel (int, optional): self.channel number.  This tells the program which self.channel to work on and which self.channel to extract from the czis. Defaults to 1.
            downsample (bool, optional): Determine if we are working on the full resolution or downsampled version. Defaults to True.
            data_path (str, optional): path to where the images and intermediate steps are stored. Defaults to '/net/birdstore/Active_Atlas_Data/data_root'.
            debug (bool, optional): determine if we are in debug mode.  This is used for development purposes. Defaults to False. (forces processing on single core)
        """
        self.task = task
        self.animal = animal
        self.rescan_number = rescan_number
        self.downsample = downsample
        self.debug = debug
        self.fileLocationManager = FileLocationManager(animal, data_path=data_path)
        self.sqlController = SqlController(animal, rescan_number)
        self.session = self.sqlController.session
        self.hostname = get_hostname()
        self.mask_image = self.sqlController.scan_run.mask
        self.check_programs()
        self.section_count = self.get_section_count()
        self.multiple_slides = []
        self.channel = channel
        self.scaling_factor = scaling_factor

        self.mips = 7 
        if self.downsample:
            self.mips = 4

        super().__init__(self.fileLocationManager.get_logdir())
        self.report_status()


    def report_status(self):
        print("RUNNING PREPROCESSING-PIPELINE WITH THE FOLLOWING SETTINGS:")
        print("\tprep_id:".ljust(20), f"{self.animal}".ljust(20))
        print("\trescan_number:".ljust(20), f"{self.rescan_number}".ljust(20))
        print("\tchannel:".ljust(20), f"{str(self.channel)}".ljust(20))
        print("\tdownsample:".ljust(20), f"{str(self.downsample)}".ljust(
            20), f"@ {str(self.scaling_factor)}".ljust(20))
        print("\thost:".ljust(20), f"{host}".ljust(20))
        print("\tschema:".ljust(20), f"{schema}".ljust(20))
        print("\tmask:".ljust(20), f"{IMAGE_MASK[self.mask_image]}".ljust(20))
        print("\tdebug:".ljust(20), f"{str(self.debug)}".ljust(20))
        print()

    def get_section_count(self):
        section_count = self.sqlController.get_section_count(self.animal, self.rescan_number)
        if section_count == 0:
            INPUT = self.fileLocationManager.get_thumbnail(channel=1)
            if os.path.exists(INPUT):
                section_count = len(os.listdir(INPUT))

        return section_count


    def extract(self):
        print(self.TASK_EXTRACT)
        self.extract_slide_meta_data_and_insert_to_database()
        #self.correct_multiples()
        self.extract_tiffs_from_czi()
        self.create_web_friendly_image()
        print(f'Finished {self.TASK_EXTRACT}.')

    def mask(self):
        print(self.TASK_MASK)
        self.apply_QC()
        self.create_normalized_image()
        self.create_mask()
        print(f'Finished {self.TASK_MASK}.')
    
    def clean(self):
        print(self.TASK_CLEAN)
        if self.channel == 1 and self.downsample:
            self.apply_user_mask_edits()
            
        self.create_cleaned_images()
        print(f'Finished {self.TASK_CLEAN}.')
    
    def histogram(self):
        print(self.TASK_HISTOGRAM)
        self.make_histogram()
        self.make_combined_histogram()
        print(f'Finished {self.TASK_HISTOGRAM}.')

    def align(self):
        """Perform the section to section alignment (registration)
        """

        print(self.TASK_ALIGN)
        self.create_within_stack_transformations()
        transformations = self.get_transformations()
        self.align_downsampled_images(transformations)
        self.align_full_size_image(transformations)
        self.create_web_friendly_sections()
        print(f'Finished {self.TASK_ALIGN}.')


    def extra_channel(self):
        """This step is in case self.channel X differs from self.channel 1 and came from a different set of CZI files. 
        This step will do everything for the self.channel, so you don't need to run self.channel X for step 2, or 4. You do need
        to run step 0 and step 1.
        TODO fix for channel variable name
        """
        print(self.TASK_EXTRA_CHANNEL)
        if self.downsample:
            self.create_normalized_image()
            self.create_downsampled_mask()
            self.apply_user_mask_edits()
            self.create_cleaned_images_thumbnail(channel=self.channel)
            self.create_dir2dir_transformations()
        else:
            self.create_full_resolution_mask(channel=self.channel)
            self.create_cleaned_images_full_resolution(channel=self.channel)
            self.apply_full_transformations(channel=self.channel)
        print(f'Finished {self.TASK_EXTRA_CHANNEL}.')

    def neuroglancer(self):
        print(self.TASK_NEUROGLANCER)
        self.create_neuroglancer()
        self.create_downsamples()
        print(f'Finished {self.TASK_NEUROGLANCER}.')

    def omezarr(self):
        print(self.TASK_OMEZARR)
        self.create_omezarr()
        print(f'Finished {self.TASK_OMEZARR}.')

    def cell_labels(self):
        """
        USED FOR AUTOMATED CELL LABELING - FINAL OUTPUT FOR CELLS DETECTED
        """
        print(self.TASK_CELL_LABELS)
        self.check_prerequisites()

        #IF ANY ERROR FROM check_prerequisites(), PRINT ERROR AND EXIT

        #ASSERT STATEMENT COULD BE IN UNIT TEST (SEPARATE)
        
        self.start_labels()
        print(f'Finished {self.TASK_CELL_LABELS}.')

        #ADD CLEANUP OF SCRATCH FOLDER


    def check_status(self):
        prep = self.fileLocationManager.prep
        neuroglancer = self.fileLocationManager.neuroglancer_data
        print(f'Checking directory status in {prep}')
        section_count = self.section_count
        print(f'Section count from DB={section_count}')

        if self.downsample:
            directories = ['thumbnail_original', f'masks/C1/thumbnail_colored', f'masks/C1/thumbnail_masked',
                           f'C{self.channel}/thumbnail', f'C{self.channel}/thumbnail_cleaned',
                           f'C{self.channel}/thumbnail_aligned']
            ndirectory = f'C{self.channel}T'
        else:
            directories = [f'masks/C{self.channel}/full_masked', f'C{self.channel}/full', 
                           f'C{self.channel}/full_cleaned', f'C{self.channel}/full_aligned']
            ndirectory = f'C{self.channel}'

        for directory in directories:
            dir = os.path.join(prep, directory)
            if os.path.exists(dir):
                filecount = len(os.listdir(dir))
                print(f'Dir={directory} exists with {filecount} files.', end=' ') 
                print(f'Sections count matches directory count: {section_count == filecount}')
            else:
                print(f'Non-existent dir={dir}')
        del dir, directory, directories
        dir = os.path.join(neuroglancer, ndirectory)
        if os.path.exists(dir):
            print(f'Dir={dir} exists.')
        else:
            print(f'Non-existent dir={dir}')


    @staticmethod
    def check_programs():
        """
        Make sure the necessary tools are installed on the machine and configures the memory of 
        involving tools to work with big images.
        We use to use java so we adjust the java heap size limit to 10 GB.  This is big enough 
        for our purpose but should be increased accordingly if your images are bigger
        If the check failed, check the workernoshell.err.log in your project directory for more information
        """
        
        error = ""
        if not os.path.exists("/usr/bin/identify"):
            error += "\nImagemagick is not installed"

        if len(error) > 0:
            print(error)
            sys.exit()
