"""
This class is used to run the entire preprocessing pipeline - 
from CZI files to a pyramid of tiles that can be viewed in neuroglancer.

Args are animal, self.channel, and downsample. With animal being
the only required argument.
All imports are listed by the order in which they are used in the 
"""

import os
import SimpleITK as sitk
import sys
import psutil
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from library.image_manipulation.elastix_manager import ElastixManager
from library.cell_labeling.cell_manager import CellMaker
from library.image_manipulation.file_logger import FileLogger
from library.image_manipulation.filelocation_manager import ALIGNED, ALIGNED_DIR, CROPPED_DIR, REALIGNED, REALIGNED_DIR, FileLocationManager
from library.image_manipulation.histogram_maker import HistogramMaker
from library.image_manipulation.image_cleaner import ImageCleaner
from library.image_manipulation.mask_manager import MaskManager
from library.image_manipulation.meta_manager import MetaUtilities
from library.image_manipulation.precomputed_manager import NgPrecomputedMaker
from library.image_manipulation.normalizer_manager import Normalizer
from library.image_manipulation.omezarr_manager import OmeZarrManager
from library.image_manipulation.parallel_manager import ParallelManager
from library.image_manipulation.prep_manager import PrepCreater
from library.controller.sql_controller import SqlController
from library.image_manipulation.tiff_extractor_manager import TiffExtractor

from library.utilities.utilities_process import get_hostname, SCALING_FACTOR, get_scratch_dir, use_scratch_dir
from library.database_model.scan_run import IMAGE_MASK

try:
    from settings import data_path, host, schema
except ImportError:
    print('Missing settings using defaults')
    data_path = "/net/birdstore/Active_Atlas_Data/data_root"
    host = "db.dk.ucsd.edu"
    schema = "brainsharer"


class Pipeline(
    CellMaker,
    ElastixManager,
    HistogramMaker,
    ImageCleaner,
    MaskManager,
    MetaUtilities,
    NgPrecomputedMaker,
    Normalizer,
    OmeZarrManager,
    ParallelManager,
    PrepCreater,
    TiffExtractor
):
    """
    This is the main class that handles the preprocessing pipeline responsible for converting Zeiss microscopy images (.czi) into neuroglancer
    viewable formats.  The Zeiss module can be swapped out to make the pipeline compatible with other microscopy setups
    """
    TASK_EXTRACT = "Extracting TIFFs and meta-data"
    TASK_MASK = "Creating masks"
    TASK_CLEAN = "Applying masks"
    TASK_HISTOGRAM =  "Making histogram"
    TASK_ALIGN = "Creating alignment process"
    TASK_REALIGN = "Creating alignment improvements"
    TASK_EXTRA_CHANNEL = "Creating separate channel"
    TASK_NEUROGLANCER = "Creating Neuroglancer data"
    TASK_CELL_LABELS = "Creating centroids for cells"
    TASK_OMEZARR = "Creating multiscaled ome zarr"
    TASK_SHELL = "Creating 3D shell outline"

    def __init__(self, animal, channel='C1', downsample=False, scaling_factor=SCALING_FACTOR, task='status', debug=False):
        """Setting up the pipeline and the processing configurations

           The pipeline performst the following steps:
           1. extracting the images from the microscopy formats (eg czi) to tiff format
           2. Prepare thumbnails of images for quality control
           3. clean the images
           4. crop the images
           5. align the images
           6. convert to Seung lab neuroglancer cloudvolume format

           steps 3, 4, and 5 are first performed on downsampled images, and the image masks(for step 3) and the within stack alignments(for step 5) are
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
        self.downsample = downsample
        self.debug = debug

        self.fileLocationManager = FileLocationManager(animal, data_path=data_path)
        self.sqlController = SqlController(animal)
        self.session = self.sqlController.session
        self.hostname = get_hostname()
        self.iteration = None
        self.mask_image = self.sqlController.scan_run.mask
        self.maskpath = self.fileLocationManager.get_thumbnail_masked(channel=1)
        self.section_count = self.get_section_count()
        self.multiple_slides = []
        self.channel = channel
        self.scaling_factor = scaling_factor
        self.checksum = os.path.join(self.fileLocationManager.www, 'checksums')
        self.use_scratch = True # set to True to use scratch space (defined in - utilities.utilities_process::get_scratch_dir)
        self.available_memory = int((psutil.virtual_memory().free / 1024**3) * 0.8)

        #self.mips = 7 
        #if self.downsample:
        #    self.mips = 4

        self.fileLogger = FileLogger(self.fileLocationManager.get_logdir(), self.debug)
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        self.check_programs()
        self.report_status()

    def report_status(self):
        print("RUNNING PREPROCESSING-PIPELINE WITH THE FOLLOWING SETTINGS:")
        print("\tprep_id:".ljust(20), f"{self.animal}".ljust(20))
        print("\tchannel:".ljust(20), f"{str(self.channel)}".ljust(20))
        print("\tdownsample:".ljust(20), f"{str(self.downsample)}".ljust(20))
        print("\tscaling factor:".ljust(20), f"{str(self.scaling_factor)}".ljust(20))
        print("\tDB host:".ljust(20), f"{host}".ljust(20))
        print("\tprocess host:".ljust(20), f"{self.hostname}".ljust(20))
        print("\tschema:".ljust(20), f"{schema}".ljust(20))
        print("\tmask:".ljust(20), f"{IMAGE_MASK[self.mask_image]}".ljust(20))
        print("\tdebug:".ljust(20), f"{str(self.debug)}".ljust(20))
        print("\ttask:".ljust(20), f"{str(self.task)}".ljust(20))
        print()

    def get_section_count(self):
        section_count = self.sqlController.get_section_count(self.animal)
        if section_count == 0:
            INPUT = self.fileLocationManager.get_thumbnail(channel=1)
            if os.path.exists(INPUT):
                section_count = len(os.listdir(INPUT))

        return section_count

    def extract(self):
        print(self.TASK_EXTRACT)
        self.extract_slide_meta_data_and_insert_to_database() #ALSO CREATES SLIDE PREVIEW IMAGE
        self.correct_multiples()
        self.extract_tiffs_from_czi()
        if self.channel == 1 and self.downsample:
            self.create_web_friendly_image()
        if self.channel == 1 and self.downsample:
            self.create_previews()
            self.create_checksums()
        print(f'Finished {self.TASK_EXTRACT}.')

    def mask(self):
        print(self.TASK_MASK)
        self.apply_QC() # symlinks from tif/thumbnail_original to CX/thumbnail or CX/full are created
            
        if self.channel == 1 and self.downsample:
            self.create_normalized_image()
            
        if self.channel == 1:
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
        if self.downsample:
            self.make_histogram()
            self.make_combined_histogram()
            print(f'Finished {self.TASK_HISTOGRAM}.')
        else:
            print(f'No histogram for full resolution images')


    def affine_align(self):
        """Perform the section to section alignment (registration)
        This method needs work. It is not currently used.
        """
        self.input = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath=CROPPED_DIR)
        self.output = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath='affine')
        os.makedirs(self.output, exist_ok=True)
        self.create_affine_transformations()

    def align(self):
        """Perform the section to section alignment (registration)
        """

        print(self.TASK_ALIGN)
        self.pixelType = sitk.sitkFloat32
        self.iteration = ALIGNED
        self.input = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath=CROPPED_DIR)
        self.output = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath=ALIGNED_DIR)
        self.logpath = os.path.join(self.fileLocationManager.prep, 'registration', 'iteration_logs')
        os.makedirs(self.logpath, exist_ok=True)

        if self.channel == 1 and self.downsample:
            self.create_within_stack_transformations()

        self.start_image_alignment()
        
        if self.channel == 1 and self.downsample:
            self.create_web_friendly_sections()

        print(f'Finished {self.TASK_ALIGN}.')


    def realign(self): 
        """Perform the improvement of the section to section alignment. It will use fiducial points to improve the already
        aligned image stack from thumnbail_aligned. This only needs to be run on downsampled channel 1 images. With the full
        resolution images, the transformations come from both iterations of the downsampled images and then scaled.
        While the transformations are only created on channel 1, the realignment needs to occur on all channels
        """        
        print(self.TASK_REALIGN)
        self.pixelType = sitk.sitkFloat32
        self.iteration = REALIGNED
        self.input = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath=ALIGNED_DIR)
        self.output = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath=REALIGNED_DIR)
        self.logpath = os.path.join(self.fileLocationManager.prep, 'registration', 'iteration_logs')
        os.makedirs(self.logpath, exist_ok=True)

        print(f'Second elastix manager alignment input: {self.input}')
        if self.channel == 1 and self.downsample:
            self.cleanup_fiducials()
            self.create_fiducial_points()
            self.create_within_stack_transformations()
            
        self.start_image_alignment()
        print(f'Finished {self.TASK_REALIGN}.')


    def neuroglancer(self):
        """This is the main method to run the entire neuroglancer process.
        We also define the input, output and progress directories.
        This method may be run from the command line as a task, or it may
        also be run from the align and realign methods
        """

        self.iteration = self.get_alignment_status()
        if self.iteration is None:
            print('No alignment iterations found.  Please run the alignment steps first.')
            return
        
        print(self.TASK_NEUROGLANCER)

        
        self.input, _ = self.fileLocationManager.get_alignment_directories(channel=self.channel, downsample=self.downsample, iteration=self.iteration)            
        self.use_scratch = use_scratch_dir(self.input)
        self.rechunkme_path = self.fileLocationManager.get_neuroglancer_rechunkme(
            self.downsample, self.channel, iteration=self.iteration, use_scratch_dir=self.use_scratch)
        self.output = self.fileLocationManager.get_neuroglancer(self.downsample, self.channel, iteration=self.iteration)
        self.progress_dir = self.fileLocationManager.get_neuroglancer_progress(self.downsample, self.channel, iteration=self.iteration)
        os.makedirs(self.progress_dir, exist_ok=True)
        print(f'Input: {self.input}')
        print(f'Output: {self.output}')
        print(f'Progress: {self.progress_dir}')
        print(f'Rechunkme: {self.rechunkme_path}')

        self.create_neuroglancer()
        self.create_downsamples()
        print(f'Make sure you delete {self.rechunkme_path}.')
        print(f'Finished {self.TASK_NEUROGLANCER}.')


    def omezarr(self):
        print(self.TASK_OMEZARR)
        self.create_omezarr()
        print(f'Finished {self.TASK_OMEZARR}.')

    def shell(self):
        print(self.TASK_SHELL, end=" ")
        self.create_shell()
        print(f'Finished {self.TASK_SHELL}.')

    def cell_labels(self):
        """
        USED FOR AUTOMATED CELL LABELING - FINAL OUTPUT FOR CELLS DETECTED
        """
        print(self.TASK_CELL_LABELS)

        scratch_tmp = get_scratch_dir()
        self.check_prerequisites(scratch_tmp)

        # IF ANY ERROR FROM check_prerequisites(), PRINT ERROR AND EXIT

        # ASSERT STATEMENT COULD BE IN UNIT TEST (SEPARATE)

        self.start_labels()
        print(f'Finished {self.TASK_CELL_LABELS}.')

        # ADD CLEANUP OF SCRATCH FOLDER

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

    def check_status(self):
        prep = self.fileLocationManager.prep
        neuroglancer = self.fileLocationManager.neuroglancer_data
        print(f'Checking directory status in {prep}')
        section_count = self.section_count
        print(f'Section count from DB={section_count}')

        if self.downsample:
            directories = [
                "thumbnail_original",
                f"masks/C1/thumbnail_colored",
                f"masks/C1/thumbnail_masked",
                f"C{self.channel}/thumbnail",
                "C1/normalized",
                f"C{self.channel}/thumbnail_cleaned",
                f"C{self.channel}/thumbnail_cropped",
                f"C{self.channel}/thumbnail_aligned",
                f"C{self.channel}/thumbnail_realigned",
            ]
            ndirectory = f"C{self.channel}T"
        else:
            directories = [
                f"masks/C1/full_masked",
                f"C{self.channel}/full",
                f"C{self.channel}/full_cleaned",
                f"C{self.channel}/full_cropped",
                f"C{self.channel}/full_aligned",
                f"C{self.channel}/full_realigned",
            ]
            ndirectory = f"C{self.channel}"

        directories.append(self.fileLocationManager.get_czi() )

        for directory in directories:
            dir = os.path.join(prep, directory)
            if os.path.exists(dir):
                filecount = len(os.listdir(dir))
                print(f'Dir={directory} exists with {filecount} files.', end=' ')
                if 'czi' in directory:
                    print()
                else:
                    print(f'Sections count matches directory count: {section_count == filecount}')
            else:
                print(f'Non-existent dir={dir}')
        del dir, directory, directories
        dir = os.path.join(neuroglancer, ndirectory)
        display_directory = dir[57:]
        if os.path.exists(dir):
            print(f'Dir={display_directory} exists.')
        else:
            print(f'Non-existent dir={display_directory}')
        # neuroglancer progress dir
        progress_dir = self.fileLocationManager.get_neuroglancer_progress(self.downsample, self.channel)
        histogram_dir = self.fileLocationManager.get_histogram(self.channel)
        for directory in [progress_dir, histogram_dir]:
            display_directory = directory[57:]
            if os.path.exists(directory):
                completed_files = len(os.listdir(directory))
                if completed_files == section_count:
                    print(f'Dir={display_directory} exists with {completed_files} files and matches section count={section_count}.')
                else:
                    print(f'Dir={display_directory} exists with {completed_files} files completed out of {section_count} total files.')
            else:
                print(f'Non-existent dir={display_directory}')
        url_status = self.check_url(self.animal)
        print(url_status)

    def check_programs(self):
        """
        Make sure imagemagick is installed.
        Make sure there is a ./src/settings.py file and make sure there is enough RAM
        I set an arbitrary limit of 50GB of RAM for the full resolution images
        """

        error = ""
        if not os.path.exists("/usr/bin/identify"):
            error += "\nImagemagick is not installed"

        if not os.path.exists("./src/settings.py"):
            error += "\nThere is no ./src/settings.py file!"

        if not self.downsample and self.available_memory < 50:
            error += f'\nThere is noot enough memory to run at full resolution: {self.available_memory}GB.'
            error += '\nYou need to free up some RAM. From the terminal run as root (login as root first: sudo su -l) then run:'
            error += '\n\tsync;echo 3 > /proc/sys/vm/drop_caches'
            

        if len(error) > 0:
            print(error)
            sys.exit()

    @staticmethod
    def check_url(animal):
        status = ""
        url = f"https://imageserv.dk.ucsd.edu/data/{animal}/"
        url_response = Request(url)
        try:
            response = urlopen(url_response, timeout=10)
        except HTTPError as e:
            # do something
            status += f"Warning: {url} does not exist. HTTP error code = {e.code}\n"
            status += f"You need to create:\n ln -svi /net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/www {animal}\n"
            status += "on the imageserv.dk.ucsd.edu server in the /srv directory."
        else:
            # do something
            status = f"Imageserver link exists for {animal}"

        return status
