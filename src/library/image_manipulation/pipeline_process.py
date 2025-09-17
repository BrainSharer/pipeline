"""
This class is used to run the entire preprocessing pipeline - 
from CZI files to a pyramid of tiles that can be viewed in neuroglancer.

Args are animal, self.channel, and downsample. With animal being
the only required argument.
All imports are listed by the order in which they are used in the 
"""

import os, sys
import shutil
import SimpleITK as sitk
import psutil
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from pathlib import Path
import uuid

from library.image_manipulation.elastix_manager import ElastixManager
from library.image_manipulation.file_logger import FileLogger
from library.image_manipulation.filelocation_manager import ALIGNED, ALIGNED_DIR, CLEANED_DIR, REALIGNED, REALIGNED_DIR, FileLocationManager
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
from library.utilities.utilities_mask import compare_directories
from library.utilities.utilities_process import get_hostname, SCALING_FACTOR, get_scratch_dir, use_scratch_dir, delete_in_background
from library.database_model.scan_run import IMAGE_MASK
from library.cell_labeling.cell_ui import Cell_UI

from library.utilities.cell_utilities import (
    copy_with_rclone
)

try:
    from settings import data_path, host, schema
except ImportError:
    print('Missing settings using defaults')
    data_path = "/net/birdstore/Active_Atlas_Data/data_root"
    host = "db.dk.ucsd.edu"
    schema = "brainsharer"


class Pipeline(
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
    TiffExtractor,
    Cell_UI
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
    TASK_NG_PREVIEW = "Creating neuroglancer preview"
    TASK_SHELL = "Creating 3D shell outline"

    def __init__(self, animal: str, channel: str ='C1', zarrlevel=0, downsample=False, scaling_factor=SCALING_FACTOR, task='status', arg_uuid: str = None, debug: bool = False):
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
        self.animal = animal
        self.task = task
        self.downsample = downsample
        self.debug = debug
        self.fileLocationManager = FileLocationManager(animal, data_path=data_path)
        self.sqlController = SqlController(animal)
        self.session = self.sqlController.session
        self.hostname = get_hostname()
        self.iteration = None
        self.mask_image = self.sqlController.scan_run.mask
        self.maskpath = self.fileLocationManager.get_thumbnail_masked(channel=1)
        self.multiple_slides = []
        self.channel = channel
        self.zarrlevel = zarrlevel
        self.scaling_factor = scaling_factor
        self.bgcolor = 0
        self.checksum = os.path.join(self.fileLocationManager.www, 'checksums')
        self.use_scratch = True # set to True to use scratch space (defined in - utilities.utilities_process::get_scratch_dir)
        total_mem = psutil.virtual_memory().total
        self.available_memory = int(total_mem * 0.65) ##### that 0.85 should match the dask config in your home directory ~/.config/dask/distributed.yaml
        self.section_count = self.get_section_count()

        #self.mips = 7 
        #if self.downsample:
        #    self.mips = 4

        self.fileLogger = FileLogger(self.fileLocationManager.get_logdir(), self.debug)
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        #self.report_status()
        #self.check_settings()
        if not hasattr(self, 'SCRATCH'):
            self.SCRATCH = get_scratch_dir()
        if arg_uuid:
            self.set_id = arg_uuid #for debug of prev. uuid
        else:
            self.set_id = uuid.uuid4().hex
        

    def report_status(self):
        print("RUNNING PREPROCESSING-PIPELINE WITH THE FOLLOWING SETTINGS:")
        print("\tprep_id:".ljust(20), f"{self.animal}".ljust(20))
        print("\tchannel:".ljust(20), f"{str(self.channel)}".ljust(20))
        print("\tdownsample:".ljust(20), f"{str(self.downsample)} @ {1/self.scaling_factor}".ljust(20))
        #print("\tscaling factor:".ljust(20), f"{str(self.scaling_factor)}".ljust(20))
        print("\tDB host:".ljust(20), f"{host}".ljust(20))
        print("\tprocess host:".ljust(20), f"{self.hostname}".ljust(20))
        print("\tschema:".ljust(20), f"{schema}".ljust(20))
        print("\tmask:".ljust(20), f"{IMAGE_MASK[self.mask_image]}".ljust(20))
        print("\tdebug:".ljust(20), f"{str(self.debug)}".ljust(20))
        print("\ttask:".ljust(20), f"{str(self.task)}".ljust(20))
        print("\tavailable RAM:".ljust(20), f"{str(self.available_memory)}GB".ljust(20))
        print()


    def get_section_count(self):
        """
        Retrieves the count of sections for the specified animal. If the section count
        from the database is zero, it calculates the count based on the number of 
        thumbnail files in the specified directory.
        I put this back in as test_dir requires it.
        :returns: int: The total count of sections.
        """
        section_count = self.sqlController.get_section_count(self.animal)
        if section_count  == 0 and os.path.exists(self.fileLocationManager.get_thumbnail()):
            section_count = len(os.listdir(self.fileLocationManager.get_thumbnail()))
        elif section_count == 0 and not os.path.exists(self.fileLocationManager.get_thumbnail()):
            section_count = self.sqlController.scan_run.number_of_slides
                                           
        return section_count


    def extract(self):
        print(self.TASK_EXTRACT)
        self.extract_slide_meta_data_and_insert_to_database() #ALSO CREATES SLIDE PREVIEW IMAGE
        if self.channel == 1 and self.downsample:
            self.correct_multiples()
        self.extract_tiffs_from_czi()
        
        self.reorder_scenes()
        if self.channel == 1 and self.downsample:
            self.create_web_friendly_image()
            self.create_previews()
            self.create_checksums()
            
            #ADD SYMLINKS TO EXTRACTED THUMBNAIL IMAGES if necessary
            if not self.url_exists(self.animal):
                target_path = str(self.fileLocationManager.www)
                link_path = str(Path('/', 'srv', self.animal))
                self.create_symbolic_link(target_path, link_path)

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
        print('SKIPPING RAM CHECK')
        # self.check_ram()
        if self.channel == 1 and self.downsample:
            self.apply_user_mask_edits()
            self.set_max_width_and_height()

        self.create_cleaned_images()
        print(f'Finished {self.TASK_CLEAN}.')


    def histogram(self):
        print(self.TASK_HISTOGRAM)
        if self.downsample:
            self.make_histogram()
            self.make_combined_histogram()
            if self.channel == 1:
                self.create_web_friendly_sections()
        else:
            print(f'No histogram for full resolution images')
        print(f'Finished {self.TASK_HISTOGRAM}.')


    def affine_align(self):
        """Perform the section to section alignment (registration)
        This method needs work. It is not currently used.
        """
        self.input = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath=CLEANED_DIR)
        self.output = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath='affine')
        os.makedirs(self.output, exist_ok=True)
        self.create_affine_transformations()


    def align(self):
        """Perform the section to section alignment (registration)
        """

        print(self.TASK_ALIGN)
        self.pixelType = sitk.sitkFloat32
        self.iteration = ALIGNED
        self.input = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath=CLEANED_DIR)
        self.output = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath=ALIGNED_DIR)
        self.logpath = os.path.join(self.fileLocationManager.prep, 'registration', 'iteration_logs')
        os.makedirs(self.logpath, exist_ok=True)

        if self.channel == 1 and self.downsample:
            self.create_within_stack_transformations()

        self.start_image_alignment()
        
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
        self.input = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath=CLEANED_DIR)
        self.output = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath=REALIGNED_DIR)
        self.logpath = os.path.join(self.fileLocationManager.prep, 'registration', 'iteration_logs')
        os.makedirs(self.logpath, exist_ok=True)

        if self.channel == 1 and self.downsample:
            print('Cleaning up realignment and creating fiducial points and within stack transformations')
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

        self.check_ram()
        self.iteration = self.get_alignment_status()
        if self.iteration is None:
            print('No alignment iterations found.  Please run the alignment steps first.')
            return
        
        print(self.TASK_NEUROGLANCER)

        self.input = self.fileLocationManager.get_directory(channel=self.channel, downsample=self.downsample, inpath=ALIGNED_DIR)
        self.input, _ = self.fileLocationManager.get_alignment_directories(channel=self.channel, downsample=self.downsample, iteration=self.iteration)  
        self.output = self.fileLocationManager.get_neuroglancer(self.downsample, self.channel, iteration=self.iteration)
        self.use_scratch = use_scratch_dir(self.input)

        #NEW WORKFLOW STORES ALL MIPS IN SAME DIRECTORY
        # self.rechunkme_path = self.fileLocationManager.get_neuroglancer_rechunkme(
        #     self.downsample, self.channel, iteration=self.iteration, use_scratch_dir=self.use_scratch, in_contents=self.input)
        
        if self.use_scratch:
            print('CHECKING IF SCRATCH SPACE HAS ENOUGH SPACE TO STORE TASK FILES...')
            SCRATCH = get_scratch_dir(self.input)
        else:
            SCRATCH = self.SCRATCH

        temp_output_path = Path(SCRATCH, 'pipeline_tmp', self.animal, 'C' + str(self.channel) + '_ng')
        self.progress_dir = self.fileLocationManager.get_neuroglancer_progress(self.downsample, self.channel, iteration=self.iteration)
        os.makedirs(self.progress_dir, exist_ok=True)

        print(f'INPUT: {self.input}')
        print(f'USING SCRATCH SPACE: {self.use_scratch}')
        print(f'TEMP DIR: {temp_output_path}')
        print(f'FINAL OUTPUT: {self.output}')
        print(f'Progress: {self.progress_dir}')
        # print(f'Rechunkme: {self.rechunkme_path}')
        
        # self.create_neuroglancer()
        # self.create_downsamples()

        max_memory_gb = 100 #muralis testing
        self.create_precomputed(self.input, temp_output_path, self.output, self.progress_dir, max_memory_gb)
        # print(f'Make sure you delete {self.rechunkme_path}.')

        copy_with_rclone(temp_output_path, self.output)
        print(f'Finished {self.TASK_NEUROGLANCER}.')


    def omezarr(self):
        """Note for RGB ndim=3 images!!!!
        The omezarr proess here works but you will need to adjust neuroglancer when you load it:
        1. Open up the JSON data in neuroglancer (Edit JSON state icon top right). Add
        "crossSectionOrientation": [0, 0, 0, -1] to the JSON state. right above crossSectionScale.
        2. Adjust the dimensions so it is x,y,z,t, click 'Apply changes.
        3. The position will now be wrong, so adjust the x,y,z to mid positions in the x,y,z top left. 
        4. Set z to 0 and
        5. If the data is RGB, rename the c' channel to c^ :
        From the developer https://github.com/google/neuroglancer/issues/298
        To make a dimension available as a "channel" dimension to the shader, you need to rename the dimension to end with "^", e.g. "c^". 
        You can do that by double clicking the dimension name in the top bar, or using the "Transform" widget on the "Source" tab of the 
        layer. However, this is currently only supported if that dimension is not chunked, i.e. the chunk size must be 3 in your case.
        """
        print(self.TASK_OMEZARR)
        self.check_ram()

        self.input, _ = self.fileLocationManager.get_alignment_directories(channel=self.channel, downsample=self.downsample) 
        self.scratch_space = os.path.join('/data', 'pipeline_tmp', self.animal, 'dask-scratch-space')
        if os.path.exists(self.scratch_space):
            shutil.rmtree(self.scratch_space)
        os.makedirs(self.scratch_space, exist_ok=True)
        print(f'Scratch space: {self.scratch_space}')
        self.create_omezarr()
        scratch_parent = os.path.dirname(self.scratch_space)
        if os.path.exists(scratch_parent) and os.path.isdir(scratch_parent):
            print(f'You should remove scratch space: {scratch_parent}')

        print(f'Finished {self.TASK_OMEZARR}.')


    def omezarr_info(self):
        self.get_omezarr_info()


    def omezarr2tif(self):
        self.write_sections_from_volume()


    def shell(self):
        print(self.TASK_SHELL, end=" ")
        self.create_shell()
        print(f'Finished {self.TASK_SHELL}.')


    def align_masks(self):
        print("Aligning masks")
        self.create_rotated_aligned_masks()
        print(f'Finished aligning masks.')


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
                f"C{self.channel}/thumbnail_aligned",
            ]
            ndirectory = f"C{self.channel}T"
        else:
            directories = [
                f"masks/C1/full_masked",
                f"C{self.channel}/full",
                f"C{self.channel}/full_cleaned",
                f"C{self.channel}/full_aligned",
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
        # compare masked dir to initial dir
        if self.downsample:
            mask_dir = self.fileLocationManager.get_thumbnail_masked(channel=1)
            orig_dir = self.fileLocationManager.get_thumbnail()
        else:
            mask_dir = self.fileLocationManager.get_full_masked(channel=1)
            orig_dir = self.fileLocationManager.get_full(channel=self.channel)

        if os.path.exists(mask_dir) and os.path.exists(orig_dir) \
            and len(os.listdir(mask_dir)) == len(os.listdir(orig_dir)) \
            and len(os.listdir(mask_dir)) > 0:
            print(f'Comparing {mask_dir[57:]} to {orig_dir[57:]}')
            compare_directories(orig_dir, mask_dir)
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


    def ng_preview(self):
        '''
        USED TO GENERATE NG STATE WITH DEFAULT SETTINGS (AVAILABLE CHANNELS, ANNOTATIONS)
        As this is used to do QC on image stacks
        '''
        print(self.TASK_NG_PREVIEW)
        self.ng_prep() #will create precomputed format if not exist
        self.gen_ng_preview() #just create ng state
        print(f'Finished {self.TASK_NG_PREVIEW}.')
        

    def check_settings(self):
        """
        Make sure there is a ./src/settings.py file
        """

        error = ""

        if not os.path.exists("./src/settings.py"):
            error += "\nThere is no ./src/settings.py file!"

        if len(error) > 0:
            print(error)
            sys.exit()


    def check_ram(self):
        """
        I set an arbitrary limit of 50GB of RAM for the full resolution images
        """

        error = ""

        if not self.downsample and self.available_memory < 50:
            error += f'\nThere is not enough memory to run this process at full resolution with only: {self.available_memory}GB RAM'
            error += '\n(Available RAM is calculated as free RAM * 0.8. You can check this by running "free -h" on the command line.)'
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
    
    @staticmethod
    def url_exists(animal):
        exists = True
        url = f"https://imageserv.dk.ucsd.edu/data/{animal}/"
        url_response = Request(url)
        try:
            response = urlopen(url_response, timeout=10)
        except HTTPError as e:
            # do something
            exists = False

        return exists