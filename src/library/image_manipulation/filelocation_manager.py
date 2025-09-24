"""This module takes care of all the file locations of all the czi, tiff and log files.
"""
import os
import sys
from library.utilities.utilities_process import get_scratch_dir


data_path = "/net/birdstore/Active_Atlas_Data/data_root"
ALIGNED = 0
REALIGNED = 1
CLEANED_DIR = 'cleaned'
CROPPED_DIR = 'cropped'
ALIGNED_DIR = 'aligned'
REALIGNED_DIR = 'realigned'


class FileLocationManager(object):
    """Master class to house all file paths for preprocessing-pipeline

    All file locations are defined in this class for reference by other methods/functions.
    Default root is UCSD-specific (birdstore fileshare) but may be modified in data_path
    Subfolders to each brain are defined based on usage (some examples below):
    -czi folder [stores raw scanner images from which tiff images are extracted]
    -tif [stores full resolution images extracted from czi files]
    -neuroglancer_data [stores 'precomputed' format of entire image stack for loading into Neuroglancer visualization tool]
    """

    def __init__(self, stack, data_path=data_path):
        """setup the directory, file locations
        Args:
            stack: the animal brain name, AKA prep_id
        """

        # These need to be set first
        self.prep_id = stack
        self.root = os.path.join(data_path, "pipeline_data")
        self.registration_info = os.path.join(data_path, "brains_info/registration")
        self.stack = os.path.join(self.root, stack)
        self.prep = os.path.join(self.root, stack, "preps")
        self.masks = os.path.join(self.prep, "masks")
        self.www = os.path.join(self.stack, "www")
        self.slides_preview = os.path.join(self.www, "slides_preview")

        # The rest
        self.brain_info = os.path.join(self.root, stack, "brains_info")
        self.czi = os.path.join(self.root, stack, "czi")
        self.elastix = os.path.join(self.prep, "elastix")
        self.histogram = os.path.join(self.www, "histogram")
        self.neuroglancer_data = os.path.join(self.www, "neuroglancer_data")
        self.neuroglancer_progress = os.path.join(self.neuroglancer_data, 'progress')
        self.cell_labels_data = os.path.join(self.prep, "cell_labels")
        self.section_web = os.path.join(self.www, "section")
        self.tif = os.path.join(self.prep, "tif")
        self.thumbnail = os.path.join(self.prep, "1", "thumbnail")
        self.thumbnail_original = os.path.join(self.prep, "thumbnail_original")
        self.thumbnail_web = os.path.join(self.www, "scene")
        self.slide_thumbnail_web = os.path.join(self.www, "slide")

    def get_czi(self):
        return self.czi        

    def get_full(self, channel=1):
        return os.path.join(self.prep, f"C{channel}", "full")

    def get_thumbnail(self, channel=1):
        return os.path.join(self.prep, f"C{channel}", "thumbnail")

    def get_full_aligned(self, channel=1):
        """This is used in cell_ui"""
        validated_path = os.path.join(self.prep, f"C{channel}", "full_aligned")
        return validated_path

    def get_thumbnail_aligned(self, channel=1):
        """This is used in cell_ui"""
        validated_path = os.path.join(self.prep, f"C{channel}", "thumbnail_aligned")
        return validated_path

    def get_alignments(self, iteration=0):
        aligments = {}
        aligments[ALIGNED] = "aligned"
        aligments[REALIGNED] = "realigned"

        try:
            aligments[iteration]
        except KeyError:
            print(f'Invalid iteration {iteration}')
            sys.exit(1)

        return aligments[iteration]

    def get_alignment_directories(self, channel, downsample, iteration=ALIGNED):

        if downsample:
            resolution = "thumbnail"
        else:
            resolution = "full"

        inpath = self.get_alignments(iteration=iteration)
        input = os.path.join(self.prep, f'C{channel}', f'{resolution}_{inpath}')

        if iteration == REALIGNED:
            outpath = "NA"
        else:
            outpath = self.get_alignments(iteration=iteration+1)

        output = os.path.join(self.prep, f'C{channel}', f'{resolution}_{outpath}')

        # os.makedirs(output, exist_ok=True)
        return input, output

    def get_directory(self, channel: int, downsample: bool, inpath: str) -> str:
        if downsample:
            resolution = "thumbnail"
        else:
            resolution = "full"

        input = os.path.join(self.prep, f'C{channel}', f'{resolution}_{inpath}')
        return input


    def get_normalized(self, channel=1):
        return os.path.join(self.prep, f"C{channel}", "normalized")

    def get_thumbnail_colored(self, channel=1):
        return os.path.join(self.masks, f"C{channel}", "thumbnail_colored")

    def get_thumbnail_masked(self, channel=1):
        return os.path.join(self.masks, f"C{channel}", "thumbnail_masked")

    def get_full_colored(self, channel=1):
        return os.path.join(self.masks, f"C{channel}", "full_colored")

    def get_full_masked(self, channel=1):
        return os.path.join(self.masks, f"C{channel}", "full_masked")

    def get_histogram(self, channel=1):
        return os.path.join(self.histogram, f"C{channel}")

    def get_cell_labels(self):
        '''
        Returns path to store cell labels

        Note: This path is also web-accessbile [@ UCSD]
        '''
        return os.path.join(self.cell_labels_data)

    def get_neuroglancer(self, downsample=True, channel=1, iteration=0):
        '''
        Returns path to store neuroglancer files ('precomputed' format)

        Note: This path is also web-accessbile [@ UCSD]
        '''
        outpath = self.get_alignments(iteration=iteration)

        channel_outdir = f"C{channel}"
        if downsample:
            channel_outdir += f"T_{outpath}"

        return os.path.join(self.neuroglancer_data, f"{channel_outdir}")


    def get_neuroglancer_rechunkme(self, downsample: bool = True, channel: int = 1, iteration: int = 0, use_scratch_dir: bool = False, in_contents: str = None) -> str:
        """
        Generates the file path for the Neuroglancer rechunkme file based on the given parameters.
        DK37 uses 552G	for 478 images or 1.15G per image

        Args:
            downsample (bool): If True, includes a 'T' in the output directory name to indicate downsampling. Default is True.
            channel (int): The channel number to include in the output directory name. Default is 1.
            iteration (int): The iteration number to include in the alignment path. Default is 0.

        Returns:
            str: The generated file path for the Neuroglancer rechunkme file.
        """

        outpath = self.get_alignments(iteration=iteration)
        channel_outdir = f"C{channel}"
        if downsample:
            channel_outdir += "T"

        channel_outdir += f"_rechunkme_{outpath}"
        if use_scratch_dir:
            scratch_tmp = get_scratch_dir(in_contents)
            rechunkme_dir = os.path.join(scratch_tmp, 'pipeline_tmp', self.prep_id, channel_outdir)
        else:
            rechunkme_dir = os.path.join(self.neuroglancer_data, f"{channel_outdir}")
        os.makedirs(rechunkme_dir, exist_ok=True)
        return rechunkme_dir

    def get_neuroglancer_progress(self, downsample=True, channel=1, iteration=0):
        outpath = self.get_alignments(iteration=iteration)
        channel_outdir = f"C{channel}"
        if downsample:
            channel_outdir += f"T_{outpath}"

        progress_dir = os.path.join(self.neuroglancer_progress, f"{channel_outdir}") 
        return progress_dir

    def get_logdir(self):
        '''
        This method is only called on first instance then stored as environment variable
        [See: FileLogger class for more information]
        '''
        stackpath = os.path.join(self.stack)
        if os.path.exists(stackpath):
            return stackpath
        else:
            print(f'Path not found {stackpath}')
            sys.exit()
