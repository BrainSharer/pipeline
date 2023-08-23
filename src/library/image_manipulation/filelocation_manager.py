"""This module takes care of all the file locations of all the czi, tiff and log files.
"""
import os

data_path = "/net/birdstore/Active_Atlas_Data/data_root"


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
        self.root = os.path.join(data_path, "pipeline_data")
        self.registration_info = os.path.join(data_path, "brains_info/registration")
        self.stack = os.path.join(self.root, stack)
        self.prep = os.path.join(self.root, stack, "preps")
        self.masks = os.path.join(self.prep, "masks")
        self.www = os.path.join(self.stack, "www")
        
        # The rest
        self.brain_info = os.path.join(self.root, stack, "brains_info")
        self.czi = os.path.join(self.root, stack, "czi")
        self.elastix = os.path.join(self.prep, "elastix")
        self.histogram = os.path.join(self.www, "histogram")
        self.neuroglancer_data = os.path.join(self.www, "neuroglancer_data")
        self.neuroglancer_progress = os.path.join(self.neuroglancer_data, 'progress')
        self.section_web = os.path.join(self.www, "section")
        self.tif = os.path.join(self.prep, "tif")
        self.thumbnail = os.path.join(self.prep, "CH1", "thumbnail")
        self.thumbnail_original = os.path.join(self.prep, "thumbnail_original")
        self.thumbnail_web = os.path.join(self.www, "scene")

    def get_czi(self, czi=0):
        czi_path = self.czi
        if czi > 0:
            czi_path = os.path.join(self.stack, f'czi_{czi}')
        
        return czi_path
    
    def get_full(self, channel=1):
        return os.path.join(self.prep, f"CH{channel}", "full")

    def get_thumbnail(self, channel=1):
        return os.path.join(self.prep, f"CH{channel}", "thumbnail")

    def get_full_cleaned(self, channel=1):
        return os.path.join(self.prep, f"CH{channel}", "full_cleaned")

    def get_full_aligned_iteration_0(self, channel=1):
        return os.path.join(self.prep, f"CH{channel}", "full_aligned_iteration_0")

    def get_full_aligned(self, channel=1):
        return os.path.join(self.prep, f"CH{channel}", "full_aligned")

    def get_thumbnail_aligned_iteration_0(self, channel=1):
        return os.path.join(self.prep, f"CH{channel}", "thumbnail_aligned_iteration_0")

    def get_thumbnail_aligned(self, channel=1):
        return os.path.join(self.prep, f"CH{channel}", "thumbnail_aligned")

    def get_thumbnail_cleaned(self, channel=1):
        return os.path.join(self.prep, f"CH{channel}", "thumbnail_cleaned")

    def get_normalized(self, channel=1):
        return os.path.join(self.prep, f"CH{channel}", "normalized")

    def get_thumbnail_colored(self, channel=1):
        return os.path.join(self.masks, f"CH{channel}", "thumbnail_colored")
    
    def get_thumbnail_masked(self, channel=1):
        return os.path.join(self.masks, f"CH{channel}", "thumbnail_masked")

    def get_full_colored(self, channel=1):
        return os.path.join(self.masks, f"CH{channel}", "full_colored")
    
    def get_full_masked(self, channel=1):
        return os.path.join(self.masks, f"CH{channel}", "full_masked")

    def get_histogram(self, channel=1):
        return os.path.join(self.histogram, f"CH{channel}")

    def get_neuroglancer(self, downsample=True, channel=1, rechunk=False):
        '''
        Returns path to store neuroglancer files ('precomputed' format)

        Note: This path is also web-accessbile [@ UCSD]
        '''
        if downsample:
            channel_outdir = f"C{channel}T"
        else:
            channel_outdir = f"C{channel}"
        if not rechunk:
            channel_outdir += "_rechunkme"
        return os.path.join(self.neuroglancer_data, f"{channel_outdir}")

    def get_neuroglancer_progress(self, downsample=True, channel=1, rechunk=False):
        if downsample:
            channel_outdir = f"C{channel}T"
        else:
            channel_outdir = f"C{channel}"
        if not rechunk:
            channel_outdir += "_rechunkme"
        return os.path.join(self.neuroglancer_progress, f"{channel_outdir}")

    def get_logdir(self):
        '''
        This method is only called on first instance then stored as environment variable
        [See: FileLogger class for more information]
        '''
        return os.path.join(self.stack)
