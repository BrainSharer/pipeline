"""This module takes care of the CZI file management. We used to use the bftool kit 
(https://www.openmicroscopy.org/)
which is a set of Java tools, but we opted to use a pure python library that
can handle CZI files. https://github.com/AllenCellModeling/aicspylibczi
"""
import os
from PIL import Image
from aicspylibczi import CziFile
from aicsimageio import AICSImage

from library.image_manipulation.file_logger import FileLogger
from library.utilities.utilities_process import write_image
from library.utilities.utilities_mask import equalized


class CZIManager(FileLogger):
    """Methods to extract meta-data from czi files using AICSImage module (Allen Institute)
    """
    
    def __init__(self, czi_file):
        """Set up the class with the name of the file and the path to it's location.

        :param czi_file: string of the name of the CZI file
        """
        
        self.czi_file = czi_file
        self.file = CziFile(czi_file)


        if "LOGFILE_PATH" in os.environ:
            LOGFILE_PATH = os.environ["LOGFILE_PATH"]
            super().__init__(LOGFILE_PATH)


    def extract_metadata_from_czi_file(self, czi_file, czi_file_path):
        """This will parse the xml metadata and return the relevant data.

        :param czi_file: string of the CZI file name
        :param czi_file_path: string of the CZI path
        :return: dictionary of the metadata
        """

        czi_aics = AICSImage(czi_file_path)
        total_scenes = czi_aics.scenes

        czi_meta_dict = {}
        scenes = {}
        for idx, scene in enumerate(total_scenes):
            czi_aics.set_scene(scene)
            dimensions = (czi_aics.dims.X, czi_aics.dims.Y)
            channels = czi_aics.dims.C

            print(f"CZI file: {czi_file}, scene: {czi_aics.current_scene}, dimensions: {dimensions}, channels: {channels}")

            scenes[idx] = {
                "scene_name": czi_aics.current_scene,
                "channels": channels,
                "dimensions": dimensions,
            }

        czi_meta_dict[czi_file] = scenes
        return czi_meta_dict
    
    def get_scene_dimension(self, scene_index):
        """Gets the bounding box size of the scene

        :param scene_index: integer of the scene index
        :return: x,y,width and height of the bounding box
        """

        scene = self.file.get_scene_bounding_box(scene_index)
        return scene.x, scene.y, scene.w, scene.h
    
    def get_scene(self, scene_index, channel, scale=1):
        """Gets the correct scene from the slide

        :param scene_index: integer of the scene index
        :param channel: integer of the channel
        :param scale: integer of the scale. Usually either 1 or 16 (full or downsampled)
        :return: the scene  
        """

        region = self.get_scene_dimension(scene_index)
        return self.file.read_mosaic(region=region, scale_factor=scale, C=channel - 1)[0]


def extract_tiff_from_czi(file_key):
    """Gets the TIFF file out of the CZI and writes it to the filesystem

    :param file_key: a tuple of: czi_file, output_path, scenei, channel, scale
    """
    czi_file, output_path, scenei, channel, scale = file_key
    czi = CZIManager(czi_file)
    data = None
    try:
        data = czi.get_scene(scale=scale, scene_index=scenei, channel=channel)
    except Exception as e:
        message = f" ERROR READING SCENE {scenei} CHANNEL {channel} [extract_tiff_from_czi] IN FILE {czi_file} to file {os.path.basename(output_path)} {e}"
        print(message)
        czi.logevent(message)
        return

    message = f"ERROR WRITING SCENE - [extract_tiff_from_czi] FROM FILE {czi_file} -> {output_path}; SCENE: {scenei}; CHANNEL: {channel} ... SKIPPING"
    write_image(output_path, data, message=message)



def extract_png_from_czi(file_key, normalize = True):
    """This method creates a PNG file from the TIFF file. This is used for viewing
    on a web page.
    
    :param file_key: tuple of _, infile, outfile, scene_index, scale
    :param normalize: a boolean that determines if we should normalize the TIFF
    """

    _, infile, outfile, scene_index, scale = file_key

    czi = CZIManager(infile)
    try:
        data = czi.get_scene(scene_index=scene_index, channel=1, scale=scale)
        if normalize:
            data = equalized(data)
        im = Image.fromarray(data)
        im.save(outfile)
    except Exception as e:
        message = f"ERROR READING SCENE - [extract_png_from_czi] IN FILE {infile} ERR: {e}"
        print(message)
        czi.logevent(message)
