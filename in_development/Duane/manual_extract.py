#PURPOSE: MANUALLY EXTRACT TIF IMAGES FROM CZI FILES (RESCANS)
#CONTACT DUANE WITH ANY QUESTIONS ABOUT THIS FILE (drinehart@ucsd.edu, duane.rinehart@gmail.com)

from scipy.ndimage import affine_transform
import tifffile as tiff
import os
import numpy as np
import copy
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from aicspylibczi import CziFile
from aicsimageio import AICSImage



PIPELINE_ROOT =  Path().resolve().parent
sys.path.append(PIPELINE_ROOT.as_posix())
sys.path.append('/data/pipeline/src/')
from library.utilities.utilities_mask import equalized

class CZIManager():
    """Methods to extract meta-data from czi files using AICSImage module (Allen Institute)
    """
    
    def __init__(self, czi_file):
        """Set up the class with the name of the file and the path to it's location.

        :param czi_file: string of the name of the CZI file
        """
        
        self.czi_file = czi_file
        self.file = CziFile(czi_file)

        # LOGFILE_PATH = os.environ["LOGFILE_PATH"]
        # super().__init__(LOGFILE_PATH)


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
        # if scene.x < 0:
        #     scene.x = 0
        return scene.x, scene.y, scene.w, scene.h
    
    def get_scene(self, scene_index, channel, scale=1):
        """Gets the correct scene from the slide

        :param scene_index: integer of the scene index
        :param channel: integer of the channel
        :param scale: integer of the scale. Usually either 1 or 16 (full or downsampled)
        :return: the scene  
        """
        print('start scene')
        region = self.get_scene_dimension(scene_index)
        print('ref: scene.x, scene.y, scene.w, scene.h')
        print(f'{region=}')
        print('end scene')
        return self.file.read_mosaic(region=region, scale_factor=scale, C=channel - 1)[0]

from tifffile import imread, imwrite
def write_image(file_path:str, data, message: str = "Error") -> None:
    """Writes an image to the filesystem
    """
    
    try:
        imwrite(file_path, data)
        #cv2.imwrite(file_path, data)
    except Exception as e:
        print(message, e)
        print("Unexpected error:", sys.exc_info()[0])
        try:
            imwrite(file_path, data)
        except Exception as e:
            print(message, e)
            print("Unexpected error:", sys.exc_info()[0])
            sys.exit()

def extract_tiff_from_czi(file_key):
    """Gets the TIFF file out of the CZI and writes it to the filesystem

    :param file_key: a tuple of: czi_file, output_path, scenei, channel, scale
    """
    czi_file, output_path, scenei, channel, scale = file_key
    czi = CZIManager(czi_file)
    data = None
    print(f'{file_key=}, {scenei} {channel} {scale}')
    try:
        print('start')
        data = czi.get_scene(scale=scale, scene_index=scenei, channel=channel)
        print('end')
    except Exception as e:
        message = f" ERROR READING SCENE {scenei} CHANNEL {channel} [extract_tiff_from_czi] IN FILE {czi_file} to file {os.path.basename(output_path)} {e}"
        print(message)
        #czi.logevent(message)
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
        #czi.logevent(message)

####################################################################################
#USER INPUTS
animal = 'DKBC008'
channel  = 1
downsample = True

czi_file = 'DKBC008_cord_slide004_loc1_2024_07_23_axion2.czi'
INPUT_DIR = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/czi_rescans'
OUTPUT_DIR = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/rescan_extractions'

scaling_factor = 32
debug = False
task = 'extract'
####################################################################################

czi_file_path = Path(INPUT_DIR, czi_file)
print(f'ANALYZING FILE: {czi_file}')

czi = CZIManager(czi_file_path)
print(czi.extract_metadata_from_czi_file(czi_file, czi_file_path))

print('+++')

####################################################################################
#SLIDE IMAGE?
scenei = 0
channel = 1
scale = 16 #1 is full resolution, 16 is downsampled
czi_file = Path(INPUT_DIR, czi_file)

output_path = Path(OUTPUT_DIR, f'{czi_file.stem}_scene{scenei}_channel{channel}_scale{scale}.tiff')
print(f'DATA WILL BE SAVED TO: {output_path}')
file_key = (czi_file_path, output_path, scenei, channel, scale)

extract_tiff_from_czi(file_key)