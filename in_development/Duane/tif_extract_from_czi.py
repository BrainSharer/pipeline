import os
from aicspylibczi import CziFile
from aicsimageio import AICSImage
from aicsimageio.readers import CziReader
import numpy as np
from skimage.transform import rescale
from tifffile import imread, imwrite

import tifffile
from pathlib import Path

####################################################################################
#USER INPUTS

animal = 'DKBC008'
channel  = 1
downsample = True
debug = True
task = 'extract'
# czi_file = 'DKBC008_cord_slide004_loc1_2024_07_23_axion2.czi'

INPUT_DIR = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/czi_rescans'
OUTPUT_DIR = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/rescan_extractions'

####################################################################################

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
        print(f'{total_scenes=}')

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
        print(f'start scene extraction: {scene_index=}')
        region = self.get_scene_dimension(scene_index)
        print('ref: scene.x, scene.y, scene.w, scene.h')
        print(f'{region=}')
        print('end scene extraction')
        return self.file.read_mosaic(region=region, scale_factor=scale, C=channel - 1)[0]

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
        print(f'start extraction: read scene {scenei}')
        data = czi.get_scene(scale=scale, scene_index=scenei, channel=channel)
        print('end extraction')
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


# Output directory for TIFF files
os.makedirs(OUTPUT_DIR, exist_ok=True)

#scale=1 is full resolution, scale=16 is downsampled
scaling_factor = 32
scale = 1 / scaling_factor if downsample else 1

####################################################################################
#START PROCESSING HERE (NOT LIBRARIES)
# Get all .czi files
czi_files = list(Path(INPUT_DIR).glob('*.czi'))
print(f'NUMBER OF FILES TO PROCESS: {len(czi_files)}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

#scale=1 is full resolution, scale=16 is downsampled
if downsample == True:
    scale = 16
else:
    scale = 1

for czi_file in czi_files:
    print(f'ANALYZING FILE: {czi_file}')
    czi_file_path = czi_file.as_posix()
    czi = CZIManager(czi_file_path)
    # print('++META DATA++')
    # print(czi.extract_metadata_from_czi_file(czi_file, czi_file_path))
    print('STARTING EXTRACTION')
    print(f'DATA WILL BE SAVED TO: {OUTPUT_DIR}')

    scenei = 0
    output_path = Path(OUTPUT_DIR, f'{czi_file.stem}_scene{scenei}_channel{channel}_scale{scale}.tiff').as_posix()
    file_key = (czi_file, output_path, scenei, channel, scale)
    extract_tiff_from_czi(file_key)