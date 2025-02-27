from datetime import datetime
import shutil
import os, sys
import socket
import concurrent
from skimage import io
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2
import numpy as np
import gc
from skimage.transform import rescale
import math
from tifffile import imread, imwrite
from concurrent.futures import Future
import random
import string

SCALING_FACTOR = 32.0
M_UM_SCALE = 1000000
DOWNSCALING_FACTOR = 1 / SCALING_FACTOR
Image.MAX_IMAGE_PIXELS = None


def delete_in_background(path: str) -> Future:
    current_date = datetime.now().strftime('%Y-%m-%d')
    old_path = f"{path}.old_{current_date}"

    if os.path.exists(old_path): #JUST IN CASE >1 PROCESSED IN SINGLE DAY
        shutil.rmtree(old_path)

    os.rename(path, old_path)  # Rename the directory

    # Delete the renamed directory in the background
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(shutil.rmtree, old_path)
    return future 


def use_scratch_dir(directory: str) -> bool:
    """
    Determines if there is enough free space in the /scratch directory to accommodate
    the specified directory with a buffer factor applied.
    Args:
    :param directory (str): The path to the directory whose size needs to be checked.
    :return: bool: True if there is enough free space in the /scratch directory, False otherwise.
    """
    
    BUFFER_FACTOR = 1.25
    dir_size = get_directory_size(directory)
    dir_size = dir_size * BUFFER_FACTOR
    total, used, free = shutil.disk_usage("/scratch")

    if free > dir_size:
        return True
    return False 


def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if not os.path.islink(file_path):
                total_size += os.path.getsize(file_path)
    return total_size


def get_hostname() -> str:
    """Returns hostname of server where code is processed

    :return: string of the current workstation
    """

    hostname = socket.gethostname()
    hostname = hostname.split(".")[0]
    return hostname


def get_image_size(filepath: str) -> tuple[int, int]:
    """
    Returns the width and height of a single image using Pillow.

    :param filepath: Path of the input file.
    :return: Tuple containing the width and height as integers.
    """
    try:
        with Image.open(filepath) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f'Error processing file={filepath}: {e}')
        return 0, 0  # Return default values in case of error
    

def test_dir(animal: str, directory: str, section_count: int, downsample: bool = True, same_size: bool = False) -> tuple[list[str], int, int, int]:
    """
    Tests the directory for image files, checks their sizes, and validates the number of files.
    Args:
    :param animal (str): The name of the animal.
    :param directory (str): The path to the directory containing the image files.
    :param section_count (int): The expected number of sections (files) in the directory.
    :param downsample (bool, optional): Whether to downsample the images. Defaults to True.
    :param same_size (bool, optional): Whether all images should be of the same size. Defaults to False.
    :return: tuple[list[str], int, int, int]: A tuple containing:
    - A list of filenames in the directory.
    - The number of files in the directory.
    - The maximum width of the images.
    - The maximum height of the images.
    :raise: SystemExit: If there are errors in processing the files or if the number of files is incorrect.
    """

    error = ""
    # thumbnail resolution ntb is 10400 and min size of DK52 is 16074
    # thumbnail resolution thion is 14464 and min size for MD585 is 21954
    # so 3000 is a good min size. I had to turn this down as we are using
    # blank images and they are small
    # min size on NTB is 8.8K
    starting_size = 20
    min_size = starting_size  * 1000
    if downsample:
        min_size = starting_size
    try:
        files = sorted(os.listdir(directory))
    except:
        error = f"{directory} does not exist\n"
        files = []

    
    if section_count == 0:
        section_count = len(files)
    widths = set()
    heights = set()

    for f in files:
        filepath = os.path.join(directory, f)

        try:
            file_stats = os.stat(filepath)
            size = file_stats.st_size
            
            if size < min_size:
                error += f"File is too small. {size} is less than min: {min_size} {filepath}\n"
                continue

            # Get width and height of the image
            width, height = get_image_size(filepath)
            
            # Add to sets
            widths.add(int(width))
            heights.add(int(height))
        
        except Exception as e:
            error += f"Error processing file {filepath}: {e}\n"

    if len(widths) == 0 or len(heights) == 0:
        error += f"No valid images in {directory}\n"
        print(error)
        sys.exit()

    min_width = min(widths)
    max_width = max(widths)
    min_height = min(heights)
    max_height = max(heights)
    if section_count != len(files):
        print(
            "[EXPECTED] SECTION COUNT:",
            section_count,
            "[ACTUAL] FILES:",
            len(files),
        )
        error += f"Number of files in {directory} is incorrect.\n"
        error += "If there are no slides in the DB, section count comes from the preps/C1/thumbnail dir. Make sure that is correct.\n"
    if min_width != max_width and min_width > 0 and same_size:
        error += f"Widths are not of equal size, min is {min_width} and max is {max_width}.\n"
    if min_height != max_height and min_height > 0 and same_size:
        error += f"Heights are not of equal size, min is {min_height} and max is {max_height}.\n"
    if len(error) > 0:
        print(error)
        sys.exit()
        
    return (files, len(files), max_width, max_height)

def get_cpus():
    """Helper method to return the number of CPUs to use
    """

    nmax = 4
    usecpus = (nmax, nmax)
    cpus = {}
    cpus["mothra"] = (2, 6)
    cpus["godzilla"] = (1, 6)
    cpus["muralis"] = (10, 20)
    cpus["basalis"] = (4, 12)
    cpus["ratto"] = (4, 8)
    cpus["tobor"] = (1, 12)
    hostname = get_hostname()
    if hostname in cpus.keys():
        usecpus = cpus[hostname]
    return usecpus

def get_scratch_dir():
    """Helper method to return the scratch dir
    Ratto can't use /scratch as it is not big enough
    """

    # usedir = {}
    # usedir['ratto'] = "/data"

    # hostname = get_hostname()
    # if hostname in usedir.keys():
    #     tmp_dir = usedir[hostname]
    # else:
    #     tmp_dir = "/scratch"
    
    #/scratch created on all servers (device or symbolic link to space with enough storage)
    tmp_dir = "/scratch"

    return tmp_dir

def convert(img, target_type_min, target_type_max, target_type):
    """Converts an image from one type to another and also resizes

    :param img: numpy array
    :param target_type_min: min size
    :param target_type_max: max size
    :param target_type: dtype of array
    :return:
    """

    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    del img
    return new_img


def create_downsample(file_key):
    """takes a big tif and scales it down to a manageable size.
    This method is used in PrepCreator
    For 16bit images, this is a good number near the high end.
    """

    infile, outpath = file_key
    try:
        img = io.imread(infile)
        img = rescale(img, SCALING_FACTOR, anti_aliasing=True)
        img = convert(img, 0, 2**16 - 1, np.uint16)
    except IOError as e:
        print(f"Could not open {infile} {e}")
    try:
        cv2.imwrite(outpath, img)
    except IOError as e:
        print(f"Could not write {outpath} {e}")
    del img
    gc.collect()
    return


def convert_size(size_bytes: int) -> str:
    """Function takes unformatted bytes, calculates human-readable format [with units] and returns string

    :param size_bytes:
    :type size_bytes: int
    :return: str:
    """

    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def write_image(file_path:str, data, message: str = "Error") -> None:
    """Writes an TIFF image to the filesystem
    """
    
    try:
        imwrite(file_path, data, compression='LZW', bigtiff=True)
    except Exception as e:
        print(message, e)
        print("Unexpected error:", sys.exc_info()[0])
        try:
            cv2.imwrite(file_path, data)
        except Exception as e:
            print(message, e)
            print("Unexpected error:", sys.exc_info()[0])
            sys.exit()


def read_image(file_path: str):
    """Reads an image from the filesystem with exceptions
    """
    img = None
    try:
        img = io.imread(file_path)
    except (OSError, ValueError) as e:
        errno, strerror = e.args
        print(f'\tCould not open {file_path} {errno} {strerror}')
    except:
        print(f"\tExiting, cannot read {file_path}, unexpected error: {sys.exc_info()[0]}")

    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        img = imread(file_path)

    if img is None:
        print(f"\tExiting, cannot read {file_path}, unexpected error: {sys.exc_info()[0]}")
        sys.exit()

    return img

def random_string():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
