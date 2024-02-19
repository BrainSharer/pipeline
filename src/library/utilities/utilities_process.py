import os, sys
from subprocess import check_output
import socket
from skimage import io
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2
import numpy as np
import gc
from skimage.transform import rescale
import math

SCALING_FACTOR = 32.0
DOWNSCALING_FACTOR = 1 / SCALING_FACTOR
Image.MAX_IMAGE_PIXELS = None


def get_hostname() -> str:
    """Returns hostname of server where code is processed

    :return: string of the current workstation
    """

    hostname = socket.gethostname()
    hostname = hostname.split(".")[0]
    return hostname


def get_image_size(filepath: str):
    """Returns width, height of single image

    :param filepath: path of input file
    :return: tuple of int width and height of the file
    """

    try:
        result_parts = str(check_output(["identify", filepath]))
    except:
        print(f'Could not identify file={filepath}')
    results = result_parts.split()
    try:
        width, height = results[2].split("x")
    except ValueError as ve:
        print(f'Could not get width/height of {filepath}')
        print(ve)
        sys.exit()
        
    return int(width), int(height)


def test_dir(animal: str, directory, section_count, downsample: bool = True, same_size: bool = False) -> int:
    """Verify image stack directory for section count and max width, height

    :param animal: string of animal name.
    :param directory: directory we are testing.
    :param section_count: integer how many sections are in the stack
    :param downsample: boolean on whether to downsample or not
    :param same_size: boolean on whether all files are same size
    :return: string of error messages
    """

    error = ""
    # thumbnail resolution ntb is 10400 and min size of DK52 is 16074
    # thumbnail resolution thion is 14464 and min size for MD585 is 21954
    # so 3000 is a good min size. I had to turn this down as we are using
    # blank images and they are small
    # min size on NTB is 8.8K
    starting_size = 20
    min_size = starting_size * SCALING_FACTOR * 1000
    if downsample:
        min_size = starting_size
    try:
        files = sorted(os.listdir(directory))
    except:
        return f"{directory} does not exist"

    if section_count == 0:
        section_count = len(files)
    widths = set()
    heights = set()
    for f in files:
        filepath = os.path.join(directory, f)
        width, height = get_image_size(filepath)
        widths.add(int(width))
        heights.add(int(height))
        size = os.path.getsize(filepath)
        if size < min_size:
            error += f"File is too small. {size} is less than min: {min_size} {filepath} \n"
    # picked 100 as an arbitrary number. the min file count is usually around 380 or so
    if len(files) > 100:
        min_width = min(widths)
        max_width = max(widths)
        min_height = min(heights)
        max_height = max(heights)
    else:
        min_width = 0
        max_width = 0
        min_height = 0
        max_height = 0
    if section_count != len(files):
        print(
            "[EXPECTED] SECTION COUNT:",
            section_count,
            "[ACTUAL] FILES:",
            len(files),
        )
        error += f"Number of files in {directory} is incorrect.\n"
    if min_width != max_width and min_width > 0 and same_size:
        error += f"Widths are not of equal size, min is {min_width} and max is {max_width}.\n"
    if min_height != max_height and min_height > 0 and same_size:
        error += f"Heights are not of equal size, min is {min_height} and max is {max_height}.\n"
    if len(error) > 0:
        print(error)
        sys.exit()
        
    return len(files)

def get_cpus():
    """Helper method to return the number of CPUs to use
    """

    nmax = 4
    usecpus = (nmax, nmax)
    cpus = {}
    cpus["mothra"] = (1, 1)
    cpus["godzilla"] = (6, 6)
    cpus["muralis"] = (10, 20)
    cpus["basalis"] = (4, 12)
    cpus["ratto"] = (4, 8)
    hostname = get_hostname()
    if hostname in cpus.keys():
        usecpus = cpus[hostname]
    return usecpus

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
    """Writes an image to the filesystem
    """
    
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

    if img is None:
        sys.exit()
    return img
