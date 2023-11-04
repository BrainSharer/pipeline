"""This module has several methods for helping create the precomputed data.
It also has the main class to convert numpy arrays (images) into the precomputed format.
"""

import os
from skimage import io
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import json
import numpy as np
from taskqueue import LocalTaskQueue
import igneous.task_creation as tc
from cloudvolume import CloudVolume
from cloudvolume.lib import touch
from collections import defaultdict

from library.utilities.utilities_process import get_cpus, read_image
MESHDTYPE = np.int8



def calculate_chunks(downsample, mip):
    """Function returns chunk sizes for different 'precomputed cloud volume' image 
    stack resolutions.
    Image stack will be created from full-resolution images but must be chunked for 
    efficient storage and loading into browser.
    Default values are [64, 64, 64] but may be modified for different resolutions.
    More info: https://github.com/seung-lab/cloud-volume

    Note: highest resolution tier (mip)  is 0 and increments

    :param downsample: boolean
    :param mip: integer telling us which pyramid level we want
    :return d: dictionary
    """

    d = defaultdict(dict)
    result = [64, 64, 64]
    d[False][-1] = [128, 128, 1]
    d[False][0] = [128, 128, 128]
    d[False][1] = [128, 128, 64]
    d[False][2] = [128, 128, 64]
    d[False][3] = [128, 128, 64]
    d[False][4] = [128, 128, 64]
    d[False][5] = [64, 64, 64]
    d[False][6] = [64, 64, 64]
    d[False][7] = [64, 64, 64]
    d[False][8] = [64, 64, 64]
    d[False][9] = [64, 64, 64]

    d[True][-1] = [64, 64, 1]
    d[True][0] = [64, 64, 64]
    d[True][1] = [64, 64, 64]
    d[True][2] = [64, 64, 32]
    d[True][3] = [32, 32, 16]

    try:
        result = d[downsample][mip]
    except:
        result = [64,64,64]

    return result


def calculate_factors(downsample, mip):
    """Scales get calculated by default by 2x2x1 downsampling

    :param downsample: boolean
    :param mip: which pyramid level to work on
    :return list: list of factors
    """

    d = defaultdict(dict)
    result = [2,2,1]
    d[False][0] = result
    d[False][1] = result
    d[False][2] = result
    d[False][3] = result
    d[False][4] = result
    d[False][5] = result
    d[False][6] = result
    d[False][7] = [2,2,2]
    d[False][8] = [2,2,2]
    d[False][9] = [2,2,2]

    d[True][0] = [2,2,1]
    d[True][1] = [2,2,1]
    d[True][2] = [2,2,1]
    d[True][3] = [2,2,1]
    try:
        result = d[downsample][mip]
    except:
        result = [2,2,1]
    return result

def get_segment_ids(volume):
    """Gets the unique values of a numpy array. This is used in Neuroglancer
    for the labels in a mesh
    
    :param volume: numpy array
    :return list: list of segment IDs
    """

    ids = [int(i) for i in np.unique(volume[:])]
    segment_properties = [(number, f'{number}: {number}') for number in ids]
    return segment_properties


class NumpyToNeuroglancer():
    """Contains collection of methods used to transform Numpy arrays into 'precomputed cloud volume' format
    More info: https://github.com/seung-lab/cloud-volume
    """
    
    def __init__(self, animal: str, volume, scales, layer_type, data_type, num_channels=1,
        chunk_size = [64,64,1], offset = [0,0,0]):
        self.volume = volume
        self.scales = scales
        self.layer_type = layer_type
        self.data_type = data_type
        self.chunk_size = chunk_size
        self.precomputed_vol = None
        self.offset = offset
        self.starting_points = None
        self.animal = animal
        self.num_channels = num_channels


    def init_precomputed(self, path: str, volume_size, starting_points=None) -> None:
        """Initializes 'precomputed' cloud volume format (directory 
        holding multiple volumes)

        :param path: str of the file location
        :param volume_size: size of the volume
        :param starting_points: initial starting points
        :param progress_id: progress ID
        """

        info = CloudVolume.create_new_info(
            num_channels=self.num_channels,
            layer_type=self.layer_type,  # 'image' or 'segmentation'
            data_type=self.data_type,  #
            encoding='raw',  # other options: 'jpeg', 'compressed_segmentation' (req. uint32 or uint64)
            resolution=self.scales,  # Size of X,Y,Z pixels in nanometers,
            voxel_offset=self.offset,  # values X,Y,Z values in voxels
            chunk_size=self.chunk_size,  # rechunk of image X,Y,Z in voxels
            volume_size=volume_size,  # X,Y,Z size in voxels
        )
        self.starting_points = starting_points
        self.precomputed_vol = CloudVolume(f'file://{path}', mip=0, info=info, compress=True, progress=False)
        self.precomputed_vol.commit_info()
        self.precomputed_vol.commit_provenance()


    def init_volume(self, path: str) -> None:
        """Initializes 'precomputed' cloud volume ('volume' is a collection image 
        stack with same resolution)

        :param path: path of file location
        """

        info = CloudVolume.create_new_info(
            num_channels = 1,
            layer_type = self.layer_type,
            data_type = self.data_type, # str(self.volume.dtype),  # Channel images might be 'uint8'
            encoding = 'raw',  # raw, jpeg, compressed_segmentation, fpzip, kempressed
            resolution = self.scales,            # Voxel scaling, units are in nanometers
            voxel_offset = self.offset,          # x,y,z offset in voxels from the origin
            chunk_size = self.chunk_size,           # units are voxels
            volume_size = self.volume.shape[:3], # e.g. a cubic millimeter dataset
        )
        self.precomputed_vol = CloudVolume(f'file://{path}', mip=0, info=info, compress=True, progress=False)
        self.precomputed_vol[:, :, :] = self.volume[:, :, :]
        self.precomputed_vol.commit_info()
        self.precomputed_vol.commit_provenance()


    def add_segment_properties(self, cloud_volume, segment_properties) -> None:
        """Augments 'precomputed' cloud volume with attribute tags 
        [resolution, chunk size]

        :param cloud_volume: Cloudvolume object
        :param segment_properties: dictionary of labels, ids
        """

        if self.precomputed_vol is None:
            raise NotImplementedError('You have to call init_precomputed before calling this function.')

        cloud_volume.info['segment_properties'] = 'names'
        cloud_volume.commit_info()

        segment_properties_path = os.path.join(cloud_volume.layerpath.replace('file://', ''), 'names')
        os.makedirs(segment_properties_path, exist_ok=True)
        info = {
            "@type": "neuroglancer_segment_properties",
            "inline": {
                "ids": [str(number) for number, _ in segment_properties.items()],
                "properties": [{
                    "id": "label",
                    "type": "label",
                    "values": [str(label) for _, label in segment_properties.items()]
                }]
            }
        }
        with open(os.path.join(segment_properties_path, 'info'), 'w') as file:
            json.dump(info, file, indent=2)

    def add_rechunking(self, outputpath, chunks=[64, 64, 64], mip=0, skip_downsamples=True) -> None:
        """Augments 'precomputed' cloud volume with additional chunk calculations
        [so format has pointers to location of individual volumes?]

        :param outpath: path of file location
        :param downsample: boolean
        :param chunks: list of chunk sizes
        """

        if self.precomputed_vol is None:
            raise NotImplementedError('You have to call init_precomputed before calling this function.')
        _, cpus = get_cpus()
        tq = LocalTaskQueue(parallel=cpus)
        #outpath = f'file://{outpath}'
        tasks = tc.create_transfer_tasks(self.precomputed_vol.layer_cloudpath, 
            dest_layer_path=outputpath, mip=mip, skip_downsamples=skip_downsamples)
        tq.insert(tasks)
        tq.execute()


    def add_downsampled_volumes(self, layer_path, chunk_size = [128, 128, 64], num_mips = 3) -> None:
        """Augments 'precomputed' cloud volume with additional resolutions using 
        chunk calculations
        tasks = tc.create_downsampling_tasks(cv.layer_cloudpath, mip=mip, num_mips=1, factor=factors, compress=True,  chunk_size=chunks)

        :param chunk_size: list size of chunks
        :param num_mips: number of levels in the pyramid
        """

        if self.precomputed_vol is None:
            raise NotImplementedError('You have to call init_precomputed before calling this function.')
        
        _, cpus = get_cpus()
        print(f'Creating downsampling tasks with {cpus} CPUs and chunks={chunk_size}')
        tq = LocalTaskQueue(parallel=cpus)
        tasks = tc.create_downsampling_tasks(layer_path, num_mips=num_mips, chunk_size=chunk_size, compress=True)
        tq.insert(tasks)
        tq.execute()

    def add_segmentation_mesh(self, layer_path, mip = 0) -> None:
        """Augments 'precomputed' cloud volume with segmentation mesh

        :param shape: list[int]
        :param mip: int, pyramid level
        """
        magnitude = 3
        if self.precomputed_vol is None:
            raise NotImplementedError('You have to call init_precomputed before calling this function.')

        _, cpus = get_cpus()
        tq = LocalTaskQueue(parallel=cpus)
        
        """A big shape does not work with big images. (the default is 448 and that does not work with 4147x4506, while 128 does)
        128 works at 4147x4506
        448 does not work at 4147x4506
        64 works at 9331x10138, NOT at 10368x11264
        64 dies at limit 50 full res
        32 dies at limit 100 full res
        """
        
        tasks = tc.create_meshing_tasks(layer_path, mip=0, compress=True, sharded=False) # The first phase of creating mesh
        tq.insert(tasks)
        tq.execute()
                
        print(f'Creating unshared multires task with {cpus} CPUs')
        tasks = tc.create_unsharded_multires_mesh_tasks(layer_path, num_lod=1)
        tq.insert(tasks)    
        tq.execute()
        
        print(f'Creating meshing manifest tasks with {cpus} CPUs')
        tasks = tc.create_mesh_manifest_tasks(layer_path, magnitude=magnitude) # The second phase of creating mesh
        tq.insert(tasks)
        tq.execute()
        

    def normalize_stack(self, layer_path, src_path=None, dest_path=None):
        """This does basically the same thing as our cleaning process.
        """

        _, cpus = get_cpus()
        print(f'Creating image normalization with {cpus} CPUs')
        tq = LocalTaskQueue(parallel=cpus)
        # first pass: create per z-slice histogram
        tasks = tc.create_luminance_levels_tasks(layer_path, coverage_factor=0.01, shape=None, offset=(0,0,0), mip=0) 
        tq.insert(tasks)    
        tq.execute()
        # second pass: apply histogram equalization
        tasks = tc.create_contrast_normalization_tasks(src_path, dest_path, shape=None, mip=0, clip_fraction=0.01, fill_missing=False, translate=(0,0,0))
        tq.insert(tasks)    
        tq.execute()


    def process_image(self, file_key):
        """This reads the image and starts the precomputed data

        :param file_key: file_key: tuple
        """

        index, infile, orientation, progress_dir = file_key
        basefile = os.path.basename(infile)
        progress_file = os.path.join(progress_dir, basefile)
        if os.path.exists(progress_file):
             print(f"Section {index} has already been processed, skipping.")
             return

        img = read_image(infile)
        if img.ndim > 2:
            print(f'{infile }Number of dimensions = {img.ndim} shape={img.shape}')
            img = img[:,:,0]

        try:
            img = img.reshape(self.num_channels, img.shape[0], img.shape[1]).T
        except:
            print(f'could not reshape {infile}')
            return
        try:
            self.precomputed_vol[:, :, index] = img
        except:
            print(f'Error:{infile} with Shape={img.shape} with dims={img.ndim}')
            return

        touch(progress_file)
        del img
        return
    
    def process_image_mesh(self, file_key):
        """This reads the image and starts the precomputed data

        :param file_key: file_key: tuple
        """

        debug = False

        index, infile, orientation, progress_dir, scaling_factor = file_key
        basefile = os.path.basename(infile)
        progress_file = os.path.join(progress_dir, basefile)
        if os.path.exists(progress_file):
             #print(f"{basefile} has already been processed, skipping.")
             return

        try:
            #img = io.imread(infile, img_num=0)
            im = Image.open(infile)
        except IOError as ioe:
            print(f'could not open {infile} {ioe}')
            return
        
        try:
            if scaling_factor > 1:
                width, height = im.size
                im = im.resize((width//scaling_factor, height//scaling_factor))
                #img = resize(img, orientation, anti_aliasing=True)
            img = np.array(im)
            img = img.astype(MESHDTYPE)
        except:
            print(f'could not resize {basefile} with shape={img.shape} to new shape={orientation}')
            return

        try:
            img = img.reshape(1, img.shape[0], img.shape[1]).T
        except:
            print(f'could not reshape {infile}')
            return

        if debug:
            ids, counts = np.unique(img, return_counts=True)
            print(f'{basefile} dtype={img.dtype}, shape={img.shape}, ids={ids}, counts={counts}')

        try:
            self.precomputed_vol[:, :, index] = img
        except:
            print(f'could not set {infile} to precomputed, adding blank img')
            return

        touch(progress_file)
        del img
        return


