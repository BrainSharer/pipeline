import numpy as np
import os
import sys
import h5py 
import json
import math
from pathlib import Path
from shutil import copyfile
from timeit import default_timer as timer
import tifffile
from scipy.ndimage import zoom
import zarr
from tqdm import tqdm
from cloudvolume.lib import touch

# from library.controller.sql_controller import SqlController
from library.image_manipulation.filelocation_manager import FileLocationManager
from library.image_manipulation.parallel_manager import ParallelManager
from library.utilities.dask_utilities import get_store
from library.utilities.utilities_process import write_image

class BrainStitcher(ParallelManager):
    """Basic class for working with Xiangs data
    There are 11,176 h5 files in this dataset
    """

    def __init__(self, animal, layer, channel, scaling_factor, debug):
        """Initiates the brain object

        Args:
            animal (string): Animal ID
        """
        self.animal = animal
        self.layer = str(layer).zfill(5)
        self.channel = channel
        self.channel_dict = {1: "CH1", 2: "CH2", 4: "CH4"}
        try:
            self.channel_source = self.channel_dict[channel]
        except KeyError as ke:
            print(f'Error: key {channel} is not in {str(self.channel_dict)}')
            sys.exit()

        self.fileLocationManager = FileLocationManager(animal)
        self.base_path = os.path.join(self.fileLocationManager.prep, 'layers')
        self.layer_path = os.path.join(self.base_path, self.layer)
        self.debug = debug
        self.scaling_factor = scaling_factor
        self.available_layers = []
        self.all_info_files = None
        #self.check_status()

    def __call__(self):
        self.check_status()

    def check_status(self):
        if len(self.available_layers) > 0:
            return
        all_layers = [layer for layer in sorted(os.listdir(self.base_path))]
        for layer in all_layers:
            len_h5 = 0
            len_info = 0
            infopath = os.path.join(self.base_path, layer, 'info')
            if os.path.exists(infopath):
                len_info = len(os.listdir(infopath))
            h5path = os.path.join(self.base_path, layer, 'h5')
            progress_path = os.path.join(self.base_path, layer, 'progress')
            os.makedirs(progress_path, exist_ok=True)
            if os.path.exists(h5path):
                len_h5 = len(os.listdir(h5path))
            len_progress = len(os.listdir(progress_path))
            print(f'Found {len_info} JSON, {len_h5} H5 files, and {len_progress} completed files in layer={layer}')

            if ((len_h5 > 0 and len_info > 0) and (len_h5 == len_info)):
                self.available_layers.append(layer)
        print(f'Available layers={self.available_layers}')

    def move_data(self):
        """First make sure output dirs exist.
        Then, get only the highest numbered subdir out of each dir and copy those h5 and json files to the new location       
        """

        outpath = os.path.join(self.layer_path)
        os.makedirs(outpath, exist_ok=True)
        tilepath = os.path.join(outpath, 'h5')
        infopath = os.path.join(outpath, 'info')
        os.makedirs(tilepath, exist_ok=True)
        os.makedirs(infopath, exist_ok=True)
        
        vessel_path = '/net/birdstore/Vessel/WBIM/Acquisition/LifeCanvas/003_20240209'
        vessel_layer_path = os.path.join(vessel_path, self.layer, 'Scan')
        if not os.path.exists(vessel_layer_path):
            print(f'Error missing dir: {vessel_layer_path}')
            sys.exit()
        dirs = sorted(os.listdir(vessel_layer_path))
        for dir in dirs:
            dir = os.path.join(vessel_layer_path, dir)
            subdirs = sorted(os.listdir(dir))
            active_dir = subdirs[-1]
            infofile = os.path.join(vessel_layer_path, dir, active_dir, 'info.json')
            tilefile = os.path.join(vessel_layer_path, dir, active_dir, 'tile.h5')
            if os.path.exists(infofile) and os.path.exists(tilefile):
                dir = os.path.basename(os.path.normpath(dir))
                newinfofile = os.path.join(infopath, f'{dir}.json')
                newtilefile = os.path.join(tilepath, f'{dir}.h5')
                if not os.path.exists(newinfofile):
                    print(f'Copy {dir} {os.path.basename(infofile)} to {newinfofile}')
                    copyfile(infofile, newinfofile)
                if not os.path.exists(newtilefile):
                    # copying takes way too long. lets try a symlink instead
                    os.symlink(tilefile, newtilefile)
                

    def parse_all_info(self):
        self.all_info_files = {}
        for layer in self.available_layers:
            infopath = os.path.join(self.base_path, layer, 'info')
            infos = sorted(os.listdir(infopath))
            for file in sorted(infos):
                inpath = os.path.join(infopath, file)
                with open(inpath) as json_data:
                    d = json.load(json_data)
                    json_data.close()
                    infostem = Path(file).stem
                    self.all_info_files[(layer, infostem)] = d



    def fetch_tif(self, inpath):
        try:
            with h5py.File(inpath, "r") as f:
                channel1_key = f['CH1']
                channel1_arr = channel1_key['raw'][()]

                channel2_key = f['CH2']
                channel2_arr = channel2_key['raw'][()]
                
                channel4_key = f['CH4']
                channel4_arr = channel4_key['raw'][()]

        except Exception as ex:
            print(f'Cannot open {inpath}')
            print(ex)
            raise
        return channel1_arr, channel2_arr, channel4_arr
    
    def compute_bbox(self, info, vol_bbox_mm_um, stitch_voxel_size_um, rows, columns, pages):
        tmp_tile_bbox_mm_um = info['tile_mmxx_um'][:2]
        tmp_tile_bbox_mm_um.append(info['layer_z_um'])
        tmp_tile_bbox_ll_um = info['tile_mmll_um'][2:]
        tmp_tile_bbox_ll_um.append(info['stack_size_um'][2])
        tmp_tile_bbox_mm_um = info['tile_mmxx_um'][:2]
        tmp_tile_bbox_mm_um.append(info['layer_z_um'])
        tmp_tile_bbox_mm_um = np.array(tmp_tile_bbox_mm_um)
        tmp_tile_bbox_ll_um = info['tile_mmll_um'][2:]
        tmp_tile_bbox_ll_um.append(info['stack_size_um'][2])
        tmp_tile_bbox_ll_um = np.array(tmp_tile_bbox_ll_um)

        # tmp_tile_ll_ds_pxl = np.round(tmp_tile_bbox_ll_um / stitch_voxel_size_um)
        """ Downsample image stack - need smoothing? """

        """ Local bounding box """ 
        tmp_local_bbox_um = tmp_tile_bbox_mm_um - vol_bbox_mm_um;
        tmp_local_bbox_mm_ds_pxl = np.round(tmp_local_bbox_um / stitch_voxel_size_um)
        """ Deal with edge: """ 
        tmp_local_bbox_mm_ds_pxl = np.maximum(tmp_local_bbox_mm_ds_pxl, 1)
        
        start_row = int(round(tmp_local_bbox_mm_ds_pxl[0])) - 1
        end_row = start_row + rows
        start_col = int(round(tmp_local_bbox_mm_ds_pxl[1])) - 1
        end_col = start_col + columns
        start_z = int(round(tmp_local_bbox_mm_ds_pxl[2])) - 1
        end_z = start_z + pages

        return start_row, end_row, start_col, end_col, start_z, end_z


    def stitch_master_volumes(self):
        #Chunks=[125, 768, 512] total time: 861.33 seconds. writing took 66.15 seconds #7 @ 50.0% done.
        #Chunks=True total time: 620.54 seconds. writing took 217.37 seconds #9
        #Chunks=(31, 192, 128) total time: 155.5 seconds.writing took 10.13 seconds
        # matlab is yxz
        # numpy is zyx
        start_time = timer()
        self.check_status()
        self.parse_all_info()
        stitch_voxel_size_um = [0.375*self.scaling_factor, 0.375*self.scaling_factor, 1*self.scaling_factor];
        overlap_size = np.array([60,60,25])

        first_element = next(iter(self.all_info_files.values()))
        stack_size_um = first_element['stack_size_um']
        min_z = min(st['layer_z_um'] for st in self.all_info_files.values())
        max_z = max(st['layer_z_um'] for st in self.all_info_files.values())

        bbox_mmxx_um = np.array([st['tile_mmxx_um'] for st in self.all_info_files.values()])
        vol_bbox_z_mx_um = [min_z, max_z + stack_size_um[2] - 1];
        vol_bbox_mm_um = np.min(bbox_mmxx_um,0)[0:2].tolist()
        vol_bbox_xx_um = np.max(bbox_mmxx_um,0)[2:4].tolist()
        vol_bbox_mm_um.append(vol_bbox_z_mx_um[0])
        vol_bbox_xx_um.append(vol_bbox_z_mx_um[1])

        vol_bbox_ll_um = [a_i - b_i for a_i, b_i in zip(vol_bbox_xx_um, vol_bbox_mm_um)]
        vol_bbox_ll_um = [a+1 for a in vol_bbox_ll_um]
        ds_bbox_ll = (np.array(vol_bbox_ll_um) / stitch_voxel_size_um)
        ds_bbox_ll = [math.ceil(a) for a in ds_bbox_ll]
        b = ds_bbox_ll
        del ds_bbox_ll
        volume_shape = [b[2], b[0], b[1]]
        print(f'Volume shape={volume_shape} composed of {len(self.all_info_files.values())} files')
        os.makedirs(self.fileLocationManager.neuroglancer_data, exist_ok=True)

        volume1 = self.create_zarr_volume(volume_shape, "1")
        volume2 = self.create_zarr_volume(volume_shape, "2")
        volume4 = self.create_zarr_volume(volume_shape, "4")
        print(f'Volume 1 type={type(volume1)}')
        
        num_tiles = len(self.all_info_files.items())
        i = 1
        for (layer, position), info in tqdm(self.all_info_files.items(), disable=self.debug):
            h5file = f"{position}.h5"
            h5path = os.path.join(self.base_path, layer, 'h5', h5file)
            progress_file = f"{position}.txt"
            progress_path = os.path.join(self.base_path, layer, 'progress', progress_file)
            if os.path.exists(progress_path):
                continue
            
            if not os.path.exists(h5path):
                print(f'Error: missing {h5path}')
                sys.exit()
            subvolume1, subvolume2, subvolume4 = self.fetch_tif(h5path)

            start_row, end_row, start_col, end_col, start_z, end_z = self.compute_bbox(info, vol_bbox_mm_um, stitch_voxel_size_um, 
                                                                                    rows=subvolume1.shape[1], 
                                                                                    columns=subvolume1.shape[2], 
                                                                                    pages=subvolume1.shape[0])
            #print(f'subvolume shape={subvolume1.shape} z size={end_z-start_z} row size={end_row-start_row} col size={end_col-start_col}')
            #continue
            #volumes = [volume1, volume2, volume4]
            #subvolumes = [subvolume1, subvolume2, subvolume4]
            volumes = [volume1]
            subvolumes = [subvolume1]

            for subvolume,volume in zip(subvolumes, volumes):
                if self.debug:
                    write_start_time = timer()
                try:
                    max_subvolume = np.maximum(volume[start_z:end_z, start_row:end_row, start_col:end_col], subvolume)
                    volume[start_z:end_z, start_row:end_row, start_col:end_col] = max_subvolume
                except Exception as e:
                    print(f'Error: {e}')

                if self.debug:
                    write_end_time = timer()     
                    write_elapsed_time = round((write_end_time - write_start_time), 2)
                    print(f'writing {position} took {write_elapsed_time} seconds', end=" ")
                    print(f'#{i} @ {round(( (i/num_tiles) * 100),2)}% done.')
            i += 1
            touch(progress_path)
        end_time = timer()     
        elapsed_time = round((end_time - start_time), 2)
        print(f'Writing {i} h5 files took {elapsed_time} seconds')

        

    def write_sections_from_volume(self):
        zarrpath = os.path.join(self.fileLocationManager.neuroglancer_data, f'C{self.channel}.zarr')
        if os.path.exists(zarrpath):
            print(f'Using existing {zarrpath}')
        else:
            print(f'No zarr: {zarrpath}')
            return
        
        store = get_store(zarrpath, 0, 'r')
        volume = zarr.open(store, 'r')
        if self.debug:
            print(volume.info)


        writing_sections_start_time = timer()
        if self.scaling_factor > 1:
            outpath = self.fileLocationManager.get_thumbnail_aligned(channel=self.channel)
        else:
            outpath = self.fileLocationManager.get_full_aligned(channel=self.channel)

        os.makedirs(outpath, exist_ok=True)

        if self.debug:
            print(f'Volume shape={volume.shape} dtype={volume.dtype}')
            print('Exiting early')
            return
        else:

            for i in tqdm(range(volume.shape[0])):
                outfile = os.path.join(outpath, f'{str(i).zfill(3)}.tif')
                if os.path.exists(outfile):
                    continue
                section = volume[i, :, :]
                if self.scaling_factor != 1:
                    section = zoom(section, (1/self.scaling_factor, 1/self.scaling_factor))

                write_image(outfile, section)

        end_time = timer()
        writing_sections_elapsed_time = round((end_time - writing_sections_start_time), 2)
        print(f'writing {i+1} sections in C{self.channel} took {writing_sections_elapsed_time} seconds')
            

    def extract(self):
        tilepath = os.path.join(self.layer_path,  'h5')
        if not os.path.exists(tilepath):
            print(f'Error, missing {tilepath}')
            sys.exit()

        outpath = os.path.join(self.layer_path, 'tif', f'scale_{self.scaling_factor}')
        os.makedirs(outpath, exist_ok=True)
        files = sorted(os.listdir(tilepath))
        if len(files) == 0:
            print('No h5 files to work with.')
            sys.exit()

        print(f'Found {len(files)} h5 files')
        file_keys = []

        for file in files:
            inpath = os.path.join(tilepath, file)
            outfile = str(file).replace('h5', 'tif')
            if not os.path.exists(inpath):
                print(f'Error, {inpath} does not exist')
                continue
            if not str(inpath).endswith('h5'):
                print(f'Error, {inpath} is not a h5 file')
                continue

            file_keys.append([inpath, self.scaling_factor, outpath, outfile])

        workers = 2
        self.run_commands_concurrently(extract_tif, file_keys, workers)

    def create_zarr_volume(self, volume_shape, channel):
        storepath = os.path.join(self.fileLocationManager.neuroglancer_data, f'C{channel}.zarr')
        store = get_store(storepath, 0)
        volume_shape = [4750, 36962, 43442]
        tile_shape = [250, 1536, 1024]
        #chunks = [150, 578, 679]
        #chunks = [1, 36962//4, 43442//4]
        chunks = tile_shape
        if os.path.exists(storepath):
            print(f'Loading existing zarr from {storepath}')
            volume = zarr.open(store)
        else:
            print(f'Creating zarr channel={channel} volume_shape={volume_shape} chunks={chunks}')
            volume = zarr.zeros(shape=(volume_shape), chunks=chunks, store=store, overwrite=False, dtype=np.uint16)

        print(volume.info)
        return volume    
        

def extract_tif(file_key):
    inpath, scaling_factor, outpath, outfile = file_key
    channel_keys = {'CH1': 'C1', 'CH2':'C2', 'CH4':'C4'}
    with h5py.File(inpath, "r") as f:
        for channel, directory in channel_keys.items():
            channel_path = os.path.join(outpath, f'{directory}')
            os.makedirs(channel_path, exist_ok=True)
            channel_file = os.path.join(channel_path, outfile)
            if os.path.exists(channel_file):
                continue
            channel_key = f[channel]
            channel_arr = channel_key['raw'][()]
            scaled_arr = zoom(channel_arr, (1/scaling_factor, 1/scaling_factor, 1/scaling_factor))
            print(channel_file, scaled_arr.dtype, scaled_arr.shape)
            #write_image(outpath, scaled_arr)
            tifffile.imwrite(channel_file, scaled_arr, bigtiff=True)
