import numpy as np
import os
import sys
import h5py 
import json
from scipy.ndimage import zoom
from tifffile import imwrite
import math
from skimage import io
from skimage.transform import resize
from pathlib import Path
from shutil import copyfile
from timeit import default_timer as timer

# from library.controller.sql_controller import SqlController
from library.image_manipulation.filelocation_manager import FileLocationManager
from library.image_manipulation.parallel_manager import ParallelManager
from library.utilities.utilities_process import write_image

class BrainStitcher(ParallelManager):
    """Basic class for working with Xiangs data
    """

    def __init__(self, animal, layer, channel, scale, debug):
        """Initiates the brain object

        Args:
            animal (string): Animal ID
        """
        self.animal = animal
        self.layer = str(layer).zfill(5)
        self.channel = channel
        channel_dict = {1: "CH1", 2: "CH2", 4: "CH4"}
        try:
            self.channel_source = channel_dict[channel]
        except KeyError as ke:
            print(f'Error: key {channel} is not in {str(channel_dict)}')
            sys.exit()

        self.fileLocationManager = FileLocationManager(animal)
        self.base_path = os.path.join(self.fileLocationManager.prep, 'layers')
        self.layer_path = os.path.join(self.base_path, self.layer)
        self.registration_path = os.path.join(self.fileLocationManager.prep,  'registration')
        self.debug = debug
        self.available_layers = []
        self.all_info_files = None
        self.check_status()
        self.scaling_factor = 1/scale


    def check_status(self):
        if len(self.available_layers) > 0:
            return
        all_layers = [layer for layer in sorted(os.listdir(self.base_path))]
        for layer in all_layers:
            infopath = os.path.join(self.base_path, layer, 'info')
            if not os.path.exists(infopath):
                continue
            h5path = os.path.join(self.base_path, layer, 'h5')
            if not os.path.exists(h5path):
                continue
            infos = sorted(os.listdir(infopath))
            h5s = sorted(os.listdir(h5path))
            if len(h5s) == 0:
                continue
            if len(infos) == 0:
                continue
            if len(infos) != len(h5s):
                continue
            print(f'Found {len(infos)} h5s and JSON files in layer={layer}')
            for info,h5 in zip(infos, h5s):
                infostem = Path(info).stem
                h5stem = Path(h5).stem
                if infostem != h5stem:
                    print(f'Error: files do not match:{layer} {info} {h5}')
                    sys.exit()


            self.available_layers.append(layer)

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


    def extract(self):
        tilepath = os.path.join(self.layer_path,  'h5')
        if not os.path.exists(tilepath):
            print(f'Error, missing {tilepath}')
            sys.exit()

        tifpath = os.path.join(self.layer_path, 'tif')
        os.makedirs(tifpath, exist_ok=True)
        files = sorted(os.listdir(tilepath))
        if len(files) == 0:
            print('No h5 files to work with.')
            sys.exit()

        print(f'Found {len(files)} h5 files')
        file_keys = []

        for file in files:
            inpath = os.path.join(tilepath, file)
            if not os.path.exists(inpath):
                print(f'Error, {inpath} does not exist')
                continue
            if not str(inpath).endswith('h5'):
                print(f'Error, {inpath} is not a h5 file')
                continue
            outfile = str(file).replace('h5', 'tif')
            outpath = os.path.join(tifpath, outfile)
            if os.path.exists(outpath):
                continue

            file_keys.append([inpath, self.channel_source, self.scaling_factor, outpath])

        # Cleaning images takes up around 20-25GB per full resolution image
        # so we cut the workers in half here
        workers = 5
        self.run_commands_concurrently(extract_tif, file_keys, workers)


    def fetch_tif(self, inpath):        
        with h5py.File(inpath, "r") as f:
            channel_key = f[self.channel_source]
            arr = channel_key['raw'][()]
        return arr

    def stitch_tile(self):
        start_time = timer()

        # matlab is yxz
        # numpy is zyx
        self.check_status()
        self.parse_all_info()
        # Parameters
        stitch_voxel_size_um = [1/self.scaling_factor, 1/self.scaling_factor, 1/self.scaling_factor]

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
        ds_bbox_ll = [b[2], b[0], b[1]]
        print(f'Big box shape={ds_bbox_ll} composed of {len(self.all_info_files.values())} files')

        try:
            tmp_stitch_data = np.zeros(ds_bbox_ll, dtype=np.uint16)
        except:
            print(f'Could not create a big box with shape={ds_bbox_ll}')
            sys.exit()


        for (layer, position), info in self.all_info_files.items():
            position_start_time = timer()
            h5file = f"{position}.h5"
            h5path = os.path.join(self.base_path, layer, 'h5', h5file)
            if not os.path.exists(h5path):
                print(f'Error: missing {h5path}')
                sys.exit()

            fetch_start_time = timer()
            if not self.debug:
                tif = self.fetch_tif(h5path)
            fetch_end_time = timer()
            fetch_elapsed_time = round((fetch_end_time - fetch_start_time), 2)
            print(f'Time to load h5 file: {fetch_elapsed_time} seconds.', end="\t")

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

            tmp_tile_ll_ds_pxl = np.round(tmp_tile_bbox_ll_um / stitch_voxel_size_um)
            """ Downsample image stack - need smoothing? """
            if not self.debug:
                change_z = tmp_tile_ll_ds_pxl[2] / tif.shape[0]
                change_rows = tmp_tile_ll_ds_pxl[0] / tif.shape[1]
                change_cols = tmp_tile_ll_ds_pxl[1] / tif.shape[2]
                zoom_start_time = timer()
                tif = zoom(tif, (change_z, change_rows, change_cols))
                zoom_end_time = timer()
                zoom_elapsed_time = round((zoom_end_time - zoom_start_time), 2)
                print(f'Time to zoom file: {zoom_elapsed_time} seconds.')

            """ Local bounding box """ 
            tmp_local_bbox_um = tmp_tile_bbox_mm_um - vol_bbox_mm_um;
            tmp_local_bbox_mm_ds_pxl = np.round(tmp_local_bbox_um / stitch_voxel_size_um)
            """ Deal with edge: """ 
            tmp_local_bbox_mm_ds_pxl = np.maximum(tmp_local_bbox_mm_ds_pxl, 1)
            tmp_local_bbox_xx_ds_pxl = tmp_local_bbox_mm_ds_pxl + tmp_tile_ll_ds_pxl - 1;
            
            if self.debug:
                print(f'tmp_local_bbox_xx_ds_pxl={tmp_local_bbox_xx_ds_pxl}')
                continue

            start_row = int(round(tmp_local_bbox_mm_ds_pxl[0])) - 1
            end_row = int(round(tmp_local_bbox_xx_ds_pxl[0])) 
            start_col = int(round(tmp_local_bbox_mm_ds_pxl[1])) - 1
            end_col = int(round(tmp_local_bbox_xx_ds_pxl[1]))
            start_z = int(round(tmp_local_bbox_mm_ds_pxl[2])) - 1
            end_z = int(round(tmp_local_bbox_xx_ds_pxl[2]))

            position_end_time = timer()
            positions_elapsed_time = round((position_end_time - position_start_time), 2)
            print(f'layer={layer} position={position}', end=" ") 
            print(f'tmp_stitch_data[{start_z}:{end_z},{start_row}:{end_row},{start_col}:{end_col}]', end="\t")
            print(f'row height is:{end_row - start_row} column width={end_col - start_col}', end="\t")

            if not self.debug:
                try:
                    tmp_stitch_data[start_z:end_z, start_row:end_row, start_col:end_col] += tif
                except Exception as e:
                    print(f'Error: {e}')
                    sys.exit()
                    
            
            print(f'adding tif shape={tif.shape} took {positions_elapsed_time} seconds.')
        
        if self.debug:
            return
        outpath = self.fileLocationManager.get_full_aligned(channel=self.channel)
        os.makedirs(outpath, exist_ok=True)
        for i in range(tmp_stitch_data.shape[0]):
            section = tmp_stitch_data[i, :, :]
            outfile = os.path.join(outpath, f'{str(i).zfill(3)}.tif')
            write_image(outfile, section)
 
        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        print(f'\nTotal time: {total_elapsed_time} seconds.\n')
        # save
        max_layer = max([int(layer) for layer in self.available_layers])
        outfile = f'{self.channel_source}.layers.1-' + str(max_layer)  + '.tif'

        outpath = os.path.join(self.registration_path, outfile)
        os.makedirs(self.registration_path, exist_ok=True)
        io.imsave(outpath, tmp_stitch_data)
        print(f'dtype={tmp_stitch_data.dtype} shape={tmp_stitch_data.shape}')
        print('saved', outpath)

def extract_tif(file_key):
    inpath, channel_source, scaling_factor, outpath = file_key
    
    with h5py.File(inpath, "r") as f:
        channel_key = f[channel_source]
        arr = channel_key['raw'][()]
        if scaling_factor < 1:
            arr = zoom(arr, (scaling_factor, scaling_factor, scaling_factor))
        imwrite(outpath, arr)
