import numpy as np
import os
import sys
import h5py 
import json
from scipy.ndimage import zoom
from tifffile import imwrite
import math
from skimage import io
from pathlib import Path
from shutil import copyfile


# from library.controller.sql_controller import SqlController
from library.image_manipulation.filelocation_manager import FileLocationManager


class BrainStitcher:
    """Basic class for working with Xiangs data
    """

    def __init__(self, animal, layer, channel, debug=False):
        """Initiates the brain object

        Args:
            animal (string): Animal ID
        """
        self.animal = animal
        self.layer = str(layer).zfill(5)
        self.channel = 'C' + str(channel)
        channel_dict = {1: "CH1", 2: "CH2", 4: "CH4"}
        self.channel_source = channel_dict[channel]
        self.fileLocationManager = FileLocationManager(animal)
        self.base_path = os.path.join(self.fileLocationManager.prep, self.channel, 'layers')
        self.layer_path = os.path.join(self.base_path, self.layer)
        self.registration_path = os.path.join(self.fileLocationManager.prep, self.channel, 'registration')
        self.debug = debug
        self.available_layers = []
        self.all_info_files = None
        self.check_status()
        self.scaling_factor = 1/10


    def check_status(self):
        if len(self.available_layers) > 0:
            return
        all_layers = [layer for layer in sorted(os.listdir(self.base_path))]
        for layer in all_layers:
            infopath = os.path.join(self.base_path, layer, 'info')
            if not os.path.exists(infopath):
                continue
            tifpath = os.path.join(self.base_path, layer, 'tif')
            if not os.path.exists(tifpath):
                continue
            infos = sorted(os.listdir(infopath))
            tifs = sorted(os.listdir(tifpath))
            if len(tifs) == 0:
                continue
            if len(infos) == 0:
                continue
            if len(infos) != len(tifs):
                continue
            print(f'Found {len(infos)} tifs and JSON files in layer={layer}')
            for info,tif in zip(infos, tifs):
                infostem = Path(info).stem
                tifstem = Path(tif).stem
                if infostem != tifstem:
                    print(f'Error: files do not match:{layer} {info} {tif}')
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
                    copyfile(tilefile, newtilefile)
                

    def parse_all_info(self):
        self.all_info_files = {}
        for layer in self.available_layers:
            layer = str(layer).zfill(5)
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
        change_z = 1

        for file in files:
            print(file, end="\t")
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

            with h5py.File(inpath, "r") as f:
                channel_key = f[self.channel_source]
                channel_arr = channel_key['raw'][()]
                print(channel_arr.dtype, channel_arr.shape, end="\t")
                scaled_arr = zoom(channel_arr, (change_z, self.scaling_factor, self.scaling_factor))
                imwrite(outpath, scaled_arr)
                print('scaled', scaled_arr.dtype, scaled_arr.shape)

    def stitch_tile(self):
        self.check_status()
        self.parse_all_info()
        # Parameters
        stitch_voxel_size_um = [0.375/self.scaling_factor, 0.375/self.scaling_factor, 1];
        first_element = next(iter(self.all_info_files.values()))
        stack_size_um = first_element['stack_size_um']
        print('stack_size_um', stack_size_um)
        ds_stack_size = [round(stack/stitch) for stack,stitch in zip(stack_size_um, stitch_voxel_size_um)]
        if self.debug:
            print('ds_stack_size', ds_stack_size)

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

        if self.debug:
            print('vol_bbox_ll_um', vol_bbox_ll_um)
        ds_bbox_ll = (np.array(vol_bbox_ll_um) / stitch_voxel_size_um)
        ds_bbox_ll = [math.ceil(a) for a in ds_bbox_ll]
        ds_bbox_ll[2] = 250
        b = ds_bbox_ll
        ds_bbox_ll = [b[2], b[0], b[1]]
    
        tmp_stitch_data = np.zeros(ds_bbox_ll, dtype=np.uint16)
        print(f'Big box shape={tmp_stitch_data.shape}')

        for (layer, position), info in self.all_info_files.items():
            tile = f"{position}.tif"
            tifpath = os.path.join(self.base_path, layer, 'tif', tile)
            if not os.path.exists(tifpath):
                print(f'Error: missing {tifpath}')
                sys.exit()
            tif = io.imread(tifpath)            
            tmp_tile_bbox_mm_um = info['tile_mmxx_um'][:2]
            tmp_tile_bbox_mm_um.append(info['layer_z_um'])
            tmp_tile_bbox_ll_um = info['tile_mmll_um'][2:]
            tmp_tile_bbox_ll_um.append(info['stack_size_um'][2])
            tmp_tile_ll_ds_pxl = [round(bbox/voxel) for bbox,voxel in zip(tmp_tile_bbox_ll_um, stitch_voxel_size_um)]
            print(f'{layer} {position} shape= {tif.shape}', end="\t" )
            print('bounding box', tmp_tile_ll_ds_pxl, end="\t")

            # Local bounding box
            tmp_local_bbox_um = [a_i - b_i for a_i, b_i in zip(tmp_tile_bbox_mm_um, vol_bbox_mm_um)]
            # tmp_local_bbox_mm_ds_pxl = round(tmp_local_bbox_um ./ stitch_voxel_size_um);
            tmp_local_bbox_mm_ds_pxl = [round(a/b) for a,b in zip(tmp_local_bbox_um, stitch_voxel_size_um)]
            # print(tmp_local_bbox_mm_ds_pxl)
            start_row = tmp_local_bbox_mm_ds_pxl[0]
            start_col = tmp_local_bbox_mm_ds_pxl[1]
            end_row = tif.shape[1] + start_row
            end_col = tif.shape[2] + start_col
            print(start_row, end_row, start_col, end_col)
            tmp_stitch_data[0:, start_row:end_row, start_col:end_col] += tif

        # save
        outfile = 'layers.' +  '.'.join(map(str, self.available_layers)) + '.tif'
        max_layer = max([int(layer) for layer in self.available_layers])
        outfile = 'layers.1-' + str(max_layer)  + '.tif'

        outpath = os.path.join(self.registration_path, outfile)
        os.makedirs(self.registration_path, exist_ok=True)
        io.imsave(outpath, tmp_stitch_data)
        print(f'dtype={tmp_stitch_data.dtype} shape={tmp_stitch_data.shape}')
        print('saved', outpath)
