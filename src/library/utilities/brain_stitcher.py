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
        if not os.path.exists(self.layer_path):
            print(f'Error: {self.layer_path} does not exist')
            sys.exit()
        self.debug = debug
        self.available_layers = [1,2,3]
        self.all_info_files = self.parse_all_info()
        self.scaling_factor = 1/10

    def check_status(self):
        for layer in self.available_layers:
            layer = str(layer).zfill(5)
            infopath = os.path.join(self.base_path, layer, 'info')
            tifpath = os.path.join(self.base_path, layer, 'tif')
            infos = sorted(os.listdir(infopath))
            tifs = sorted(os.listdir(tifpath))
            print(f'Found {len(infos)} info.json files in layer={layer}')
            assert len(infos) == len(tifs), "Error, number of tiles does not equal number of json files"
            for info,tif in zip(infos, tifs):
                infostem = Path(info).stem
                tifstem = Path(tif).stem
                if infostem != tifstem:
                    print(layer, info, tif)

    def parse_all_info(self):
        all_info_files = {}
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
                    all_info_files[(layer, infostem)] = d

        return all_info_files

    def create_channel_volume_from_h5(self):
        INPUT = os.path.join(self.layer_path,  'h5')
        OUTPUT = os.path.join(self.layer_path, 'tif')
        os.makedirs(OUTPUT, exist_ok=True)
        files = sorted(os.listdir(INPUT))
        print(f'Found {len(files)} h5 files')
        change_z = 1

        for file in files:
            print(file, end="\t")
            inpath = os.path.join(INPUT, file)
            if not os.path.exists(inpath):
                print(f'Error, {inpath} does not exist')
                continue
            if not str(inpath).endswith('h5'):
                print(f'Error, {inpath} is not a h5 file')
                continue
            outfile = str(file).replace('h5', 'tif')
            outpath = os.path.join(OUTPUT, outfile)

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
        infopath = os.path.join(self.layer_path, 'info')
        if not os.path.exists(infopath):
            print(f'Error: {infopath} does not exist')
            sys.exit()
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
        outpath = os.path.join(self.registration_path, outfile)
        io.imsave(outpath, tmp_stitch_data)
        print(f'dtype={tmp_stitch_data.dtype} shape={tmp_stitch_data.shape}')
        print('saved', outpath)
