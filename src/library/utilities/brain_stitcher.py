import numpy as np
import os
import h5py 
import json
from scipy.ndimage import zoom
from tifffile import imwrite
import math
from skimage import io


#from library.controller.sql_controller import SqlController
from library.image_manipulation.filelocation_manager import FileLocationManager


class BrainStitcher:
    """Basic class for working with Xiangs data
    """

    def __init__(self, animal, layer, channel, debug=False):
        """Initiates the brain object

        Args:
            animal (string): Animal ID
        """
        self.layer_path = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK20230126-003/preps/{channel}/layers'
        self.animal = animal
        self.layer = layer
        self.channel = channel
        #self.sqlController = SqlController(self.animal)
        #self.path = FileLocationManager(animal)

    def create_channel_volume_from_h5(self):
        INPUT = os.path.join(self.layer_path, self.layer, 'h5')
        OUTPUT = os.path.join(self.layer_path, 'tif')
        os.makedirs(OUTPUT, exist_ok=True)
        files = sorted(os.listdir(INPUT))
        print(f'Found {len(files)} h5 files')
        change_z = 1
        change_x = 1/10
        change_y = change_x

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
                ch1_key = f['CH1']
                ch1_arr = ch1_key['raw'][()]
                print(ch1_arr.dtype, ch1_arr.shape, end="\t")
                scaled_arr = zoom(ch1_arr, (change_z, change_x, change_y))
                imwrite(outpath, scaled_arr)
                print('scaled', scaled_arr.dtype, scaled_arr.shape)

    def stitch_tile(self):
        INPUT = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK20230126-003/preps/layers/00001'
        infopath = os.path.join(INPUT, 'info')
        infos = sorted(os.listdir(infopath))
        print(f'found {len(infos)} info.json files')        
        bbox_mmxx_um = []
        bbox_mmxx_pxl = []
        layer_z_um = []
        stitch_tiles = []
        for file in sorted(infos):
            inpath = os.path.join(infopath, file)
            with open(inpath) as json_data:
                d = json.load(json_data)
                stitch_tiles.append(d)
                json_data.close()
                dl = d['tile_mmxx_pxl']
                y_min, x_min, y_max, x_max = dl
                height = y_max - y_min
                width = x_max - x_min
                print(f'height={height}, width={width}')
                bbox_mmxx_um.append(d['tile_mmxx_um'])
                bbox_mmxx_pxl.append(d['tile_mmxx_pxl'])
                layer_z_um.append(d['layer_z_um'])

        # set to np arrays
        bbox_mmxx_um = np.array(bbox_mmxx_um)

        TIFPATH = os.path.join(INPUT, 'tif')
        tiles = sorted(os.listdir(TIFPATH))
        # Parameters
        scaling_factor = 10
        stitch_voxel_size_um = [0.375*scaling_factor, 0.375*scaling_factor, 1];
        stack_size_um = stitch_tiles[0]['stack_size_um'];
        print('stack_size_um', stack_size_um)
        stack_size = stitch_tiles[0]['stack_size'];
        ds_stack_size = [round(stack/stitch) for stack,stitch in zip(stack_size_um, stitch_voxel_size_um)]
        print('ds_stack_size', ds_stack_size)


        vol_bbox_z_mx_um = [56770, 56770, 249]
        vol_bbox_mm_um = np.min(bbox_mmxx_um,0)[0:2].tolist()
        vol_bbox_xx_um = np.max(bbox_mmxx_um,0)[2:4].tolist()
        vol_bbox_mm_um.append(vol_bbox_z_mx_um[0])
        vol_bbox_xx_um.append(vol_bbox_z_mx_um[1])
        print('vol_bbox_mm_um', vol_bbox_mm_um)
        print('vol_bbox_xx_um', vol_bbox_xx_um)

        vol_bbox_ll_um = [a_i - b_i for a_i, b_i in zip(vol_bbox_xx_um, vol_bbox_mm_um)]
        vol_bbox_ll_um = [a+1 for a in vol_bbox_ll_um]

        print('vol_bbox_ll_um', vol_bbox_ll_um)
        ds_bbox_ll = (np.array(vol_bbox_ll_um) / stitch_voxel_size_um)
        ds_bbox_ll = [math.ceil(a) for a  in ds_bbox_ll]
        #ds_bbox_ll = (np.array(vol_bbox_ll_um) / stitch_voxel_size_um)
        ds_bbox_ll[2] = 250
        print('ds_bbox_ll', ds_bbox_ll)
        tmp_stitch_data = np.zeros(ds_bbox_ll, dtype=np.uint16)
        print(f'Big box shape={tmp_stitch_data.shape}')

        for info,tile in zip(stitch_tiles, tiles):
            tifpath = os.path.join(TIFPATH, tile)
            tif = io.imread(tifpath)
            #tif = np.swapaxes(tif, 0, 2)
            #tif = np.swapaxes(tif, 0, 1)
            
            tmp_tile_bbox_mm_um = info['tile_mmxx_um'][:2]
            tmp_tile_bbox_mm_um.append(info['layer_z_um'])
            tmp_tile_bbox_ll_um = info['tile_mmll_um'][2:]
            tmp_tile_bbox_ll_um.append(info['stack_size_um'][2])
            #bbox = [round(bbox/voxel) for bbox,voxel in zip(tmp_tile_bbox_ll_um, stitch_voxel_size_um)]
            
            print(f'TIF shape= {tif.shape}', end="\t" )
            #print('bounding box', bbox, end="\t")
            
            # Local bounding box 
            tmp_local_bbox_um = [a_i - b_i for a_i, b_i in zip(tmp_tile_bbox_mm_um, vol_bbox_mm_um)]
            #tmp_local_bbox_mm_ds_pxl = round(tmp_local_bbox_um ./ stitch_voxel_size_um);
            tmp_local_bbox_mm_ds_pxl = [round(a/b) for a,b in zip(tmp_local_bbox_um, stitch_voxel_size_um)]
            start_row = tmp_local_bbox_mm_ds_pxl[0]
            start_col = tmp_local_bbox_mm_ds_pxl[1]
            end_row = tif.shape[1] + start_row
            end_col = tif.shape[2] + start_col
            print(start_row, end_row, start_col, end_col)
            tmp_stitch_data[0:, start_row:end_row, start_col:end_col] += tif
