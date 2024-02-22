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

    def __init__(self, animal, *arg, **kwarg):
        """Initiates the brain object

        Args:
            animal (string): Animal ID
        """
        self.layer_path = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK20230126-003/preps/layers/00001'
        self.animal = animal
        #self.sqlController = SqlController(self.animal)
        #self.path = FileLocationManager(animal)

    def create_channel_volume_from_h5(self):
        INPUT = os.path.join(self.layer_path, 'h5')
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
                #print(file, end="\t")
                #pprint(d['stack_size'])
                bbox_mmxx_um.append(d['tile_mmxx_um'])
                bbox_mmxx_pxl.append(d['tile_mmxx_pxl'])
                layer_z_um.append(d['layer_z_um'])

        # set to np arrays
        bbox_mmxx_um = np.array(bbox_mmxx_um)
        tifpath = os.path.join(INPUT, 'tif')
        tiles = sorted(os.listdir(tifpath))
        # Parameters
        #stitch_voxel_size_um = [2, 2, 2];
        stitch_voxel_size_um = [1, 1, 1];
        zero_num_sec = 0;
        zero_last_section_Q = True
        medfilt_Q = False
        stack_size_um = stitch_tiles[0]['stack_size_um'];
        print('stack_size_um', stack_size_um)
        stack_size = stitch_tiles[0]['stack_size'];
        #ds_stack_size = np.round(stack_size_um / stitch_voxel_size_um);
        ds_stack_size = [round(stack/stitch) for stack,stitch in zip(stack_size_um, stitch_voxel_size_um)]
        print('ds_stack_size', ds_stack_size)
        #MATLAB bbox_mmxx_pxl = cat(1, stitch_tiles.tile_mmxx_pxl);
        #MATLAB vol_bbox_z_mx_um = [min(layer_z_um), max(layer_z_um) + stack_size_um(3) - 1];
        #MATLAB vol_bbox_mm_um = [min(bbox_mmxx_um(:, 1:2), [], 1), vol_bbox_z_mx_um(1)];
        #MATLAB vol_bbox_xx_um = [max(bbox_mmxx_um(:, 3:4), [], 1), vol_bbox_z_mx_um(2)];
        #MATLAB vol_bbox_ll_um = vol_bbox_xx_um - vol_bbox_mm_um + 1;


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
        print('ds_bbox_ll', ds_bbox_ll)
        #print(f'vol_bbox_xx_um - vol_bbox_mm_um = {np.array(vol_bbox_xx_um) - np.array(vol_bbox_mm_um)}')

        assert len(infos) == len(tiles), "Error, number of tiles does not equal number of json files"
        for info,tile in zip(stitch_tiles, tiles):
            tmp_tile = info
            tmp_tile_data_path = os.path.join(tifpath, tile)
            tmp_tile_data = io.imread(tmp_tile_data_path)
            tmp_tile_data = np.swapaxes(tmp_tile_data, 0, 2)
            
            #tmp_tile_data = tmp_tile_data[i];
            #if zero_num_sec and (tmp_tile.layer > 1):
            #    tmp_tile_data[:, :, 1:zero_num_sec] = 0;
            
            #tmp_tile_data[:, :, -1] = 0;
            tmp_tile_bbox_mm_um = tmp_tile['tile_mmxx_um'][:2]
            tmp_tile_bbox_mm_um.append(tmp_tile['layer_z_um'])
            #tmp_tile_bbox_mm_um = [tmp_tile.tile_mmxx_um(1:2), tmp_tile.layer_z_um];    
            #tmp_tile_bbox_ll_um = [tmp_tile.tile_mmll_um(3:4), tmp_tile.stack_size_um(3)];
            tmp_tile_bbox_ll_um = tmp_tile['tile_mmll_um'][2:]
            tmp_tile_bbox_ll_um.append(tmp_tile['stack_size_um'][2])
            #tmp_tile_ll_ds_pxl = round(tmp_tile_bbox_ll_um ./ stitch_voxel_size_um);
            tmp_tile_ll_ds_pxl = [round(bbox/voxel) for bbox,voxel in zip(tmp_tile_bbox_ll_um, stitch_voxel_size_um)]
            print('TIF dtype and orignal shape',tile, tmp_tile_data.dtype, tmp_tile_data.shape, end="\t" )
            print('tmp_tile_ll_ds_pxl', tmp_tile_ll_ds_pxl)
            """
            # Downsample image stack - need smoothing? 
            tmp_tile_data = imresize3(tmp_tile_data, tmp_tile_ll_ds_pxl);
            tmp_tile.clear_buffer();
            # Local bounding box 
            tmp_local_bbox_um = tmp_tile_bbox_mm_um - vol_bbox_mm_um;
            tmp_local_bbox_mm_ds_pxl = round(tmp_local_bbox_um ./ stitch_voxel_size_um);
            # Deal with edge: 
            tmp_local_bbox_mm_ds_pxl = max(tmp_local_bbox_mm_ds_pxl, 1);
            tmp_local_bbox_xx_ds_pxl = tmp_local_bbox_mm_ds_pxl + tmp_tile_ll_ds_pxl - 1;
            # Max - rendering
            tmp_stitch_data(tmp_local_bbox_mm_ds_pxl(1) : tmp_local_bbox_xx_ds_pxl(1), ...
                tmp_local_bbox_mm_ds_pxl(2) : tmp_local_bbox_xx_ds_pxl(2), ...
                tmp_local_bbox_mm_ds_pxl(3) : tmp_local_bbox_xx_ds_pxl(3)) = max(tmp_stitch_data(...
                tmp_local_bbox_mm_ds_pxl(1) : tmp_local_bbox_xx_ds_pxl(1), ...
                tmp_local_bbox_mm_ds_pxl(2) : tmp_local_bbox_xx_ds_pxl(2), ...
                tmp_local_bbox_mm_ds_pxl(3) : tmp_local_bbox_xx_ds_pxl(3)), tmp_tile_data);
            """        






