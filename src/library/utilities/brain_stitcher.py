import numpy as np
import os
import h5py 
from scipy.ndimage import zoom
from tifffile import imwrite

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






