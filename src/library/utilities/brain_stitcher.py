import numpy as np
import os
import h5py 
from scipy.ndimage import zoom

from library.controller.sql_controller import SqlController
from library.image_manipulation.filelocation_manager import FileLocationManager
from library.utilities.utilities_process import write_image


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
        self.sqlController = SqlController(self.animal)
        self.path = FileLocationManager(animal)

    def create_channel_volume_from_h5(self):
        INPUT = os.path.join(self.layer_path, 'h5')
        OUTPUT = os.path.join(self.layer_path, 'tif')
        os.makedirs(OUTPUT, exist_ok=True)
        files = sorted(os.listdir(INPUT))
        change_z = 1
        change_x = 10
        change_y = change_x
        for i, file in enumerate(files):
            if i > 9:
                continue
            inpath = os.path.join(INPUT, file)
            outpath = os.path.join(OUTPUT, file)
            if os.path.exists(outpath):
                continue
            with h5py.File(inpath, "r") as f:
                ch1_key = f['CH1']
                ch1_arr = ch1_key['raw'][()]
                print(file, ch1_arr.dtype, ch1_arr.shape, end="\t")
                scaled_arr = zoom(ch1_arr, (change_z, change_x, change_y))
                write_image(outpath, scaled_arr)
                print('scaled', scaled_arr.dtype, scaled_arr.shape)






