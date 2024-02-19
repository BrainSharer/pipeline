import numpy as np
import os
import h5py 
from library.controller.sql_controller import SqlController
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
        self.sqlController = SqlController(self.animal)
        self.path = FileLocationManager(animal)

    def create_channel_volume_from_h5(self):
        INPUT = os.path.join(self.layer_path, 'h5')
        OUTPUT = os.path.join(self.layer_path, 'tif')
        os.makedirs(OUTPUT, exist_ok=True)
        files = sorted(os.listdir(INPUT))

        for file in files:
            inpath = os.path.join(INPUT, file)
            with h5py.File(inpath, "r") as f:
                ch1_key = f['CH1']
                ch1_arr = ch1_key['raw'][()]
                print(ch1_arr.dtype, ch1_arr.shape)





