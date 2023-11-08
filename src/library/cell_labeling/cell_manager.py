import os, sys
import json

sys.path.append(os.path.abspath('./../../'))
# from library.utilities.utilities_process import SCALING_FACTOR, test_dir
from library.image_manipulation.filelocation_manager import FileLocationManager

class CellMaker:
    """Kui's cell labeler
    """

    def __init__(self, stack):
        """Set up the class with the name of the file and the path to it's location.

        """
        self.fileLocationManager = FileLocationManager(stack)
        self.channel = 1

    def check_prerequisites(self):
        '''
        CELL LABELING REQUIRES A) AVAILABLE FULL-RESOLUTION IMAGES, B) 2 CHANNELS (NAME, TYPE), C) SCRATCH DIRECTORY, D) OUTPUT DIRECTORY
        '''
        #CHECK FOR OME-ZARR (NOT IMPLEMENTED AS OF 31-OCT-2023)
        # INPUT = self.fileLocationManager.get_ome_zarr(channel=self.channel)
        # print(f'OME-ZARR FOUND: {INPUT}') #SEND TO LOG FILE

        #CHECK FOR FULL-RESOLUTION TIFF IMAGES (IF OME-ZARR NOT PRESENT)
        INPUT = self.fileLocationManager.get_full_aligned(channel=self.channel)
        print(f'FULL-RESOLUTION TIFF STACK FOUND: {INPUT}') #SEND TO LOG FILE

        OUTPUT = self.fileLocationManager.get_cell_labels()
        print(f'CELL LABELS OUTPUT DIR: {OUTPUT}')

        SCRATCH = '/scratch' #REMOVE HARD-CODING LATER; SEE IF WE CAN AUTO-DETECT NVME
        print(f'TEMP STORAGE LOCATION: {SCRATCH}')

        #CHECK FOR PRESENCE OF meta-data.json
        meta_data_file = 'meta-data.json'
        meta_store = os.path.join(self.fileLocationManager.prep, meta_data_file)

        if os.path.isfile(meta_store):
            print(f'FOUND; READING FROM {meta_store}')

            #verify you have 2 channels requir
            with open(meta_store) as fp:
                info = json.load(fp)
            dyes = [item['description'] for item in info['Neuroanatomical_tracing']]
            assert 'GFP' in dyes and 'NeurotraceBlue' in dyes
            print('TWO CHANNELS READY')
  
        else:
            #CREATE META-DATA STORE
            print(f'NOT FOUND; CREATING @ {meta_store}')


            #kui to add neuroanatomical_tracing
            #mode: dye_stain, virus
            #description: 'GFP' or "Neurotraceblue"
            #channel_name: (str)
            #channel (int)
            info = {}
            info['Neuroanatomical_tracing'] = []
            #steps to create
            channels_count = 3
            table = {1:{'mode':'dye', 'description':'NeurotraceBlue'}, 3:{'mode':'virus', 'description':'GFP'}}
            for channel in table.keys():
                print(channel)
                data = {}
                data['mode'] = table[channel]['mode']
                data['description'] = table[channel]['description']
                data['channel_name'] = f'C{channel}'
                data['channel'] = int(channel)
                info['Neuroanatomical_tracing'].append(data)
            with open(meta_store, 'w') as fp:
                json.dump(info, fp, indent=4)
            



    def start_labels(self):
        """Get aligned images
        """


        #add cell type to logs, pass argument to function
        #only premotor cell type for now


        INPUT = self.fileLocationManager.get_full_aligned(channel=self.channel)
        files = sorted(os.listdir(INPUT))
        for file in files:
            filepath = os.path.join(INPUT, file)
            ##### do stuff on each file here
            print(filepath)




