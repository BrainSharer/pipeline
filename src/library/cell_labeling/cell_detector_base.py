import os
import sys
import re
from glob import glob
import pickle as pkl
import pandas as pd
import numpy as np
from pathlib import Path
from library.cell_labeling.cell_predictor import Predictor
from library.cell_labeling.detector import Detector
import xgboost as xgb
import shutil

from library.controller.sql_controller import SqlController
from library.image_manipulation.filelocation_manager import FileLocationManager

class Brain:

    def __init__(self, animal, sql=True, *arg, **kwarg):
        """Initiates the brain object, starts the sqalchemy session, sets the location for the pipeline, readies the plotter
        and loads the resolution of the brian

        Args:
            animal (string): Animal ID
        """
        self.animal = animal
        if sql:
            self.sqlController = SqlController(self.animal)
            to_um = self.get_resolution()
            self.pixel_to_um = np.array([to_um, to_um, 20])
            self.um_to_pixel = 1 / self.pixel_to_um
        self.path = FileLocationManager(animal)

    def get_resolution(self):
        """get the scan resolution of the animal from the scan run table

        Returns:
            float: scan resolution
        """
        return self.sqlController.scan_run.resolution

    def get_image_dimension(self):
        """get the dimension of the biggest image in the scan

        Returns:
            np array: np array of width and height
        """
        width = self.sqlController.scan_run.width
        height = self.sqlController.scan_run.height
        return np.array([width, height])


class CellDetectorBase(Brain):

    #TODO: REMOVE HARD-CODED VARIABLES
    def __init__(
        self,
        animal="DK55",
        section=0,
        disk="/net/birdstore/Active_Atlas_Data",
        step=1,
        segmentation_threshold=2000,
        replace=False,
    ):
        super().__init__(animal)
        self.replace = replace
        self.disk = disk
        self.step = step
        self.segmentation_threshold = segmentation_threshold
        #TODO: self.ncol, self.nrow ALSO DEFINED IN cell_manager.py (total_virtual_tile_rows, total_virtual_tile_columns) ~line 437; CONSOLIDATE (AND !HARD CODING VARIBLES)
        self.ncol = 2
        self.nrow = 5
        self.section = section
        self.set_folder_paths()
        # self.check_path_exists()
        self.get_tile_and_image_dimensions()
        self.get_tile_origins()
        self.check_tile_information() #TODO: remove since we do virtual tiling in dask

    def set_folder_paths(self):
        self.DATA_PATH = f"{self.disk}/cell_segmentation"
        self.ANIMAL_PATH = os.path.join(self.DATA_PATH, self.animal)
        self.DETECTOR = os.path.join(self.ANIMAL_PATH, "detectors")
        self.MODELS = os.path.join(self.DATA_PATH, "models")
        self.FEATURE_PATH = os.path.join(self.ANIMAL_PATH, "features")
        self.DETECTION = os.path.join(self.ANIMAL_PATH, "detections")
        self.AVERAGE_CELL_IMAGE_DIR = os.path.join(
            self.ANIMAL_PATH, "average_cell_image.pkl"
        )
        self.TILE_INFO_DIR = os.path.join(self.ANIMAL_PATH, "tile_info.csv")
        self.CH3 = os.path.join(self.ANIMAL_PATH, "CH3")
        self.CH1 = os.path.join(self.ANIMAL_PATH, "CH1")
        self.CH3_SECTION_DIR = os.path.join(self.CH3, f"{self.section:03}")
        self.CH1_SECTION_DIR = os.path.join(self.CH1, f"{self.section:03}")
        self.QUALIFICATIONS = os.path.join(
            self.FEATURE_PATH, f"categories_round{self.step}.pkl"
        )
        self.POSITIVE_LABELS = os.path.join(
            self.FEATURE_PATH,
            f"positive_labels_for_round_{self.step}_threshold_{self.segmentation_threshold}.pkl",
        )
        self.DETECTOR_PATH = os.path.join(
            self.DETECTOR,
            f"detector_round_{self.step}_threshold_{self.segmentation_threshold}.pkl",
        )
        self.DETECTION_RESULT_DIR = os.path.join(
            self.DETECTION,
            f"detections_{self.animal}.{str(self.step)}_threshold_{self.segmentation_threshold}.csv",
        )
        self.ALL_FEATURES = os.path.join(
            self.FEATURE_PATH,
            f"all_features_threshold_{self.segmentation_threshold}.csv",
        )
        self.MODEL_PATH = os.path.join(
            self.MODELS,
            f"models_round_{self.step}_threshold_{self.segmentation_threshold}.pkl",
        )

#POSSIBLE DEPRECATION
    # def check_path_exists(self):
    #     check_paths = [self.ANIMAL_PATH,self.FEATURE_PATH,self.DETECTION,self.DETECTOR,self.MODELS]
    #     for path in check_paths:
    #         os.makedirs(path,exist_ok = True)

    def get_tile_information(self):
        self.get_tile_origins()
        ntiles = len(self.tile_origins)
        tile_information = pd.DataFrame(columns = ['id','tile_origin','ncol','nrow','width','height'])
        for tilei in range(ntiles):
            tile_informationi = pd.DataFrame(dict(
                id = [tilei],
                tile_origin = [self.tile_origins[tilei]],
                ncol = [self.ncol],
                nrow = [self.nrow],
                width = [self.width],
                height = [self.height]) )
            tile_information = pd.concat([tile_information,tile_informationi],ignore_index=True)
        return tile_information

    def get_detection_by_category(self):
        detections = self.load_detections()
        sures = detections[detections.predictions==2]
        unsures = detections[detections.predictions==0]
        not_cell = detections[detections.predictions==-2]
        return sures,unsures,not_cell

    def save_tile_information(self):
        tile_information = self.get_tile_information()
        try:
            tile_information.to_csv(self.TILE_INFO_DIR,index = False)
        except IOError as e:
            print(e)

    def check_tile_information(self):
        print(f'Checking tile information at {self.TILE_INFO_DIR}')
        if os.path.exists(self.TILE_INFO_DIR):
            tile_information = pd.read_csv(self.TILE_INFO_DIR)
            tile_information.tile_origin = tile_information.tile_origin.apply(eval)
            assert (tile_information == self.get_tile_information()).all().all(), 'tile information does not match'
        else:
            self.save_tile_information()

    def list_detectors(self):
        print(f'listing detectors at {self.DETECTOR}')
        return os.listdir(self.DETECTOR)

    def get_tile_and_image_dimensions(self):
        self.width,self.height = self.get_image_dimension()
        self.tile_height = int(self.height / self.nrow )
        self.tile_width=int(self.width/self.ncol )

    def get_tile_origins(self):
        if not hasattr(self,'tile_origins'):
            assert hasattr(self,'width')
            self.tile_origins={}
            for i in range(self.nrow*self.ncol):
                row=int(i/self.ncol)
                col=i%self.ncol
                self.tile_origins[i] = (row*self.tile_height,col*self.tile_width)

    def get_tile_origin(self,tilei):
        self.get_tile_origins()
        return np.array(self.tile_origins[tilei],dtype=np.int32)

    def get_all_sections(self):
        return os.listdir(self.CH3)

    def get_sections_with_string(self,search_string):
        sections = self.get_all_sections()
        sections_with_string = []
        for sectioni in sections:
            if glob(os.path.join(self.CH3,sectioni,search_string)):
                sections_with_string.append(int(sectioni))
        return sorted(sections_with_string)

    def get_sections_without_string(self,search_string):
        sections = self.get_all_sections()
        sections_with_string = []
        for sectioni in sections:
            if not glob(os.path.join(self.CH3,sectioni,search_string)):
                sections_with_string.append(int(sectioni))
        return sorted(sections_with_string)

    def get_sections_with_csv(self):
        return self.get_sections_with_string('*.csv')

    def get_sections_without_csv(self):
        return self.get_sections_without_string('*.csv')

    def get_sections_with_example(self,threshold=2000):
        return self.get_sections_with_string(f'extracted_cells*{threshold}*')

    def get_sections_without_example(self,threshold=2000):
        return self.get_sections_without_string(f'extracted_cells*{threshold}*')

    def get_sections_with_features(self,threshold=2000):
        return self.get_sections_with_string(f'puntas_*{threshold}*')

    def get_sections_without_features(self,threshold=2000):
        return self.get_sections_without_string(f'puntas_*{threshold}*')

    def get_example_save_path(self):
        return self.CH3_SECTION_DIR+f'/extracted_cells_{self.section}_threshold_{self.segmentation_threshold}.pkl'

    def get_feature_save_path(self):
        return self.CH3_SECTION_DIR+f'/puntas_{self.section}_threshold_{self.segmentation_threshold}.csv'

    def load_examples(self):
        save_path = self.get_example_save_path()
        try:
            with open(save_path,'br') as pkl_file:
                self.Examples=pkl.load(pkl_file)
        except IOError as e:
            print(e)

#POSSIBLE DEPRECATION
    # def load_all_examples_in_brain(self,label = 1):
    #     sections = self.get_sections_with_csv()
    #     examples = []
    #     for sectioni in sections:
    #         base = CellDetectorBase(self.animal,sectioni)
    #         base.load_examples()
    #         examplei = [i for tilei in base.Examples for i in tilei if i['label'] == label]
    #         examples += examplei
    #     return examples

    # This was originally named load_features,
    # but there is another function with the same name
    # that loads a pickle file
    def load_features_csv(self):
        path=self.get_feature_save_path()
        try:
            self.features = pd.read_csv(path)
        except IOError as e:
            print(e)

    def save_features(self):
        df=pd.DataFrame()
        i = 0
        for featurei in self.features:
            df_dict = pd.DataFrame(featurei,index = [i])
            i+=1
            df=pd.concat([df,df_dict])
        outfile=self.get_feature_save_path()
        print('df shape=',df.shape,'output_file=',outfile)
        try:
            df.to_csv(outfile,index=False)
        except IOError as e:
            print(e)

    def save_examples(self):
        try:
            with open(self.get_example_save_path(),'wb') as pkl_file:
                pkl.dump(self.Examples,pkl_file)
        except IOError as e:
            print(e)

    def get_manual_annotation_in_tilei(self,annotations,tilei):
        tile_origin= self.get_tile_origin(tilei)
        manual_labels_in_tile=[]
        n_manual_label = 0
        if annotations is not None:  
            manual_labels=np.int32(annotations)-tile_origin   
            for i in range(manual_labels.shape[0]):
                row,col=list(manual_labels[i,:])
                if row<0 or row>=self.tile_height or col<0 or col>=self.tile_width:
                    continue
                manual_labels_in_tile.append(np.array([row,col]))
            if not manual_labels_in_tile ==[]:
                manual_labels_in_tile=np.stack(manual_labels_in_tile)
            else:
                manual_labels_in_tile = np.array([])
            n_manual_label = len(manual_labels_in_tile) 
        return manual_labels_in_tile,n_manual_label

#POSSIBLE DEPRECATION
    # def get_combined_features_of_train_sections(self):
    #     dirs=glob(self.CH3 + f'/*/{self.animal}*.csv')
    #     dirs=['/'.join(d.split('/')[:-1]) for d in dirs]
    #     df_list=[]
    #     for dir in dirs:
    #         filename=glob(dir + '/puntas*{self.segmentation_threshold}*.csv')[0]
    #         df=pd.read_csv(filename)
    #         print(filename,df.shape)
    #         df_list.append(df)
    #     full_df=pd.concat(df_list)
    #     full_df.index=list(range(full_df.shape[0]))
    #     drops = ['animal', 'section', 'index', 'row', 'col'] 
    #     full_df=full_df.drop(drops,axis=1)
    #     return full_df

#POSSIBLE DEPRECATION
    # def get_combined_features(self):
    #     if not os.path.exists(self.ALL_FEATURES):
    #         self.create_combined_features()
    #     print(f'loading combined features from {self.ALL_FEATURES}')
    #     return pd.read_csv(self.ALL_FEATURES,index_col=False)

    def get_combined_features_for_detection(self):
        all_features = self.get_combined_features()
        drops = ['animal', 'section', 'index', 'row', 'col'] 
        all_features=all_features.drop(drops,axis=1)
        print('all_features shape=',all_features.shape)
        print(all_features.head())
        return all_features

#POSSIBLE DEPRECATION
    # def create_combined_features(self):
    #     print('creating combined features')
    #     files=glob(self.CH3+f'/*/punta*{self.segmentation_threshold}.csv')
    #     if len(files) == 0:
    #         print(f"no files found at {self.CH3+f'/*/punta*{self.segmentation_threshold}.csv'}")
    #         sys.exit()

    #     df_list=[]
    #     for filei in files:
    #         if os.path.getsize(filei) == 1:
    #             continue
    #         df=pd.read_csv(filei)
    #         df_list.append(df)
    #     full_df=pd.concat(df_list)
    #     full_df.index=list(range(full_df.shape[0]))
    #     full_df.to_csv(self.ALL_FEATURES,index=False)

#POSSIBLE DEPRECATION
    # def get_qualifications(self):
    #     return pkl.load(open(self.QUALIFICATIONS,'rb'))

#POSSIBLE DEPRECATION
    # def save_detector(self,detector):
    #     pkl.dump(detector,open(self.DETECTOR_PATH,'wb'))

#POSSIBLE DEPRECATION
    # def load_detector(self):
    #     models = self.load_models()
    #     detector = Detector(models,Predictor())
    #     return detector

    def save_custom_features(self,features,file_name):
        path = os.path.join(self.FEATURE_PATH,f'{file_name}.pkl')
        pkl.dump(features,open(path,'wb'))

    def list_available_features(self):
        return os.listdir(self.FEATURE_PATH)

    def load_features(self,file_name):
        path = os.path.join(self.FEATURE_PATH,f'{file_name}.pkl')
        if os.path.exists(path):
            features = pkl.load(open(path,'rb'))
        else:
            print(file_name + ' do not exist')
        return features

#POSSIBLE DEPRECATION
    # def load_average_cell_image(self):
    #     if os.path.exists(self.AVERAGE_CELL_IMAGE_DIR):
    #         try:
    #             average_image = pkl.load(open(self.AVERAGE_CELL_IMAGE_DIR,'rb'))
    #         except IOError as e:
    #             print(e)
    #         self.average_image_ch1 = average_image['CH1']
    #         self.average_image_ch3 = average_image['CH3']

    def load_detections(self):
        return pd.read_csv(self.DETECTION_RESULT_DIR)

#POSSIBLE DEPRECATION
    # def has_detection(self):
    #     return os.path.exists(self.DETECTION_RESULT_DIR)

#POSSIBLE DEPRECATION
    # def get_available_animals(self):
    #     path = self.DATA_PATH
    #     dirs = os.listdir(path)
    #     dirs = [i for i in dirs if os.path.isdir(path+i)]
    #     dirs.remove('detectors')
    #     dirs.remove('models')
    #     return dirs

#POSSIBLE DEPRECATION
    # def get_animals_with_examples():
    #     ...

    # def get_animals_with_features():
    #     ...

    # def get_animals_with_detections():
    #     ...

    # def report_detection_status():
    #     ...

    def save_models(self, models: list[xgb.Booster], model_filepath: Path, local_scratch: Path) -> None:
        '''
        SAVE LOCAL THEN MOVE
        '''
        permanent_model_file_location = Path(self.MODELS, model_filepath)
        tmp_model_file_location = Path(local_scratch, model_filepath.name)
        print(f'SAVING MODEL, ROC METRIC TO STAGING FOLDER: {local_scratch}')
        try:
            with open(tmp_model_file_location,'wb') as pkl_file:
                pkl.dump(models, pkl_file)
        except IOError as e:
            print(e)
        print(f'DEBUG: {permanent_model_file_location=}, {tmp_model_file_location=}')
        
        print(f'MOVING MODEL, ROC METRIC TO CENTRALIZED STORAGE: {permanent_model_file_location.parent}')
        try:
            with open(permanent_model_file_location,'wb') as pkl_file:
                pkl.dump(models,pkl_file)
        except IOError as e:
            print(e)

        pattern = re.compile(r"^roc_curve_.*\.tif$")
        matching_files = [
            file for file in local_scratch.glob("*.tif") 
            if pattern.match(file.name)
        ]
        if not matching_files:
            print(f"No matching ROC curve files found in {local_scratch}")
            return
        else:
            file_to_move = matching_files[0]
            destination_dir = permanent_model_file_location.parent
            shutil.move(str(file_to_move), str(destination_dir / file_to_move.name))


    def load_models(self, model: str = None, step: int = None) -> tuple[np.ndarray, Path]:
        """
            Load models from file.

            Args:
                model (str): Model name. Defaults to None.
                step (int): Step number. Defaults to None.

            Returns:
                tuple[np.ndarray, Path]: Loaded models as a NumPy array and the model file path.
        """
        if model and step:
            models_file = Path(self.MODELS, f'models_{model}_step_{step}.pkl')
        else:
            models_file = Path(self.MODEL_PATH)

        if models_file.exists() and models_file.stat().st_size == 0:
            print(f"Warning: {models_file} is empty.")
            return np.array([]), models_file

        try:
            with open(models_file,'rb') as pkl_file:
                models = pkl.load(pkl_file)
            return models, models_file
        except (IOError, FileNotFoundError) as e:
            print(f"Error loading model: {e}")
            return np.array([]), models_file

#POSSIBLE DEPRECATION
# def get_sections_with_annotation_for_animali(animal):
#     base = CellDetectorBase(animal)
#     return base.get_sections_with_csv()

#POSSIBLE DEPRECATION
# def get_sections_without_annotation_for_animali(animal):
#     base = CellDetectorBase(animal)
#     return base.get_sections_without_csv()

#POSSIBLE DEPRECATION
# def get_all_sections_for_animali(animal):
#     base = CellDetectorBase(animal)
#     return base.get_all_sections()

#POSSIBLE DEPRECATION
# def list_available_animals(disk = '/net/birdstore/Active_Atlas_Data/',has_example = True,has_feature = True):
#     base = CellDetectorBase(disk = disk)
#     animals = os.listdir(base.DATA_PATH)
#     animals = [os.path.isdir(i) for i in animals]
#     animals.remove('detectors')
#     animals.remove('models')
#     for animali in animals:
#         base = CellDetectorBase(disk = disk,animal = animali)
#         nsections = len(base.get_all_sections())
#         remove = False
#         if has_example:
#             nexamples = len(base.get_sections_with_example())
#             if not nexamples == nsections:
#                 remove = True
#         if has_feature:
#             nfeatures = len(base.get_sections_with_features())
#             if not nfeatures == nsections:
#                 remove = True
#         if remove:
#             animals.remove(animali)
#     return animals

#POSSIBLE DEPRECATION
# def parallel_process_all_sections(animal,processing_function,*args,njobs = 10,sections=None,**kwargs):
#     if sections is None:
#         sections = get_all_sections_for_animali(animal)
#     with concurrent.futures.ProcessPoolExecutor(max_workers=njobs) as executor:
#         results = []
#         for sectioni in sections:
#             results.append(executor.submit(processing_function,animal,int(sectioni),*args,**kwargs))
#         print('done')