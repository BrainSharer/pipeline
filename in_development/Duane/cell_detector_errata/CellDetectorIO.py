#ORG FILE BELOW; CONSOLIDATE - DELETE IF POSSIBLE AS THIS IS VERY CONFUSING

import os
from glob import glob
import pickle as pkl
import pandas as pd
import numpy as np
from cell_extractor.DetectionTool import Predictor
from cell_extractor.DetectionTool import Detector
import concurrent.futures

class CellDetectorIO:
    def __init__(self,animal='DK55',section = 0,\
        disk = '/net/birdstore/Active_Atlas_Data/',round = 1,segmentation_threshold=2000,replace=False):
        """class for handling file storage and loading involved in the cell detection process

        :param animal: ID of animal being processed, defaults to 'DK55'
        :type animal: str, optional
        :param section: section number being processed, defaults to 0
        :type section: int, optional
        :param round: Detector version being used, defaults to 1
        :type round: int, optional
        :param segmentation_threshold: threshold used for detecting conntected segments, defaults to 2000
        :type segmentation_threshold: int, optional
        :param replace: regenerate files or skip processed sections, defaults to False
        :type replace: bool, optional
        """
        self.animal = animal
        self.replace = replace
        self.disk = disk
        self.round = round
        self.segmentation_threshold=segmentation_threshold
        self.ncol = 2
        self.nrow = 5
        self.section = section
        self.set_folder_paths()
        self.check_path_exists()
        self.get_tile_and_image_dimensions()
    
    def set_folder_paths(self):
        """generates all the file path involved in cell detection
        """
        self.DATA_PATH = f"/{self.disk}/cell_segmentation/"
        self.ANIMAL_PATH = os.path.join(self.DATA_PATH,self.animal)
        self.DETECTOR = os.path.join(self.ANIMAL_PATH,'detectors')
        self.MODELS = os.path.join(self.DATA_PATH,'models')
        self.FEATURE_PATH = os.path.join(self.ANIMAL_PATH,'features')
        self.DETECTION = os.path.join(self.ANIMAL_PATH,'detections')
        self.AVERAGE_CELL_IMAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','lib','average_cell_image.pkl'))
        self.TILE_INFO_DIR = os.path.join(self.ANIMAL_PATH,'tile_info.csv')
        self.fluorescence_channel_output = os.path.join(self.ANIMAL_PATH,"fluorescence_channel")
        self.cell_body_channel_output = os.path.join(self.ANIMAL_PATH,"cell_body_channel")
        self.fluorescence_channel_output_SECTION_DIR=os.path.join(self.fluorescence_channel_output,f"{self.section:03}")
        self.cell_body_channel_output_SECTION_DIR=os.path.join(self.cell_body_channel_output,f"{self.section:03}")
        self.QUALIFICATIONS = os.path.join(self.FEATURE_PATH,f'categories_round{self.round}.pkl')
        self.POSITIVE_LABELS = os.path.join(self.FEATURE_PATH,f'positive_labels_for_round_{self.round}_threshold_{self.segmentation_threshold}.pkl')
        self.DETECTOR_PATH = os.path.join(self.DETECTOR,f'detector_round_{self.round}_threshold_{self.segmentation_threshold}.pkl')
        self.DETECTION_RESULT_DIR = os.path.join(self.DETECTION,f'detections_{self.animal}.{str(self.round)}_threshold_{self.segmentation_threshold}.csv')
        self.ALL_FEATURES = os.path.join(self.FEATURE_PATH,f'all_features_threshold_{self.segmentation_threshold}.csv')
        self.MODEL_PATH = os.path.join(self.MODELS,f'models_from_qc_round_{self.round}_threshold_{self.segmentation_threshold}.pkl')

    def check_path_exists(self):
        """Makes the path if it does not exit
        """
        check_paths = [self.ANIMAL_PATH,self.FEATURE_PATH,self.fluorescence_channel_output,self.cell_body_channel_output
        ,self.DETECTION,self.DETECTOR,self.MODELS]
        for path in check_paths:
            os.makedirs(path,exist_ok = True)
    
    def get_tile_information(self):
        """returns the tile dimension in pixels

        :return: dictionary of tile dimensions
        :rtype: _type_
        """
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
        """Get sure unsure and no detections

        :return: _description_
        :rtype: dfs
        """
        detections = self.load_detections()
        sures = detections[detections.predictions==2]
        unsures = detections[detections.predictions==0]
        not_cell = detections[detections.predictions==-2]
        return sures,unsures,not_cell
    
    def save_tile_information(self):
        """generate tile information from the first image in the input directory and stores it for later use
        """
        tile_information = self.get_tile_information()
        try:
            tile_information.to_csv(self.TILE_INFO_DIR,index = False)
        except IOError as e:
            print(e)
    
    def check_tile_information(self):
        """Check that the image size matches information stored previously
        """
        if os.path.exists(self.TILE_INFO_DIR):
            tile_information = pd.read_csv(self.TILE_INFO_DIR)
            tile_information.tile_origin = tile_information.tile_origin.apply(eval)
            assert (tile_information == self.get_tile_information()).all().all()
        else:
            self.save_tile_information()
    
    def list_detectors(self):
        """list available detectors

        :return: list of detectors available
        :rtype: list
        """
        return os.listdir(self.DETECTOR)

    def get_tile_and_image_dimensions(self):
        """parse the image dimension from the tile_information dictionary
        """
        if os.path.isfile(self.TILE_INFO_DIR):
            tile_information = pd.read_csv(self.TILE_INFO_DIR)
            tile_information.tile_origin = tile_information.tile_origin.apply(eval)
            self.tile_origins=tile_information.tile_origin
            self.tile_height = np.unique(tile_information.height)[0]
            self.tile_width= np.unique(tile_information.width)[0]

    def get_tile_origin(self,tilei):
        """gets the origin in pixels of a specfic tile

        :param tilei: tile number
        :type tilei: int
        :return: x and y of tile origin
        :rtype: np array
        """
        return np.array(self.tile_origins[tilei],dtype=np.int32)
    
    def get_all_sections(self):
        """get a list of all sections

        :return: list of folder names to be processed(each folder contains result from one image section)
        :rtype: list
        """
        # return os.listdir(self.fluorescence_channel_output)
        return [f for f in os.listdir(self.fluorescence_channel_output) if not f.startswith('.')]

    
    def get_sections_with_string(self,search_string):
        """get section folders that contain files with a specfic string pattern

        :param search_string: string pattern
        :type search_string: str
        :return: list of section folders
        :rtype: list
        """
        sections = self.get_all_sections()
        sections_with_string = []
        for sectioni in sections:
            if glob(os.path.join(self.fluorescence_channel_output,sectioni,search_string)):
                sections_with_string.append(int(sectioni))
        return sorted(sections_with_string)

    def get_sections_without_string(self,search_string):
        """get section folders that do not have any file matching string pattern

        :param search_string: string pattern
        :type search_string: str
        :return: list of section folders
        :rtype: list
        """
        sections = self.get_all_sections()
        sections_with_string = []
        for sectioni in sections:
            if not glob(os.path.join(self.fluorescence_channel_output,sectioni,search_string)):
                sections_with_string.append(int(sectioni))
        return sorted(sections_with_string)

    def get_sections_with_csv(self):
        """get list of sections with a .csv file. the .csv file contains information about manunal label(depricated)

        :return: _description_
        :rtype: _type_
        """
        return self.get_sections_with_string('*.csv')
    
    def get_sections_without_csv(self):
        """get sections folders without a .csv file(depricated)

        :return: _description_
        :rtype: _type_
        """
        return self.get_sections_without_string('*.csv')

    def get_sections_with_example(self,threshold=2000):
        """get sections that have finished the example extraction step

        :param threshold: image segmentation threshold, defaults to 2000
        :type threshold: int, optional
        :return: list of sections that finished example extraction
        :rtype: _type_
        """
        return self.get_sections_with_string(f'extracted_cells*{threshold}*')

    def get_sections_without_example(self,threshold=2000):
        """Get sections that have not been through example extraction

        :param threshold: image segmentation threshold, defaults to 2000
        :type threshold: int, optional
        :return: list of section folders
        :rtype: _type_
        """
        return self.get_sections_without_string(f'extracted_cells*{threshold}*')
    
    def get_sections_with_features(self,threshold=2000):
        """get sections that have finished feature extraction step

        :param threshold: image segmentation threshold, defaults to 2000
        :type threshold: int, optional
        :return: _description_
        :rtype: _type_
        """
        return self.get_sections_with_string(f'puntas_*{threshold}*')

    def get_sections_without_features(self,threshold=2000):
        """get sections that do not have feature extracted

        :param threshold: image segmentation threshold, defaults to 2000
        :type threshold: int, optional
        :return: _description_
        :rtype: _type_
        """
        return self.get_sections_without_string(f'puntas_*{threshold}*')

    def get_example_save_path(self):
        """generate save path for example extraction step result"""
        return self.fluorescence_channel_output_SECTION_DIR+f'/extracted_cells_{self.section}_threshold_{self.segmentation_threshold}.pkl'
    
    def get_feature_save_path(self):
        """generate save path for feature extraction step result"""
        return self.fluorescence_channel_output_SECTION_DIR+f'/puntas_{self.section}_threshold_{self.segmentation_threshold}.csv'
    
    def load_examples(self):
        """load extracted examples for one section
        """
        save_path = self.get_example_save_path()
        try:
            with open(save_path,'br') as pkl_file:
                self.Examples=pkl.load(pkl_file)
        except IOError as e:
            print(e)
        
    def load_all_examples_in_brain(self,label = 1):
        """load all examples extracted from one brain

        :param label: include manual label(depricated), defaults to 1
        :type label: int, optional
        :return: _description_
        :rtype: _type_
        """
        sections = self.get_sections_with_csv()
        examples = []
        for sectioni in sections:
            base = CellDetectorIO(self.animal,sectioni)
            base.load_examples()
            examplei = [i for tilei in base.Examples for i in tilei if i['label'] == label]
            examples += examplei
        return examples
    
    def load_features(self):
        """load features for one sections"""
        path=self.get_feature_save_path()
        try:
            self.features = pd.read_csv(path)
        except IOError as e:
            print(e)
    
    def save_features(self):
        """save features for one section
        """
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
        """save examples for one section
        """
        try:
            with open(self.get_example_save_path(),'wb') as pkl_file:
                pkl.dump(self.Examples,pkl_file)
        except IOError as e:
            print(e)

    def get_manual_annotation_in_tilei(self,annotations,tilei):
        """get manual annotation for tilei (depricated)

        :param annotations: _description_
        :type annotations: _type_
        :param tilei: _description_
        :type tilei: _type_
        :return: _description_
        :rtype: _type_
        """
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
    
    def get_combined_features_of_train_sections(self):
        """get feature of all examples extracted from one brain without manual annotation (depricated)

        :return: _description_
        :rtype: _type_
        """
        dirs=glob(self.fluorescence_channel_output + f'/*/{self.animal}*.csv')
        dirs=['/'.join(d.split('/')[:-1]) for d in dirs]
        df_list=[]
        for dir in dirs:
            filename=glob(dir + '/puntas*{self.segmentation_threshold}*.csv')[0]
            df=pd.read_csv(filename)
            print(filename,df.shape)
            df_list.append(df)
        full_df=pd.concat(df_list)
        full_df.index=list(range(full_df.shape[0]))
        drops = ['animal', 'section', 'index', 'row', 'col'] 
        full_df=full_df.drop(drops,axis=1)
        return full_df
    
    def get_combined_features(self):
        """get feature of all sections extracted from one brain

        :return: _description_
        :rtype: _type_
        """
        if not os.path.exists(self.ALL_FEATURES):
            self.create_combined_features()
        return pd.read_csv(self.ALL_FEATURES,index_col=False)

    def get_combined_features_for_detection(self):
        """get features for all sections without the coordinate of sample location in the section"

        :return: _description_
        :rtype: _type_
        """
        all_features = self.get_combined_features()
        drops = ['animal', 'section', 'index', 'row', 'col'] 
        all_features=all_features.drop(drops,axis=1)
        return all_features
    
    def create_combined_features(self):
        """combines features from different sections"""
        print('creating combined features')
        files=glob(self.fluorescence_channel_output+f'/*/punta*{self.segmentation_threshold}.csv')  
        df_list=[]
        for filei in files:
            if os.path.getsize(filei) == 1:
                continue
            df=pd.read_csv(filei)
            df_list.append(df)
        full_df=pd.concat(df_list)
        full_df.index=list(range(full_df.shape[0]))
        full_df.to_csv(self.ALL_FEATURES,index=False)

    def get_qualifications(self):
        """get manual qc result"""
        return pkl.load(open(self.QUALIFICATIONS,'rb'))
    
    def save_detector(self,detector):
        """save the detector being trained"""
        pkl.dump(detector,open(self.DETECTOR_PATH,'wb'))
    
    def load_detector(self):
        """load the detector specified"""
        models = self.load_models()
        detector = Detector(models,Predictor())
        return detector
    
    def save_custom_features(self,features,file_name):
        """save custom feature inputs"""
        path = os.path.join(self.FEATURE_PATH,f'{file_name}.pkl')
        pkl.dump(features,open(path,'wb'))
    
    def list_available_features(self):
        """list availble features for training"""
        return os.listdir(self.FEATURE_PATH)
    
    def load_features(self,file_name):
        """load a specific feature set for training"""
        path = os.path.join(self.FEATURE_PATH,f'{file_name}.pkl')
        if os.path.exists(path):
            features = pkl.load(open(path,'rb'))
        else:
            print(file_name + ' do not exist')
        return features
    
    def load_average_cell_image(self):
        """load the average cell image for averaging"""
        if ~os.path.exists(self.AVERAGE_CELL_IMAGE_DIR):
            self.calulate_average_cell_image()
        try:
            average_image = pkl.load(open(self.AVERAGE_CELL_IMAGE_DIR,'rb'))
        except IOError as e:
            print(e)
        self.average_image_ch1 = average_image['CH1']
        self.average_image_ch3 = average_image['CH3']
    
    def calulate_average_cell_image(self):
        """calculate and saves the average cell image for subtraction
        """
        examples = self.load_all_examples_in_brain(label=1)
        if len(examples)==0:
            average_image = pkl.load(open('average_cell_image.pkl','rb'))
        else:
            self.average_image_ch1 = self.calculate_average_cell_images(examples,channel = 1)
            self.average_image_ch3 = self.calculate_average_cell_images(examples,channel = 3)
            average_image = dict(zip(['CH1','CH3'],[self.average_image_ch1,self.average_image_ch3]))
        pkl.dump(average_image,open(self.AVERAGE_CELL_IMAGE_DIR,'wb'))

    def calculate_average_cell_images(self,examples,channel = 3):
        """calculate average imaged for one channel"""
        images = []
        for examplei in examples:
            images.append(examplei[f'image_CH{channel}'])
        images = np.stack(images)
        average = np.average(images,axis=0)
        average = (average - average.mean())/average.std()
        return average

    def load_detections(self):
        """load detection results

        :return: df containing detection results
        :rtype: df
        """
        return pd.read_csv(self.DETECTION_RESULT_DIR)
    
    def has_detection(self):
        """check if detections exist

        :return: boolean
        :rtype: boolean
        """
        return os.path.exists(self.DETECTION_RESULT_DIR)
    
    def get_available_animals(self):
        """get list of available animals

        :return: _description_
        :rtype: _type_
        """
        path = self.DATA_PATH
        dirs = os.listdir(path)
        dirs = [i for i in dirs if os.path.isdir(path+i)]
        dirs.remove('detectors')
        dirs.remove('models')
        return dirs
    
    def get_animals_with_examples():
        ...
    
    def get_animals_with_features():
        ...
    
    def get_animals_with_detections():
        ...
    
    def report_detection_status():
        ...

    def save_models(self,models):
        """save xgboost model"""
        try:
            with open(self.MODEL_PATH,'wb') as pkl_file:
                pkl.dump(models,pkl_file)
        except IOError as e:
            print(e)
    
    def load_models(self):
        """load xgboost model

        :return: _description_
        :rtype: _type_
        """
        try:
            with open(self.MODEL_PATH,'rb') as pkl_file:
                models = pkl.load(pkl_file)
            return models
        except IOError as e:
            print(e)
        
def get_sections_with_annotation_for_animali(animal,**kwargs):
    base = CellDetectorIO(animal,**kwargs)
    return base.get_sections_with_csv()

def get_sections_without_annotation_for_animali(animal,**kwargs):
    base = CellDetectorIO(animal,**kwargs)
    return base.get_sections_without_csv()

def get_all_sections_for_animali(animal,**kwargs):
    base = CellDetectorIO(animal,**kwargs)
    return base.get_all_sections()

def list_available_animals(disk = '/net/birdstore/Active_Atlas_Data/',has_example = True,has_feature = True):
    base = CellDetectorIO(disk = disk)
    animals = os.listdir(base.DATA_PATH)
    animals = [os.path.isdir(i) for i in animals]
    animals.remove('detectors')
    animals.remove('models')
    for animali in animals:
        base = CellDetectorIO(disk = disk,animal = animali)
        nsections = len(base.get_all_sections())
        remove = False
        if has_example:
            nexamples = len(base.get_sections_with_example())
            if not nexamples == nsections:
                remove = True
        if has_feature:
            nfeatures = len(base.get_sections_with_features())
            if not nfeatures == nsections:
                remove = True
        if remove:
            animals.remove(animali)
    return animals

def parallel_process_all_sections(animal,processing_function,*args,njobs = 10,sections=None,**kwargs):
    """process all sections for one brain in parallel

    :param animal: animal ID
    :type animal: _type_
    :param processing_function: function containg detection step
    :type processing_function: _type_
    :param njobs: n parallel, defaults to 10
    :type njobs: int, optional
    :param sections: list of sections to annalyze, defaults to None
    :type sections: _type_, optional
    """
    if sections is None:
        sections = get_all_sections_for_animali(animal,**kwargs)
    with concurrent.futures.ProcessPoolExecutor(max_workers=njobs) as executor:
        results = []
        for sectioni in sections:
            results.append(executor.submit(processing_function,animal,int(sectioni),*args,**kwargs))
        print('done')