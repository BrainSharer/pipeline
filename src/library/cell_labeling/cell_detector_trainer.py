# ## Setting Parameters for XG Boost
# * Maximum Depth of the Tree = 3 _(maximum depth of each decision trees)_
# * Step size shrinkage used in update to prevents overfitting = 0.3 _(how to weigh trees in subsequent iterations)_
# * Maximum Number of Iterations = 1000 _(total number trees for boosting)_
# * Early Stop if score on Validation does not improve for 5 iterations
#
# [Full description of options](https://xgboost.readthedocs.io/en/latest//parameter.html)

import os
import numpy as np
import xgboost as xgb
import pickle as pk
from glob import glob
# import pandas as pd
import polars as pl #replacement for pandas (multi-core)
from tqdm import tqdm

from library.cell_labeling.cell_detector_base import CellDetectorBase
from library.cell_labeling.cell_predictor import GreedyPredictor
from library.cell_labeling.detector import Detector   


print("XGBoost Version:", xgb.__version__)

class CellDetectorTrainer(Detector, CellDetectorBase):

    def __init__(self, animal, step=2, segmentation_threshold=2000):
        CellDetectorBase.__init__(self, animal, step=step, segmentation_threshold=segmentation_threshold)
        self.last_step = CellDetectorBase(animal, step=step - 1, segmentation_threshold=segmentation_threshold)
        self.init_parameter()
        self.predictor = GreedyPredictor()

    #POSSIBLE DEPRECATION
    # def create_positive_labels(self):
    #     combined_features = self.get_combined_features()
    #     test_counts,train_sections = pk.load(open(self.last_step.QUALIFICATIONS,'rb'))
    #     all_segment = np.array([combined_features.col,combined_features.row,combined_features.section]).T

    #     cells = test_counts['computer sure, human unmarked']
    #     cells = np.array([[ci[1]['x'],ci[1]['y'],ci[1]['section']] for ci in cells])
    #     cells_index = self.find_cloest_neighbor_among_points(all_segment,cells)

    #     original = train_sections['original training set after mind change']
    #     original = np.array([[ci[1]['x'],ci[1]['y'],ci[1]['section']] for ci in original])
    #     original_index = self.find_cloest_neighbor_among_points(all_segment,original)

    #     qc_annotation_input_path = os.path.join(os.path.dirname(__file__),'retraining')
    #     neg = qc_annotation_input_path+'/DK55_premotor_manual_negative_round1_2021-12-09.csv'
    #     pos = qc_annotation_input_path+'/DK55_premotor_manual_positive_round1_2021-12-09.csv'
    #     neg = pd.read_csv(neg,header=None).to_numpy()
    #     pos = pd.read_csv(pos,header=None).to_numpy()
    #     positive = self.find_cloest_neighbor_among_points(all_segment,pos)
    #     negative = self.find_cloest_neighbor_among_points(all_segment,neg)
    #     dirs=glob('/net/birdstore/Active_Atlas_Data/cell_segmentation/DK55/CH3/*/DK55*.csv') 
    #     manual_sections = [int(i.split('/')[-2]) for i in dirs]
    #     labels = np.zeros(len(combined_features))
    #     positive_index = cells_index+original_index+positive
    #     for i in positive_index:
    #         labels[i] = 1
    #     include = [labels[i]==1 or i in negative or all_segment[i,2] in manual_sections for i in range(len(combined_features))]
    #     pk.dump((labels,include),open(self.POSITIVE_LABELS,'wb'))    

#POSSIBLE DEPRECATION
    # def get_positive_labels(self):
    #     if not os.path.exists(self.POSITIVE_LABELS):
    #         self.create_positive_labels()
    #     return pk.load(open(self.POSITIVE_LABELS,'rb'))

#POSSIBLE DEPRECATION
    # def load_new_features_with_coordinate(self):
    #     labels,include = self.get_positive_labels()
    #     combined_features = self.get_combined_features()
    #     combined_features['label'] = labels
    #     return combined_features[include]

#POSSIBLE DEPRECATION
    # def load_new_features(self):
    #     df_in_section = self.load_new_features_with_coordinate()
    #     drops = ['animal', 'section', 'index', 'row', 'col'] 
    #     df_in_section=df_in_section.drop(drops,axis=1)
    #     return df_in_section

    def gen_scale(self,n,reverse=False):
        '''
        Used for plot predictions
        '''
        s=np.arange(0,1,1/n)
        while s.shape[0] !=n:
            if s.shape[0]>n:
                s=s[:n]
            if s.shape[0]<n:
                s=np.arange(0,1,1/(n+0.1))
        if reverse:
            s=s[-1::-1]
        return s


    def get_train_and_test(self, df: pl.DataFrame, frac: float = 0.8) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        '''
        frac is sampling percent of the data (default .8) [because we split dataframe into training and testing sets]
        
        Ed notes: I changed the fraction from 0.5 to 0.8 - Ed
        '''
        df = df.with_columns(pl.arange(0, df.height).alias("idx"))

        train = df.sample(n=int(len(df) * frac)) # Sample the training set
        test = df.filter(~df['idx'].is_in(train['idx'])) # Filter out rows in the test set that are already in the training set using `is_in`
        
        # Apply the createDM method to training, testing and all data
        #createDM only removes 'label' column, why not do before call?
        train = self.createDM(train)
        test = self.createDM(test)
        all_data = self.createDM(df)
        
        return train, test, all_data


    def init_parameter(self):
        '''
        CORE PARAMETERS FOR xgboost
        '''
        self.default_param = {
            'objective': 'binary:logistic',
            'eta': 0.3,  # Learning rate
            'max_depth': 3,  # Default depth limit
            'nthread': os.cpu_count() -1,  # Dynamic core allocation
            'eval_metric': 'logloss'  # Metric to monitor
        }

        # shrinkage_parameter = 0.3
        # self.default_param['eta'] =shrinkage_parameter
        # self.default_param['objective'] = 'binary:logistic'
        # self.default_param['nthread'] = 7 
        print("xgboost Default Parameters:", self.default_param)


    def train_classifier(self, features: pl.DataFrame, niter: int, depth: int = None, models: xgb.Booster = None, **kwrds) -> xgb.Booster:
        
        param = self.default_param

        if depth is not None:
            param["max_depth"] = depth

        df = features
        train, test, _ = self.get_train_and_test(df)  # Split data
        evallist = [(train, "train"), (test, "eval")]  # Evaluation list
        bst_list = []

        for i in tqdm(range(30), desc="Training on models"):
            if models is None:
                bst = xgb.train(
                    param,  # Parameter dictionary
                    train,  # Training data
                    num_boost_round=niter,  # Maximum boosting rounds
                    evals=evallist,  # Validation set for early stopping
                    verbose_eval=False,  # Disable verbose output
                    early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
                    **kwrds
                )
            else:
                bst = xgb.train(
                    param,
                    train,
                    num_boost_round=niter,
                    evals=evallist,
                    verbose_eval=False,
                    early_stopping_rounds=50,
                    xgb_model=models[i],  # Use existing model if provided
                    **kwrds
                )

            # Use best_iteration for predictions (rather than 'best_ntree_limit')
            best_iteration = bst.best_iteration
            y_pred = bst.predict(test, iteration_range=(0, best_iteration + 1))
            y_test = test.get_label()

            # Process predictions
            pos_preds = y_pred[y_test == 1]
            neg_preds = y_pred[y_test == 0]
            pos_preds = np.sort(pos_preds)
            neg_preds = np.sort(neg_preds)

            bst_list.append(bst)

        return bst_list

#POSSIBLE DEPRECATION
    # def test_xgboost(self,df,depths = [1,3,5],num_round = 1000,**kwrds):
    #     for depthi in depths:
    #         self.test_xgboost_at_depthi(df,depth = depthi,num_round=num_round,**kwrds)

#POSSIBLE DEPRECATION
    # def test_xgboost_at_depthi(self,features,depth=1,num_round=1000,**kwrds):
    #     param = self.default_param
    #     param['max_depth']= depth
    #     train,test,_=self.get_train_and_test(features)
    #     evallist = [(train, 'train'), (test, 'eval')]
    #     #_, axes = plt.subplots(1,2,figsize=(12,5))
    #     i=0
    #     for _eval in ['error','logloss']:
    #         Logger=logger()
    #         logall=Logger.get_logger()  
    #         param['eval_metric'] = _eval 
    #         bst = xgb.train(param, train, num_round, evallist, verbose_eval=False, callbacks=[logall],**kwrds)
    #         #_=Logger.parse_log(ax=axes[i])
    #         i+=1
    #     #plt.show()
    #     print(depth)
    #     return bst,Logger

#POSSIBLE DEPRECATION
    # def save_predictions(self,features):
    #     detection_df = self.load_new_features_with_coordinate()
    #     scores,labels,_mean,_std = self.calculate_scores(features)
    #     predictions=self.get_prediction(_mean,_std)
    #     detection_df['mean_score'],detection_df['std_score'] = _mean,_std
    #     detection_df['label'] = labels
    #     detection_df['predictions'] = predictions
    #     detection_df=detection_df[predictions!=-2]
    #     detection_df = detection_df[['animal', 'section', 'row', 'col','label', 'mean_score','std_score', 'predictions']]
    #     detection_df.to_csv(self.DETECTION_RESULT_DIR,index=False)

#POSSIBLE DEPRECATION
    # def add_detection_to_database(self):
    #     detection_df = trainer.load_detections()
    #     points = np.array([detection_df.col,detection_df.row,detection_df.section]).T
    #     for pointi in points:
    #         pointi = pointi*np.array([0.325,0.325,20])
    #         trainer.sqlController.add_layer_data_row(trainer.animal,34,1,pointi,52,f'detected_soma_round_{self.step}')


    def save_detector(self):
        detector = Detector(self.model,self.predictor)
        return super().save_detector(detector)

    def load_detector(self):
        detector = super().load_detector()
        self.model = detector.model
        self.predictor = detector.predictor
