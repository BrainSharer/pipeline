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
import pandas as pd
import pickle as pk
print(f'xgboost version={xgb.__version__}')
from glob import glob
import pandas as pd
from tqdm import tqdm

from library.cell_labeling.cell_detector_base import CellDetectorBase
from library.cell_labeling.cell_predictor import GreedyPredictor
from library.cell_labeling.detector import Detector   


class CellDetectorTrainer(Detector, CellDetectorBase):

    def __init__(self, animal, round=2, segmentation_threshold=2000):
        CellDetectorBase.__init__(self, animal, round=round, segmentation_threshold=segmentation_threshold)
        self.last_round = CellDetectorBase(animal, round=round - 1, segmentation_threshold=segmentation_threshold)
        self.init_parameter()
        self.predictor = GreedyPredictor()

    def create_positive_labels(self):
        combined_features = self.get_combined_features()
        test_counts,train_sections = pk.load(open(self.last_round.QUALIFICATIONS,'rb'))
        all_segment = np.array([combined_features.col,combined_features.row,combined_features.section]).T

        cells = test_counts['computer sure, human unmarked']
        cells = np.array([[ci[1]['x'],ci[1]['y'],ci[1]['section']] for ci in cells])
        cells_index = self.find_cloest_neighbor_among_points(all_segment,cells)

        original = train_sections['original training set after mind change']
        original = np.array([[ci[1]['x'],ci[1]['y'],ci[1]['section']] for ci in original])
        original_index = self.find_cloest_neighbor_among_points(all_segment,original)

        qc_annotation_input_path = os.path.join(os.path.dirname(__file__),'retraining')
        neg = qc_annotation_input_path+'/DK55_premotor_manual_negative_round1_2021-12-09.csv'
        pos = qc_annotation_input_path+'/DK55_premotor_manual_positive_round1_2021-12-09.csv'
        neg = pd.read_csv(neg,header=None).to_numpy()
        pos = pd.read_csv(pos,header=None).to_numpy()
        positive = self.find_cloest_neighbor_among_points(all_segment,pos)
        negative = self.find_cloest_neighbor_among_points(all_segment,neg)
        dirs=glob('/net/birdstore/Active_Atlas_Data/cell_segmentation/DK55/CH3/*/DK55*.csv') 
        manual_sections = [int(i.split('/')[-2]) for i in dirs]
        labels = np.zeros(len(combined_features))
        positive_index = cells_index+original_index+positive
        for i in positive_index:
            labels[i] = 1
        include = [labels[i]==1 or i in negative or all_segment[i,2] in manual_sections for i in range(len(combined_features))]
        pk.dump((labels,include),open(self.POSITIVE_LABELS,'wb'))    

    def get_positive_labels(self):
        if not os.path.exists(self.POSITIVE_LABELS):
            self.create_positive_labels()
        return pk.load(open(self.POSITIVE_LABELS,'rb'))

    def load_new_features_with_coordinate(self):
        labels,include = self.get_positive_labels()
        combined_features = self.get_combined_features()
        combined_features['label'] = labels
        return combined_features[include]

    def load_new_features(self):
        df_in_section = self.load_new_features_with_coordinate()
        drops = ['animal', 'section', 'index', 'row', 'col'] 
        df_in_section=df_in_section.drop(drops,axis=1)
        return df_in_section

    def gen_scale(self,n,reverse=False):
        s=np.arange(0,1,1/n)
        while s.shape[0] !=n:
            if s.shape[0]>n:
                s=s[:n]
            if s.shape[0]<n:
                s=np.arange(0,1,1/(n+0.1))
        if reverse:
            s=s[-1::-1]
        return s

    def get_train_and_test(self,df,frac=0.8):
        """I changed the fraction from 0.5 to 0.8 - Ed"""
        train = pd.DataFrame(df.sample(frac=frac))
        test = df.drop(train.index,axis=0)
        test = test[train.columns]
        #print(train.shape,test.shape,train.index.shape,df.shape)
        train=self.createDM(train)
        test=self.createDM(test)
        all=self.createDM(df)
        return train,test,all

    def init_parameter(self):
        self.default_param = {}
        shrinkage_parameter = 0.3
        self.default_param['eta'] =shrinkage_parameter
        self.default_param['objective'] = 'binary:logistic'
        self.default_param['nthread'] = 7 
        print(self.default_param)

    def train_classifier(self, features, niter, depth=None, models=None, **kwrds):
        param = self.default_param
        if depth is not None:
            param["max_depth"] = depth
        df = features
        train, test, all = self.get_train_and_test(df)
        evallist = [(train, "train"), (test, "eval")]
        bst_list = []
        for i in tqdm(range(30), desc="Training on models"):
            train, test, all = self.get_train_and_test(df)
            if models is None:
                bst = xgb.train(
                    param, train, niter, evallist, verbose_eval=False, **kwrds
                )
            else:
                bst = xgb.train(
                    param,
                    train,
                    niter,
                    evallist,
                    verbose_eval=False,
                    **kwrds,
                    xgb_model=models[i],
                )
            bst_list.append(bst)
            best_ntree_limit = 676 # I had to hard code this
            y_pred = bst.predict(test, iteration_range=[1, best_ntree_limit], output_margin=True)
            y_test = test.get_label()
            pos_preds = y_pred[y_test == 1]
            neg_preds = y_pred[y_test == 0]
            pos_preds = np.sort(pos_preds)
            neg_preds = np.sort(neg_preds)
            #plt.plot(pos_preds, self.gen_scale(pos_preds.shape[0]))
            #plt.plot(neg_preds, self.gen_scale(neg_preds.shape[0], reverse=True))
        return bst_list

    def test_xgboost(self,df,depths = [1,3,5],num_round = 1000,**kwrds):
        for depthi in depths:
            self.test_xgboost_at_depthi(df,depth = depthi,num_round=num_round,**kwrds)

    def test_xgboost_at_depthi(self,features,depth=1,num_round=1000,**kwrds):
        param = self.default_param
        param['max_depth']= depth
        train,test,_=self.get_train_and_test(features)
        evallist = [(train, 'train'), (test, 'eval')]
        #_, axes = plt.subplots(1,2,figsize=(12,5))
        i=0
        for _eval in ['error','logloss']:
            Logger=logger()
            logall=Logger.get_logger()  
            param['eval_metric'] = _eval 
            bst = xgb.train(param, train, num_round, evallist, verbose_eval=False, callbacks=[logall],**kwrds)
            #_=Logger.parse_log(ax=axes[i])
            i+=1
        #plt.show()
        print(depth)
        return bst,Logger

    def save_predictions(self,features):
        detection_df = self.load_new_features_with_coordinate()
        scores,labels,_mean,_std = self.calculate_scores(features)
        predictions=self.get_prediction(_mean,_std)
        detection_df['mean_score'],detection_df['std_score'] = _mean,_std
        detection_df['label'] = labels
        detection_df['predictions'] = predictions
        detection_df=detection_df[predictions!=-2]
        detection_df = detection_df[['animal', 'section', 'row', 'col','label', 'mean_score','std_score', 'predictions']]
        detection_df.to_csv(self.DETECTION_RESULT_DIR,index=False)

    def add_detection_to_database(self):
        detection_df = trainer.load_detections()
        points = np.array([detection_df.col,detection_df.row,detection_df.section]).T
        for pointi in points:
            pointi = pointi*np.array([0.325,0.325,20])
            trainer.sqlController.add_layer_data_row(trainer.animal,34,1,pointi,52,f'detected_soma_round_{self.round}')

    def save_detector(self):
        detector = Detector(self.model,self.predictor)
        return super().save_detector(detector)

    def load_detector(self):
        detector = super().load_detector()
        self.model = detector.model
        self.predictor = detector.predictor
