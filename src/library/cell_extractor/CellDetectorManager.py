
import numpy as np
from numpy import *
import xgboost as xgb
import pandas as pd
from glob import glob
import pickle as pk
from  os.path import getsize # get the size of file. 
from cell_extractor.CellDetectorBase import CellDetectorBase
class CellDetector(CellDetectorBase):

    def __init__(self,animal,round = 2, *args, **kwargs):
        print('version of xgboost is:',xgb.__version__,'should be at least 1.5.0')
        super().__init__(animal,round=round,*args, **kwargs)

    def read_csv_files(self):
        pattern=self.CH3+'/*/puntas_*.csv'
        files=glob(pattern)
        files=sorted(files)
        print(len(files),' csv files')
        df_list=[]
        for filename in files:
            size=getsize(filename)
            if size <10:
                continue
            df=pd.read_csv(filename)
            df_list.append(df)
        print(len(df_list),'non empty csv files')
        full_df=pd.concat(df_list)
        full_df.index=list(range(full_df.shape[0]))
        print('number of candidate cells:',full_df.shape[0])
        return full_df

    def read_classifier(self):
        with open(self.CLASSIFIER_PATH,'br') as pkl_file:
            bst_list=pk.load(pkl_file)
        return bst_list

    def calculate_scores(self,df,bst_list):
        drops = ['animal', 'section', 'index', 'row', 'col','label']

        def createDM(df):
            labels=df['label']
            features=df.drop(drops,axis=1)
            return xgb.DMatrix(features, label=labels)
        AllDM=createDM(df)
        labels=AllDM.get_label()
        scores=np.zeros([df.shape[0],len(bst_list)])
        for i in range(len(bst_list)):
            bst=bst_list[i]
            scores[:,i] = bst.predict(AllDM, iteration_range=[1,bst.best_ntree_limit],\
                                    output_margin=True)
        return scores

    def calc_score_stats(self,scores,full_df):
        _mean=np.mean(scores,axis=1)
        _std=np.std(scores,axis=1)
        _mean= np.mean(scores,axis=1)
        full_df['mean_score']= _mean 
        _std=np.std(scores,axis=1)
        full_df['std_score']=_std
        _median = np.median(scores,axis=1)
        full_df['median_score']=_median
        _spread=np.percentile(scores,90,axis=1) - np.percentile(scores,10,axis=1)
        full_df['score_spread']=_spread
        return full_df

    def generate_decision(self):
        def points2line(p1,p2,i):
            x1,y1=p1
            x2,y2=p2
            a=(y1-y2)/(x1-x2)
            b=y1-a*x1
            return a,b
        p=[[0,0.8],[2,2.5],[0,3.3],[-2,2.5],[-10,4],[10,4]]

        def aboveline(p,l):
            return l[0]*p[0]+l[1] < p[1]
        L=[]
        L.append(points2line(p[0],p[1],0))
        L.append(points2line(p[1],p[2],1))
        L.append(points2line(p[2],p[3],2))
        L.append(points2line(p[3],p[0],3))
        L.append(points2line(p[1],p[5],4))
        L.append(points2line(p[3],p[4],5))

        def decision(x,y):
            p=[x,y]
            if aboveline(p,L[0]) and not aboveline(p,L[1])\
            and not aboveline(p,L[2]) and aboveline(p,L[3]):
                return 0
            if (x<0 and not aboveline(p,L[5])) or (x>0 and aboveline(p,L[4])):
                return -2
            if (x>0 and not aboveline(p,L[4])) or (x<0 and aboveline(p,L[5])):
                return 2
        return decision

    def calculate_stats(self):
        classifier = self.load_model()
        self.decision = self.generate_decision()
        features = self.read_csv_files()
        scores = self.calculate_scores(features,classifier)
        self.stats = self.calc_score_stats(scores,features)

    def make_predictions(self):
        predictions = []
        for i,row in self.stats.iterrows():
            prediction = self.decision(float(row['mean_score']),float(row['std_score']))
            predictions.append(prediction)
            if i %100==0:
                print('\r%d'%i,end='')
        self.stats['predictions'] = predictions
    
    def save_prediction_results(self):
        detection_result = self.stats[self.stats['predictions']!=-2]
        detection_result = detection_result[['animal', 'section', 'row', 'col','label', \
            'mean_score','std_score', 'predictions']]
        detection_result.to_csv(self.DETECTION_RESULT_DIR,index=False)

    def calculate_and_save_detection_results(self):
        self.calculate_stats()
        self.make_predictions()
        self.save_prediction_results()

if __name__ == '__main__':
    detector = CellDetector('DK52')
    detector.calculate_and_save_detection_results()