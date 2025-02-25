from library.cell_labeling.cell_predictor import GreedyPredictor
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

class Detector():
    def __init__(self,model=None,predictor:GreedyPredictor=GreedyPredictor()):
        self.model = model
        self.predictor = predictor
        self.depth = None
        self.niter = None

    def createDM(self,df):
        labels=df['label']
        features=df.drop('label',axis=1)
        return xgb.DMatrix(features, label=labels)
    
    def calculate_scores(self,features):
        all=self.createDM(features)
        labels=all.get_label()
        scores=np.zeros([features.shape[0],len(self.model)])
        for i in range(len(self.model)):
            print(f'predicting for model {i}')
            bst=self.model[i]
            best_ntree_limit = 676
            scores[:,i] = bst.predict(all, iteration_range=[1, best_ntree_limit], output_margin=True)
        mean=np.mean(scores,axis=1)
        std=np.std(scores,axis=1)
        return scores,labels,mean,std

    def get_prediction(self,mean,std):
        predictions=[]
        for mean,std in zip(mean,std):
            p=self.predictor.decision(float(mean),float(std))
            predictions.append(p)
        return np.array(predictions)
    
    def calculate_and_set_scores(self,df):
        if not hasattr(self,'mean') or not hasattr(self,'std') or not hasattr(self,'labels'):
            _,self.labels,self.mean,self.std = self.calculate_scores(df)

    def set_plot_limits(self,lower,higher):
        if lower is not None and higher is not None:
            plt.ylim([lower,higher])

    def plot_score_scatter(self,df,lower_lim = None,upper_lim = None,alpha1 = 0.5,alpha2 = 0.5,color1='teal',color2 = 'orangered',size1=3,size2=3,title = None):
        self.calculate_and_set_scores(df)
        plt.figure(figsize=[15,10])
        mean_has_label = self.mean[self.labels==1]
        mean_no_label = self.mean[self.labels==0]
        std_has_label = self.std[self.labels==1]
        std_no_label = self.std[self.labels==0]
        plt.scatter(mean_no_label,std_no_label,color=color2,s=size2,alpha=alpha2)
        plt.scatter(mean_has_label,std_has_label,color=color1,s=size1,alpha=alpha1)
        plt.title('mean and std of scores for 30 classifiers')
        plt.xlabel('mean')
        plt.ylabel('std')
        plt.grid()
        if title is not None:
            plt.title(title)
        self.set_plot_limits(lower_lim,upper_lim)


    def plot_decision_scatter(self,features,lower_lim = None,upper_lim = None,title = None):
        self.calculate_and_set_scores(features)
        if not hasattr(self,'predictions'):
            self.predictions=self.get_prediction(self.mean,self.std)
        plt.figure(figsize=[15,10])
        plt.scatter(self.mean,self.std,c=self.predictions+self.labels,s=5)
        plt.title('mean and std of scores for 30 classifiers')
        plt.xlabel('mean')
        plt.ylabel('std')
        plt.grid()
        if title is not None:
            plt.title(title)
        self.set_plot_limits(lower_lim,upper_lim)
