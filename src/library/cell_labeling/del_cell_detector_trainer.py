#DEPRECATED - DELETE
#QUESTIONS: SEE DUANE
#11-AUG-2025


# ## Setting Parameters for XG Boost
# * Maximum Depth of the Tree = 3 _(maximum depth of each decision trees)_
# * Step size shrinkage used in update to prevents overfitting = 0.3 _(how to weigh trees in subsequent iterations)_
# * Maximum Number of Iterations = 1000 _(total number trees for boosting)_
# * Early Stop if score on Validation does not improve for 5 iterations
#
# [Full description of options](https://xgboost.readthedocs.io/en/latest//parameter.html)

import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen' # Prevents GUI error on HPC
import numpy as np
import xgboost as xgb
import pickle as pk
from glob import glob
import polars as pl #replacement for pandas (multi-core)
import imageio
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from library.cell_labeling.del_cell_detector_base import CellDetectorBase
from library.cell_labeling.del_cell_predictor import GreedyPredictor
from library.cell_labeling.del_detector import Detector   
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
from matplotlib import pyplot as plt

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

#POSSIBLE DEPRECATION (IF ROC CALC WORKS)
    # def gen_scale(self,n,reverse=False):
    #     '''
    #     Used for plot predictions: appears to be true positive rate vs. false positive rate
    #     '''
    #     s=np.arange(0,1,1/n)
    #     while s.shape[0] !=n:
    #         if s.shape[0]>n:
    #             s=s[:n]
    #         if s.shape[0]<n:
    #             s=np.arange(0,1,1/(n+0.1))
    #     if reverse:
    #         s=s[-1::-1]
    #     return s
    

    def evaluate_model(self, test_features: xgb.DMatrix, true_labels: pl.Series, new_models: list[xgb.Booster], model_filename: str) -> np.ndarray:
        """
        Evaluates model performance with ROC curve and confusion matrix.

        Args:
            test_features: Test data features
            true_labels: Ground truth labels
            new_models: List of trained XGBoost models
            model_filename: Name of the model for the title
        Returns:
            np.ndarray: RGB image of ROC plot
        """
        # Ensure we're working with matching chunks
        batch_size = test_features.num_row()  # Get DMatrix row count
        true_labels = true_labels[:batch_size]  # Slice to match
        y_true = true_labels.to_numpy()

        # Get predictions
        all_preds = [model.predict(test_features, output_margin=False) for model in new_models]
        predicted_probs = np.mean(all_preds, axis=0)
        
        # Check if lengths match
        if len(y_true) != len(predicted_probs):
            print(f"Warning: Length mismatch - predictions: {len(predicted_probs)}, labels: {len(y_true)}")
            
            # If labels are longer, truncate to match predictions
            if len(y_true) > len(predicted_probs):
                y_true = y_true[:len(predicted_probs)]
            else:
                # If predictions are longer (unlikely), truncate them
                predicted_probs = predicted_probs[:len(y_true)]
        
        # Filter for definite labels (-2 or 2)
        is_definite = (y_true == -2) | (y_true == 2)
        y_true_binary = (y_true[is_definite] == 2).astype(int)
        
        # Handle 1D (binary) vs 2D (multiclass) probabilities
        if predicted_probs.ndim == 1:
            y_prob = predicted_probs[is_definite]
        else:
            y_prob = predicted_probs[is_definite, 1]  # Assuming class 1 is neuron

        # Filter for definite labels (-2 or 2)
        is_definite = (y_true == -2) | (y_true == 2)
        y_true_binary = (y_true[is_definite] == 2).astype(int)
        y_prob = predicted_probs[is_definite] if predicted_probs.ndim == 1 else predicted_probs[is_definite, 1]
        y_pred = (y_prob > 0.5).astype(int)  # Binary predictions for confusion matrix

        # --- Plot Setup ---
        fig = plt.figure(figsize=(16, 6))
        fig.suptitle(model_filename, fontsize=12, y=0.98)  # Main title at top
        
        # Create subplots
        ax1 = plt.subplot(1, 2, 1)  # ROC curve
        ax2 = plt.subplot(1, 2, 2)  # Confusion matrix

        # GENERATE ROC CURVE
        if len(np.unique(y_true_binary)) >= 2:
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob)
            auc_score = roc_auc_score(y_true_binary, y_prob)
            ax1.plot(fpr, tpr, label=f'AUC={auc_score:.2f}')
            ax1.set_title('ROC Curve')
        else:
            ax1.text(0.5, 0.5, 'Insufficient class data', ha='center')
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_title('ROC Curve')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.legend(loc='lower right')

        # GENERATE CONFUSION MATRIX
        try:
            cm = confusion_matrix(y_true_binary, y_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=['Non-Neuron', 'Neuron'])
            disp.plot(ax=ax2, cmap='Blues', values_format='d')
            ax2.set_title('Confusion Matrix')
        except ValueError as e:
            ax2.text(0.5, 0.5, f'Error: {str(e)}', ha='center')

        # Add generation date in lower right corner
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        fig.text(
            0.98,  # Right edge
            0.02,  # Bottom edge
            f"Generated: {current_date}",
            ha='right',
            va='bottom',
            fontsize=10,
            color='gray'
        )

        # Convert to RGB image
        fig.tight_layout()
        fig.canvas.draw()
        img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return img_data
        

    def createDM(self, df: pl.DataFrame) -> xgb.DMatrix:
        '''
        MOVED FROM class Detector() [detector.py]
        '''
        if 'label' in df.columns:
            labels = df.get_column('label') # Extract the label column and store it
            features = df.drop('label') # Drop the label column to keep only features
        else:
            # Raise an error if 'label' is not found in the DataFrame
            raise ValueError("Column 'label' not found in the DataFrame")
    
        return xgb.DMatrix(features, label=labels)


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
            'eval_metric': 'logloss',  # Metric to monitor
            "n_jobs": -1,  # Use all available CPU cores (if run on CPU)

            'device': 'cuda', # Use GPU (replaces 'tree_method': 'gpu_hist')
            'predictor': 'gpu_predictor',  # Use GPU for prediction
        }

        # shrinkage_parameter = 0.3
        # self.default_param['eta'] =shrinkage_parameter
        # self.default_param['objective'] = 'binary:logistic'
        # self.default_param['nthread'] = 7 
        print("xgboost Default Parameters:", self.default_param)


    def train_classifier(self, features: pl.DataFrame, local_scratch: Path, model_filename: str, niter: int, depth: int = None, debug: bool = False, models: xgb.Booster = None, **kwrds) -> xgb.Booster:
        
        param = self.default_param

        if depth is not None:
            param["max_depth"] = depth
        
        # print(xgb.build_info())
        df = features
        train, test, _ = self.get_train_and_test(df)  # Split data
        evallist = [(train, "train"), (test, "eval")]  # Evaluation list
        bst_list = []

        for i in tqdm(range(30), desc="Training on models"):
            if models is None:
                bst = xgb.train(
                    param, train, num_boost_round=niter, evals=evallist, verbose_eval=False, **kwrds
                )
                #test remove early stopping rounds
                # bst = xgb.train(
                #     param,  # Parameter dictionary
                #     train,  # Training data
                #     num_boost_round=niter,  # Maximum boosting rounds
                #     evals=evallist,  # Validation set for early stopping
                #     verbose_eval=False,  # Disable verbose output
                #     early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
                #     **kwrds
                # )
            else:
                bst = xgb.train(
                    param,
                    train,
                    num_boost_round=niter,
                    evals=evallist,
                    verbose_eval=False,
                    # early_stopping_rounds=50,
                    xgb_model=models[i],  # Use existing model if provided
                    **kwrds
                )

            # Use best_iteration for predictions (rather than 'best_ntree_limit')
            # best_iteration = bst.best_iteration [only if early stopping is used]
            #best_ntree_limit = 676 [prev. Kui hard-coded value]
            #y_pred = bst.predict(test, iteration_range=[1, best_ntree_limit], output_margin=True)
            best_iteration = niter
            y_pred = bst.predict(test, iteration_range=(1, best_iteration))
            y_test = test.get_label()

            # Process predictions
            pos_preds = y_pred[y_test == 1]
            neg_preds = y_pred[y_test == 0]
            pos_preds = np.sort(pos_preds)
            neg_preds = np.sort(neg_preds)

            bst_list.append(bst)

        #ROC_CURVE CREATE ON SCRATCH THEN MOVE TO CENTRALIZED REPOSITORY
        true_labels = pl.Series(test.get_label())

        roc_img = self.evaluate_model(test, true_labels, bst_list, model_filename)

        roc_filename = f"roc_curve_{model_filename.stem}.tif"
        ROC_OUTPUT = Path(local_scratch, roc_filename)
        if debug:
            print(f"Saving model metrics: [ROC curve to {ROC_OUTPUT}], [Confusion Matrix to {local_scratch}]")
        imageio.imsave(ROC_OUTPUT, roc_img)
            
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

#POSSIBLE DEPRECATION
    # def save_detector(self):
    #     detector = Detector(self.model,self.predictor)
    #     return super().save_detector(detector)

#POSSIBLE DEPRECATION
    # def load_detector(self):
    #     detector = super().load_detector()
    #     self.model = detector.model
    #     self.predictor = detector.predictor