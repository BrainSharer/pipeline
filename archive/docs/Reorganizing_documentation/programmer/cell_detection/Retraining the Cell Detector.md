# Retraining the Cell Detector

add repo/pipeline to python path with sys.path.append.  you can use os.path.abspath() to append the path relative to your file

``````python
from cell_extractor.CellDetectorTrainer import CellDetectorTrainer
from cell_extractor.Predictor import GreedyPredictor

trainer = CellDetectorTrainer('DKXX',round=2,segmentation_threshold=2000)
``````

`trainer.list_detectors()` lists currently available detectors to create a new detector, increment the round number

To Specify the input for training, you need to modify the DataLoader class, which is inherited by the CellDetectorTrainer.  Since the input format for each generation is likely unique

use `features = trainer.load_new_features()` to load the features for retraining.  You would need to rewrite this for future training

use `trainer.test_xgboost(features,depths=[1,3,5,7])` to find the optimal training depth and steps for the detector.  This will generate a plot like this for each depth specified:

<img src="https://cdn.rawgit.com/ActiveBrainAtlas2/preprocessing-pipeline/master/docs/programmer/cell_detection/selecting%20depth%20and%20iteration.png" />

for this step, the final error settles around 0.005 at step 909.  The title specifies the step with the smallest difference between the train and test group, and reasonably low error.  The plot on the right show the same information with a log scale

Pick the depth with the lowest error and small difference for eval-error

The detector can be trained once the optimal depth and iteration is selected by running

`bsts = trainer.train_classifier(features,depth=3,niter = 531)`

This function outputs the list of 30 detectors that constitute the model along with a diagnostic plot

<img src="https://cdn.rawgit.com/ActiveBrainAtlas2/preprocessing-pipeline/master/docs/programmer/cell_detection/training%20detector%20metric.png" />

Check that the plot is like the example shown here.  Each curve shows metrics for a single detectors.  Make sure that the two bundles are not too distributed. (forms a narrow with span)

save the new model by running:

```python
trainer.model = bsts
trainer.save_detector()
```

### Evaluating and Tunning the model

To plot the mean sore and standard deviation of cells:

If manual annotation exists in the database, it would show up as teal colored dots on the right.

```python
trainer = CellDetectorTrainer('DK55',round=2,segmentation_threshold=2000)
trainer.load_detector()
features = trainer.load_new_features()
trainer.plot_score_scatter(features)
```

<img src="https://cdn.rawgit.com/ActiveBrainAtlas2/preprocessing-pipeline/master/docs/programmer/cell_detection/score%20scatter.png" />

A detector is a combination of model and predictor, two types of predictors are available:

 `GreedyPredictor` creates a dimand shaped region of unsure while `Predictor` predicts all cells 3 std from the mean as unsure.

Greedy

<img src="https://cdn.rawgit.com/ActiveBrainAtlas2/preprocessing-pipeline/master/docs/programmer/cell_detection/predictor%20tunning.png" />

`Predictor` should be used by default

#### Tunning the greedy predictor:

<img src="https://cdn.rawgit.com/ActiveBrainAtlas2/preprocessing-pipeline/master/docs/programmer/cell_detection/greedy%20predictor.png" />

The greedy predictor have 6 points specifying the diamond shape.  Tune and examine the parameter by:

```python
from cell_extractor.Predictor import GreedyPredictor
trainer.predictor = GreedyPredictor()
trainer.predictor.set_boundary_points([[0, 2], [3, 3], [0, 4.2], [-3, 3], [-10, 5], [10, 5]])
trainer.plot_decision_scatter(features)
```

The region within the diamond contains all the unsures

### Saving new detector

```python
from cell_extractor.Predictor import Predictor
trainer = CellDetectorTrainer('DK55',round=2,segmentation_threshold=2000)
trainer.model = bst
trainer.predictor = Predictor() # or your tuned predictor
trainer.save_detector()

```

Detectors are uniquely identified by the combination of animal ID, round number and threshold.  Tune the detector for each threshold when doing multi-threshold retraining
