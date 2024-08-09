# Running the Cell Detector

## Single Threshold Cell Detector:

running the cell detector involves the following stages:

1. full_aligned_image of DKXX> 2. tiff image tiles> 3. cell examples> 4. cell features> 5.detection result

   The image threshold is used for creating cell candidates, if unspecified, the image threshold is 2000

### Step 1 Full aligned image

​    The full resolution, within stack aligned image from the pipeline process is used for the cell detection.

​    These files are generally in the following location:

​    `    /net/birdstore/ActiveBrainAtlas/data_root/pipeline_data/DKXX/preps/CH1/full_aligned` (Neuro Trace Blue channel)

​    `/net/birdstore/ActiveBrainAtlas/data_root/pipeline_data/DKXX/preps/CH3/full_aligned` (Fluorescence channel with cells to label)

​    Also, most of the scripts for generating the following steps are located in the `/pipeline/cell_extractor/scripts/` folder, cd in to that directory in your terminal for the following steps.

### Step2 Generate image tiles

The full resolution images are too big to work with, therefore we break them down to smaller chunks for processing:

the functions are packages in the script [generate_tif_tiles.py](https://github.com/ActiveBrainAtlas2/preprocessing-pipeline/blob/master/pipeline/cell_extractor/scripts/generate_tif_tiles.py)

​       To call this script from terminal, run `python generate_tif_files.py --animal DKXX --disk /net/birdstore/ActiveAtlasData/ --njobs 4`

​        --animal specifies the animal to work on

​        --disk specifies the location to store the tile images If a custom location is used, then it need to be used for all following steps.

​        Use a physical drive instead of the network drive birdstore if you want to increase processing speed 

​        --njobs number of jobs running in parallel, this can be increased for more powerful computers

### Step3 Generate cell examples

The cell examples are patches of image from both channel 1 and 3.  Each cell example represent a candidate for cell detection.  The candidates are found by the following step:

1. Blurring the image

2. Generate an image mask based an intensity threshold, all pixels with intensity > threshold will be labeled as 1 and the rest 0.

3. Finding the connected segments of the images mask with CV2

4. All connected segments with area <100000 are considered as cell candidates

5. candidates on the edges of the tiles are excluded

   Generate examples by running the following:

   Remember to add the repo/pipeline folder to path, do the same for all the following steps as well

   ``````python
   from cell_extractor.ExampleFinder import create_examples_for_all_sections 
   import argparse
   
   animal = 'DKXX'
   create_examples_for_all_sections('DKXX',disk = '/net/birdstore/Active_Atlas_Data/', segmentation_threshold=threshold, njobs=7)
   

The arguments are similar to step2, with the addition of :

- segmentation_threshold: the threshold for masking default to 2000

### Step 4 Calculate Cell Features

A set of image features are calculated from each cell candidate.  The features are used as the input to the machine learning algorithm

`````` python
from cell_extractor.FeatureFinder import create_features_for_all_sections 
from cell_extractor.CellDetectorBase import CellDetectorBase
base = CellDetectorBase()
create_features_for_all_sections('DKXX',disk = '/net/birdstore/Active_Atlas_Data/', segmentation_threshold=threshold, njobs=7)

``````

### Step5 Detect Cells

In this step, 30 previously trained models are used to calculate a prediction score for features calculated in step 4.  The mean and standard deviation of the 30 detectors are then used to make a decision if a candidate is a sure or unsure detection.

``````python
from cell_extractor.CellDetector import detect_cell
detect_cell('DKXX',disk = '/net/birdstore/Active_Atlas_Data',round=2,segmentation_threshold=2000)
``````

This step is light weight so no parallelization is required

- round: detector version, default to 2

### Examining the result

You can get the single threshold detection result by:

``````python
from cell_extractor.CellDetectorBase import CellDetectorBase
base = CellDetectorBase('DKXX',disk,segmentation_threshold)
detection = base.load_detections()
``````

This will return a dataframe with the coordinate of the cell and the predictions.  There are different places in the code refer to the coordinates in different ways.  Remember that columns always comes before row, and x before y

The detection result is stored in the predictions column in the dataFrame.  the result is -2 if it is not a cell, 0 if it is unsure and 2 if it is sure

### Running Multi-threshold detection


Pseudo code

use the following script to run multi-threshold detection on a brain, adjust the parameters accordingly.  This script can pick up where it left off if it is interrupted

``````python
from cell_extractor.ExampleFinder import create_examples_for_one_section 
from cell_extractor.FeatureFinder import create_features_for_one_section
from cell_extractor.CellDetector import detect_cell,detect_cell_multithreshold
from cell_extractor.CellDetectorBase import parallel_process_all_sections,CellDetectorBase
animal = 'DKXX'
for threshold in [2100,2200,2300,2700]:
    base = CellDetectorBase(animal)
    sections = base.get_sections_without_example(threshold)
    parallel_process_all_sections(animal,create_examples_for_one_section,disk = '/net/birdstore/Active_Atlas_Data', segmentation_threshold=threshold, sections=sections,njobs=3)
    sections = base.get_sections_without_features(threshold)
    parallel_process_all_sections(animal,create_features_for_one_section,disk = '/net/birdstore/Active_Atlas_Data', segmentation_threshold=threshold,sections=sections,njobs=3)
    detect_cell(animal,round=2,segmentation_threshold=threshold)
    detect_cell_multithreshold(animal)
``````

