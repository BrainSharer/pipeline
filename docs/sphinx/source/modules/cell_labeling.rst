Cell Labeling Scripts
---------------------

This Python module contains utility methods related to:

- generating labeled masks
- converting annotations
- processing segmentation outputs
- training a model (via CLI argument)

.. automodule:: labeling.scripts.create_labels
   :members:
   :undoc-members:
   :show-inheritance:



Detecting Cells
~~~~~~~~~~~~~~~

To run cell detection/segmentation using a trained model, use the `--task detect` argument:

.. code-block:: bash

   python -m labeling.scripts.create_labels --task detect --animal {ANIMAL_ID} --model {neuron_model_type} 

This mode:

- Loads the trained model from the centralized model directory
- Loads average cell image on 2 channels (virus and dye)
- Calculates features for cell candidates: energy coorelation and Hu moments
- Applies the model to score the cell candidates
- Saves detection results (e.g. detections_{section}.csv) to:
  `/net/birdstore/Active_Atlas_Data/cell_segmentation/data_root/pipeline_data/{ANIMAL_ID}/preps/cell_labels/`

Optional:
Ensure the image stacks (tif or OME-Zarr) are available in preps folder, and that the specified model exists if argument `--model` is used.



Training the Model
~~~~~~~~~~~~~~~~~~

To train the cell labeling model, run the script with the `--train` argument:

.. code-block:: bash

   python -m labeling.scripts.create_labels --task train --animal {ANIMAL_ID} --model {neuron_model_type} --step 1

This mode:

- Reads the 'ground truth' neuron coordinates from cell_labels{step} directory
- Trains specific model type using supplied putative identifications (positive, negative)
- Saves model to centralized location (with metrics) to /net/birdstore/Active_Atlas_Data/cell_segmentation/models/
