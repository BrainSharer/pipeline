from detectron2.utils.logger import setup_logger
setup_logger()

import os, random, sys
from detectron2.data.datasets import register_coco_instances

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog
import torch
import cv2

ROOT = '/net/birdstore/Active_Atlas_Data/data_root/'
DATA = os.path.join(ROOT, 'brains_info', 'masks', 'structures', 'detectron')
train_data = os.path.join(DATA, 'train')
train_json = os.path.join(DATA, 'MD589_training.json')
register_coco_instances(f"structure_train", {}, train_json, train_data)

test_data = os.path.join(ROOT, 'pipeline_data/DK37/preps/C1/thumbnail_aligned')
test_json = os.path.join(DATA, 'DK37_testing.json')
register_coco_instances(f"structure_test", {}, test_json, test_data)

cfg = get_cfg()
if not torch.cuda.is_available(): 
    cfg.MODEL.DEVICE = "cpu"
#cfg.merge_from_file("model config")
cfg.OUTPUT_DIR = os.path.join(DATA, 'output')
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.DATASETS.TEST = ("structure_test", )
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25 
predictor = DefaultPredictor(cfg)

im = cv2.imread(test_data[0]["file_name"])
print(im.dtype, im.shape)
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
               metadata=MetadataCatalog.get(cfg.DATASETS.TEST[0]),
               scale=0.5,
               instance_mode=ColorMode.IMAGE_BW)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite(f"/home/eddyod/tmp/predict_out.tif", out.get_image())
