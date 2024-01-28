import os
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

import torch
import cv2

ROOT = '/net/birdstore/Active_Atlas_Data/data_root/'
DATA = os.path.join(ROOT, 'brains_info', 'masks', 'structures', 'detectron')
PREDICTED = os.path.join(DATA, 'predicted')
os.makedirs(PREDICTED, exist_ok=True)
train_data = os.path.join(DATA, 'train')
train_json = os.path.join(DATA, 'structure_training.json')
register_coco_instances("structure_train", {}, train_json, train_data)

test_data = os.path.join(ROOT, 'pipeline_data/DK37/preps/C1/thumbnail_aligned')
test_json = os.path.join(DATA, 'DK37_testing.json')
register_coco_instances("structure_test", {}, test_json, test_data)

cfg = get_cfg()
if not torch.cuda.is_available(): 
    cfg.MODEL.DEVICE = "cpu"

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("structure_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.OUTPUT_DIR = os.path.join(DATA, 'output')
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ("structure_test", )
predictor = DefaultPredictor(cfg)

animal = "DK37"
test_path = os.path.join(ROOT, f'pipeline_data/{animal}/preps/C1/thumbnail_aligned')
files = sorted(os.listdir(test_path))
for file in files:
    test_file = os.path.join(test_path, file)
    img = cv2.imread(test_file)
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1],
                metadata=MetadataCatalog.get(cfg.DATASETS.TEST[0]), 
                scale=1.0,
                instance_mode=ColorMode.IMAGE_BW
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
    filename = f'{animal}.{file}'
    outpath = os.path.join(PREDICTED, file)
    cv2.imwrite(outpath, img)

