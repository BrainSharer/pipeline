import os
import numpy as np
import torch
from torchvision.io import read_image
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
from pathlib import Path
import sys

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

#from library.utilities.utilities_process import write_image



def create(img, filename):
    img = img.detach()
    img = F.to_pil_image(img)
    img = np.asarray(img)
    outpath = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/MD589/preps/C1/torch'
    outfile = os.path.join(outpath, filename)
    #write_image(outfile, img)

inpath = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/MD589/preps/C1/png'
brain_list = []
filelist = sorted(os.listdir(inpath))
for filename in filelist:
    continue
    filepath = os.path.join(inpath, filename)
    brain_list.append(read_image(filepath))

"""
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
transforms = weights.transforms()
images = [transforms(d) for d in brain_list]
model = maskrcnn_resnet50_fpn(weights=weights, progress=False)
model = model.eval()
outputs = model(images)

proba_threshold = 0.5


for brain_img, output, filename in zip(brain_list, outputs, filelist):
    filepath = os.path.join(inpath, filename)
    brain_bool_masks = output['masks'] > proba_threshold
    brain_bool_masks = brain_bool_masks.squeeze(1)
    image_mask = draw_segmentation_masks(brain_img, brain_bool_masks, alpha=0.9)
    create(image_mask, filename)
"""
modelpath = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/mask.model.pth'
model = torch.hub.load('/home/eddyod/programming/yolov5', 'custom', path=modelpath, source='local') 
imgpath = os.path.join(inpath, filelist[0])
img = read_image(imgpath)
# Inference
results = model(img)
# Results, change the flowing to: results.show()
results.show()  # or .show(), .save(), .crop(), .pandas(), etc
