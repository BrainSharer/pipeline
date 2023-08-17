import argparse
import os
import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def merge_mask(image, mask):
    """Merge image with mask [so user can edit]
    stack 3 channels on single image (black background, image, then mask)

    :param image: numpy array of the image
    :param mask: numpy array of the mask
    :return: merged numpy array
    """

    b = mask
    g = image
    r = np.zeros_like(image).astype(np.uint8)
    merged = np.stack([r, g, b], axis=2)
    return merged


def get_model_instance_segmentation(num_classes):
    """This loads the mask model CNN

    :param num_classes: int showing how many classes, usually 2, brain tissue, not brain tissue
    """

    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def predict(filepath):
    # Edit this path to the model
    modelpath = os.path.join(
        "/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/mask.model.pth"
    )
    loaded_model = get_model_instance_segmentation(num_classes=2)
    workers = 2
    torch.multiprocessing.set_sharing_strategy('file_system')

    device = torch.device('cpu')
    print(f' using CPU with {workers} workers')

    if os.path.exists(modelpath):
        loaded_model.load_state_dict(torch.load(modelpath, map_location = device))
    else:
        print('No model to load.')
        return

    transform = torchvision.transforms.ToTensor()
    img = Image.open(filepath)
    torch_input = transform(img)
    torch_input = torch_input.unsqueeze(0)
    loaded_model.eval()
    with torch.no_grad():
        pred = loaded_model(torch_input)
    masks = [(pred[0]["masks"] > 0.5).squeeze().detach().cpu().numpy()]
    mask = masks[0]
    raw_img = np.array(img)
    mask = mask.astype(np.uint8)
    mask[mask > 0] = 255
    merged_img = merge_mask(raw_img, mask)
    del mask
    cv2.imwrite("output.tif", merged_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mask from raw image")
    parser.add_argument("--filepath", help="Enter the full path to the raw image", required=True)
    args = parser.parse_args()
    filepath = args.filepath
    predict(filepath)
