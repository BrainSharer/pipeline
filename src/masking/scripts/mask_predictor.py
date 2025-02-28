import argparse
import os
import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2
from tqdm import tqdm
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


def predict(animal, debug=False):
    # Edit this path to the model
    modelpath = os.path.join("/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/structures/TG/mask.model.pth")
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
    base_path = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps'
    input = os.path.join(base_path, 'C1/thumbnail_aligned')
    files = sorted(os.listdir(input))
    output = os.path.join(base_path, 'predictions')
    os.makedirs(output, exist_ok=True)
    transform = torchvision.transforms.ToTensor()
    for file in tqdm(files, disable=debug):
        filepath = os.path.join(input, file)
        img = Image.open(filepath)
        testimg = np.array(img)
        if testimg.dtype == np.uint16:
            testimg = (testimg / 256).astype(np.uint8)
            img = Image.fromarray(testimg)
        torch_input = transform(img)
        torch_input = torch_input.unsqueeze(0)
        loaded_model.eval()
        with torch.no_grad():
            prediction = loaded_model(torch_input)
        masks = [(prediction[0]["masks"] > 0.5).squeeze().detach().cpu().numpy()]
        mask = masks[0]
        if mask.shape[0] == 0:
            continue
        if mask.ndim == 3:
            mask = mask[0, ...]
        if debug:
            print(f'{file} mask type={type(mask)} shape={mask.shape} ndim={mask.ndim}')
        raw_img = np.array(img)
        mask = mask.astype(np.uint8)
        mask[mask > 0] = 255
        merged_img = merge_mask(raw_img, mask)
        del mask
        outpath = os.path.join(output, file)
        cv2.imwrite(outpath, merged_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mask from raw image")
    parser.add_argument("--animal", help="Enter the animal", required=True)
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)

    args = parser.parse_args()
    animal = args.animal
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    predict(animal, debug)
