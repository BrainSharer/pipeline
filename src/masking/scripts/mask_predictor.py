import argparse
from collections import defaultdict
import os
import shutil
import sys
import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2
from tqdm import tqdm
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pathlib import Path

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.utilities.utilities_mask import combine_dims
from library.annotation_utilities.annotation_helper import AnnotationHelper
from library.utilities.utilities_process import M_UM_SCALE, SCALING_FACTOR


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
    modelpath = os.path.join("/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/TG/models/mask.model.pth")
    loaded_model = get_model_instance_segmentation(num_classes=2)
    workers = 2
    torch.multiprocessing.set_sharing_strategy('file_system')

    device = torch.device('cpu')
    print(f' using CPU with {workers} workers')

    if os.path.exists(modelpath):
        print(f'Loading model from {modelpath}')
        ck = torch.load(modelpath, map_location=device)
        loaded_model.load_state_dict(ck['model_state'] if 'model_state' in ck else ck)

    else:
        print(f'No model to load at {modelpath}')
        return
    base_path = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps'
    input = os.path.join(base_path, 'C1', 'thumbnail_realigned')
    if not os.path.exists(input):
        print(f'No input directory found at {input}')
        return
    else:
        print(f'Predicting masks for images in {input}')
    files = sorted(os.listdir(input))
    if len(files) == 0:
        print(f'No files found in {input}')
        return
    output = os.path.join(base_path, 'C4', 'thumbnail_aligned')
    if os.path.exists(output):
        print(f'Removing {output}')
        shutil.rmtree(output)
    os.makedirs(output, exist_ok=True)
    print(f'Writing output to {output}')
    transform = torchvision.transforms.ToTensor()
    threshold = 0.75
    polygons = defaultdict(list)
    for file in tqdm(files, disable=debug, desc="Creating masks and points"):
        section = int(file.replace(".tif", ""))
        filepath = os.path.join(input, file)
        merged_outpath = os.path.join(output, f'{file}')
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
        masks = [(prediction[0]["masks"] > threshold).squeeze().detach().cpu().numpy()]
        mask = masks[0]
        if mask.shape[0] == 0:
            continue
        dims = mask.ndim
        if dims > 2:
            mask = combine_dims(mask)        
        raw_img = np.array(img)
        mask = mask.astype(np.uint8)
        mask[mask > 0] = 255
        mask_outpath = os.path.join(output, f'mask_{file}')
        cv2.imwrite(mask_outpath, mask)

        merged_img = merge_mask(raw_img, mask)
        merged_outpath = os.path.join(output, f'merged_{file}')
        cv2.imwrite(merged_outpath, merged_img)
        

def predict_save_annotations(animal, debug=False):
    # Edit this path to the model
    modelpath = os.path.join("/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/TG/models/mask.model.pth")
    loaded_model = get_model_instance_segmentation(num_classes=2)
    workers = 2
    torch.multiprocessing.set_sharing_strategy('file_system')

    device = torch.device('cpu')
    print(f' using CPU with {workers} workers')

    if os.path.exists(modelpath):
        print(f'Loading model from {modelpath}')
        ck = torch.load(modelpath, map_location=device)
        loaded_model.load_state_dict(ck['model_state'] if 'model_state' in ck else ck)

    else:
        print(f'No model to load at {modelpath}')
        return
    base_path = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps'
    input = os.path.join(base_path, 'C1', 'thumbnail_realigned')
    if not os.path.exists(input) or len(os.listdir(input)) == 0:
        print(f'No input directory found at {input}')
        input = os.path.join(base_path, 'C1', 'thumbnail_aligned')
        print(f'Using {input}')
    files = sorted(os.listdir(input))
    if len(files) == 0:
        print(f'No files found in {input}')
        return
    transform = torchvision.transforms.ToTensor()
    threshold = 0.75
    polygons = defaultdict(list)
    for file in tqdm(files, disable=debug, desc="Creating masks and points"):
        section = int(file.replace(".tif", ""))
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
        masks = [(prediction[0]["masks"] > threshold).squeeze().detach().cpu().numpy()]
        mask = masks[0]
        if mask.shape[0] == 0:
            continue
        dims = mask.ndim
        if dims > 2:
            mask = combine_dims(mask)        
        mask = mask.astype(np.uint8)
        mask[mask > 0] = 255
        sampled_points = draw_perimeter_points(mask=mask)
        polygons[section] = sampled_points

    annotation_helper = AnnotationHelper(animal, debug=debug)
    structure = 'TG_L'
    annotation_helper.upsert_annotation(polygons, structure)



def draw_perimeter_points(
    mask,
    output_image=None,
    spacing=20,
    point_radius=3,
    point_color=(0, 0, 255),
    contour_color=(0, 255, 0),
    draw_contour=False,
):
    """
    Draw equally-spaced points around the top-most contour in a binary mask.

    Parameters
    ----------
    mask : ndarray
        Binary mask (0 background, non-zero foreground).
    output_image : ndarray or None
        Image on which to draw. If None, a BGR copy of the mask is created.
    spacing : int
        Approximate spacing between perimeter points (pixels).
    point_radius : int
        Radius of each drawn point.
    point_color : tuple
        BGR color of points.
    contour_color : tuple
        BGR color of contour (if draw_contour=True).
    draw_contour : bool
        Draw the selected contour.

    Returns
    -------
    image : ndarray
        Image with points drawn.
    points : ndarray (N,2)
        x,y coordinates of sampled perimeter points.
    contour : ndarray
        Selected contour.
    """

    # Ensure binary uint8
    binary = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )


    # Find contour with smallest centroid y (highest in image)
    top_contour = None
    min_y = np.inf

    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cy = M["m01"] / M["m00"]

        if cy < min_y:
            min_y = cy
            top_contour = c

    if top_contour is None:
        return None


    contour_pts = top_contour[:, 0, :]  # (N,2)

    # Compute cumulative arc length
    segment_lengths = np.sqrt(
        np.sum(np.diff(contour_pts, axis=0) ** 2, axis=1)
    )
    cumulative = np.concatenate(([0], np.cumsum(segment_lengths)))
    perimeter = cumulative[-1]

    if perimeter == 0:
        return contour_pts

    sample_distances = np.arange(0, perimeter, spacing)

    sampled_points = []

    for d in sample_distances:
        idx = np.searchsorted(cumulative, d)

        if idx >= len(contour_pts):
            idx = len(contour_pts) - 1

        pt = contour_pts[idx]
        sampled_points.append(pt)


    sampled_points = np.array(sampled_points)

    return sampled_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create mask from raw image")
    parser.add_argument("--animal", help="Enter the animal", required=True)
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)

    args = parser.parse_args()
    animal = args.animal
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    predict_save_annotations(animal, debug)
