from collections import defaultdict
import os
import sys
import numpy as np
import warnings
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import cv2
from PIL import Image


Image.MAX_IMAGE_PIXELS = None
from matplotlib import pyplot as plt

import pymysql
from sqlalchemy import exc

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.utilities.utilities_mask import combine_dims
from library.utilities.utilities_process import SCALING_FACTOR
from library.controller.sql_controller import SqlController
from library.controller.annotation_session_controller import AnnotationSessionController
from library.registration.brain_structure_manager import BrainStructureManager
from library.mask_utilities.mask_class import (
    MaskDataset,
    StructureDataset,
    get_transform,
)
from library.mask_utilities.utils import collate_fn
from library.mask_utilities.engine import train_one_epoch
from library.image_manipulation.filelocation_manager import FileLocationManager


"""bad structures
MD585.229.tif
MD585.253.tif
MD589.295.tif
"""


class MaskPrediction:
    def __init__(self, animal, abbreviation, epochs, debug=False):
        self.mask_root = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/"
        self.pipeline_root = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/'
        self.animal = animal
        self.abbreviation = abbreviation
        self.epochs = epochs
        self.debug = debug
        self.num_classes = 2 # 1 class (person) + background. This is different then detectron2!
        self.modelpath = os.path.join(self.mask_root, f"structures/{self.abbreviation}/mask.model.pth")
        self.annotator_id = 1

        if self.animal is not None:
            self.fileLocationManager = FileLocationManager(animal)
            self.input = self.fileLocationManager.get_thumbnail_aligned()
            self.sqlController = SqlController(self.animal)


        if self.abbreviation is not None:
            abbreviation = str(self.abbreviation)
            if abbreviation.endswith('_L') or abbreviation.endswith('_R'):
                abbreviation = abbreviation[:-2] 
            self.mask_root = os.path.join(self.mask_root, 'structures', abbreviation)
            os.makedirs(self.mask_root, exist_ok=True)
            self.modelpath = os.path.join(self.mask_root, "mask.model.pth")
            self.output = os.path.join(self.fileLocationManager.masks, 'C1', abbreviation)
            os.makedirs(self.output, exist_ok=True)

    def get_model_instance_segmentation(self):
        # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, self.num_classes
        )
        return model

    def load_machine_learning_model(self):
        """Load the CNN model used to generate image masks"""

        loaded_model = self.get_model_instance_segmentation()
        workers = 2
        torch.multiprocessing.set_sharing_strategy('file_system')

        device = torch.device('cpu')
        print(f' using CPU with {workers} workers')

        if os.path.exists(self.modelpath):
            loaded_model.load_state_dict(torch.load(self.modelpath, map_location = device))
        else:
            print('No model to load.')
            sys.exit()

        if self.debug:
            print(f'Loading model from: {self.modelpath}')
            
        return loaded_model

    def mask_trainer(self):

        if self.abbreviation is None:
            dataset = MaskDataset(self.mask_root, transforms=get_transform(train=True))
        else:
            dataset = StructureDataset(self.mask_root, transforms=get_transform(train=True))


        indices = torch.randperm(len(dataset)).tolist()

        if self.debug:
            test_cases = 12
            torch.manual_seed(1)
            dataset = torch.utils.data.Subset(dataset, indices[0:test_cases])
        else:
            dataset = torch.utils.data.Subset(dataset, indices)

        workers = 2
        batch_size = 4
        torch.multiprocessing.set_sharing_strategy("file_system")

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(
                f"Using Nvidia graphics card GPU with {workers} workers at a batch size of {batch_size}"
            )
        else:
            warnings.filterwarnings("ignore")
            device = torch.device("cpu")
            print(f"Using CPU with {workers} workers at a batch size of {batch_size}")

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            collate_fn=collate_fn,
        )

        n_files = len(dataset)
        print_freq = 10
        if n_files > 1000:
            print_freq = 100
        print(
            f"We have: {n_files} images to train and printing loss info every {print_freq} iterations."
        )
        # our dataset has two classs, tissue or 'not tissue'
        # create logging file
        logpath = os.path.join(self.mask_root, "mask.logger.txt")
        logfile = open(logpath, "w")
        logheader = f"Masking {datetime.now()} with {self.epochs} epochs\n"
        logfile.write(logheader)
        # get the model using our helper function
        model = self.get_model_instance_segmentation()
        # move model to the right device
        model.to(device)
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler which decreases the learning rate by # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.1
        )
        loss_list = []
        # original version with train_one_epoch
        for epoch in range(self.epochs):
            # train for one epoch, printing every 10 iterations
            mlogger = train_one_epoch(
                model, optimizer, data_loader, device, epoch, print_freq=print_freq
            )
            loss_txt = str(mlogger.loss)
            x = loss_txt.split()
            loss = float(x[0])
            del x
            loss_mask_txt = str(mlogger.loss_mask)
            x = loss_mask_txt.split()
            loss_mask = float(x[0])
            loss_list.append([loss, loss_mask])
            # update the learning rate
            lr_scheduler.step()
        print(f"Saving model to {self.modelpath}")
        torch.save(model.state_dict(), self.modelpath)

        logfile.write(str(loss_list))
        logfile.write("\n")
        print("Finished with masks")
        logfile.close()
        print("Creating loss chart")
        return
        fig = plt.figure()
        output_path = os.path.join(self.mask_root, "loss_plot.png")
        x = [i for i in range(len(loss_list))]
        l1 = [i[0] for i in loss_list]
        l2 = [i[1] for i in loss_list]
        plt.plot(
            x,
            l1,
            color="green",
            linestyle="dashed",
            marker="o",
            markerfacecolor="blue",
            markersize=5,
            label="Loss",
        )
        plt.plot(
            x,
            l2,
            color="red",
            linestyle=":",
            marker="o",
            markerfacecolor="yellow",
            markersize=5,
            label="Mask loss",
        )
        plt.style.use("ggplot")
        plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss over {len(x)} epochs with {len(dataset)} images")
        plt.legend()
        plt.close()
        fig.savefig(output_path, bbox_inches="tight")
        print("Finished with loss plot")

    def predict_masks(self):
        loaded_model = self.load_machine_learning_model()
        transform = torchvision.transforms.ToTensor()

        files = sorted(os.listdir(self.input))
        for file in tqdm(files[80:133], disable=self.debug):
            filepath = os.path.join(self.input, file)
            mask_dest_file = (
                os.path.splitext(file)[0] + ".tif"
            )  # colored mask images have .tif extension
            maskpath = os.path.join(self.output, mask_dest_file)
            if os.path.exists(maskpath):
                continue

            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            pimg = Image.fromarray(img)
            img_transformed = transform(pimg)
            img_transformed = img_transformed.unsqueeze(0)
            loaded_model.eval()
            with torch.no_grad():
                prediction = loaded_model(img_transformed)
            threshold = 0.5
            masks = [(prediction[0]["masks"] > threshold).squeeze().detach().cpu().numpy()]
            mask = masks[0]
            del masks
            if mask.shape[0] == 0:
                continue
            if mask.ndim == 3:
                mask = mask[0, ...]
            mask = mask.astype(np.uint8)
            mask[mask > 0] = 255

            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            if self.debug:
                print(f'{file} threshold={threshold} #contours={len(contours)}')
            cv2.drawContours(img, contours, -1, 255, 2, cv2.LINE_AA)
            cv2.imwrite(maskpath, img)

    def update_session(self):
        annotation_label = self.sqlController.get_annotation_label(self.abbreviation)
        loaded_model = self.load_machine_learning_model()
        if annotation_label is None:
            print(f'Could not find database entry for structure={self.abbreviation}')
            print('Exiting. Try again with a real structure abbreviation')
            sys.exit()
        
        annotation_session = self.sqlController.get_annotation_session(self.animal, annotation_label.id, self.annotator_id)
        annotation = {}

        transform = torchvision.transforms.ToTensor()
        files = sorted(os.listdir(self.input))
        xy_resolution = self.sqlController.scan_run.resolution
        z_resolution = self.sqlController.scan_run.zresolution
        index_points = defaultdict(list)
        index_orders = defaultdict(list)
        index_points_sorted = {}
        default_props = ["#ffff00", 1, 1, 5, 3, 1]
        m_um_scale = 1000000

        for file in tqdm(files[80:134], disable=self.debug):
            filepath = os.path.join(self.input, file)
            section = os.path.splitext(file)[0]

            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            pimg = Image.fromarray(img)
            img_transformed = transform(pimg)
            img_transformed = img_transformed.unsqueeze(0)
            loaded_model.eval()
            with torch.no_grad():
                prediction = loaded_model(img_transformed)
            masks = [(prediction[0]["masks"] > 0.5).squeeze().detach().cpu().numpy()]
            mask = masks[0]
            del masks
            if mask.shape[0] == 0:
                continue
            if mask.ndim == 3:
                mask = mask[0, ...]
            if self.debug:
                print(f'{file} mask type={type(mask)} shape={mask.shape} ndim={mask.ndim}')
            mask = mask.astype(np.uint8)
            mask[mask > 0] = 255
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            areaArray = []
            for contour in contours:
                area = cv2.contourArea(contour)
                areaArray.append(area)
            # first sort the array by area
            sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
            largest_contour = sorteddata[0][1]
            approx = cv2.approxPolyDP(largest_contour, 0.0009 * cv2.arcLength(largest_contour, True), True)
            for j in range(approx.shape[0]):
                x = approx[j][0][0] * SCALING_FACTOR * xy_resolution
                y = approx[j][0][1] * SCALING_FACTOR * xy_resolution
                z = float(section) * z_resolution
                index = int(z)
                point_order = j
                index_points[index].append([x, y, z])
                index_orders[index].append(point_order)

        for index, points in index_points.items():
            points = np.array(points)
            point_indices = np.array(index_orders[index])
            point_indices = point_indices - point_indices.min()
            sorted_points = np.array(points)[point_indices, :] / m_um_scale
            index_points_sorted[index] = sorted_points
            
        polygons = []
        for index in sorted(list(index_points_sorted.keys())):
            if index not in index_points_sorted: 
                continue
            points = index_points_sorted[index]

            lines = []
            for i in range(len(points) - 1):
                lines.append({
                    "type": "line",
                    "props": default_props,
                    "pointA": points[i].tolist(),
                    "pointB": points[i + 1].tolist(),
                })
            lines.append({
                "type": "line",
                "props": default_props,
                "pointA": points[-1].tolist(),
                "pointB": points[0].tolist(),
            })

            polygons.append({
                "type": "polygon",
                "props": default_props,
                "source": points[0].tolist(),
                "centroid": np.mean(points, axis=0).tolist(),
                "childJsons": lines
            })

        if len(polygons) > 0:
            volume = {
                "type": "volume",
                "props": default_props,
                "source": polygons[0]["source"],
                "centroid": polygons[len(polygons) // 2]["centroid"],
                "childJsons": polygons,
                "description": self.abbreviation
            }


        if self.debug:
            action = "finding"
        else:
            action = "inserting"
            annotation = volume
            annotation_session.annotation = annotation
            annotation_session.updated = datetime.now()
            self.sqlController.update_row(annotation_session)
        print(
            f"Finished {action} {len(polygons)} polygons for {self.abbreviation} of animal={self.animal} with session ID={annotation_session.id}"
        )


def create_coords(points):
    px = [a[0] for a in points]
    py = [a[1] for a in points]
    poly = [(x, y) for x, y in zip(px, py)]
    poly = [p for x in poly for p in x]
    x1 = int(np.min(px))
    y1 = int(np.min(py))
    x2 = int(np.max(px))
    y2 = int(np.max(py))
    w = x2 - x1
    h = y2 - y1
    return x1, x2, y1, y2, w, h
