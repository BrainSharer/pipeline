import os
import sys
import numpy as np
import warnings
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import shutil
import json

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

from library.image_manipulation.filelocation_manager import FileLocationManager
from library.utilities.utilities_mask import combine_dims, merge_mask
from library.utilities.utilities_process import SCALING_FACTOR
from library.controller.sql_controller import SqlController
from library.controller.annotation_session_controller import AnnotationSessionController
from library.controller.polygon_sequence_controller import PolygonSequenceController
from library.controller.structure_com_controller import StructureCOMController
from library.database_model.annotation_points import AnnotationType, PolygonSequence
from library.registration.brain_structure_manager import BrainStructureManager
from library.mask_utilities.mask_class import (
    MaskDataset,
    StructureDataset,
    get_transform,
)
from library.mask_utilities.utils import collate_fn
from library.mask_utilities.engine import train_one_epoch

ROOT = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/"


class MaskPrediction:
    def __init__(self, animal, structures, num_classes, epochs, debug=False):
        self.animal = animal
        self.structures = structures
        self.num_classes = num_classes
        self.epochs = epochs
        self.debug = debug
        self.fileLocationManager = FileLocationManager(animal)
        self.input = os.path.join(
            self.fileLocationManager.prep, "C1", "thumbnail_aligned"
        )
        self.output = os.path.join(self.fileLocationManager.masks, "C1", "structures")
        os.makedirs(self.output, exist_ok=True)
        self.modelpath = os.path.join(
            "/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/structures/mask.model.pth"
        )
        OUTPUT = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/structures/detectron"
        self.training_path = os.path.join(OUTPUT, "train")
        self.validation = os.path.join(OUTPUT, "validation")
        os.makedirs(self.validation, exist_ok=True)
        self.structure_ids = {0: 33, 1: 21}
        self.annotator_id = 1
        self.setup_training_directory()
        self.training_files = sorted(os.listdir(self.training_path))
        self.image_ids = {k: v for v, k in enumerate(self.training_files)}

        if False:
            annotationSessionController = AnnotationSessionController(animal)
            structureController = StructureCOMController(animal)
            self.brainManager = BrainStructureManager(animal)
            self.sqlController = SqlController(animal)
            FK_brain_region_id = structureController.structure_abbreviation_to_id(
                abbreviation=self.abbreviation
            )
            self.annotation_session = (
                annotationSessionController.get_annotation_session(
                    self.animal, FK_brain_region_id, 1, AnnotationType.POLYGON_SEQUENCE
                )
            )

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
        if os.path.exists(self.modelpath):
            self.model.load_state_dict(
                torch.load(self.modelpath, map_location=torch.device("cpu"))
            )
        else:
            print("no model to load")
            sys.exit()

    def mask_trainer(self):
        if self.structures:
            ROOT = os.path.join(ROOT, "structures")
            dataset = StructureDataset(ROOT, transforms=get_transform(train=True))
            print(dataset[1])
        else:
            dataset = MaskDataset(
                ROOT, self.animal, transforms=get_transform(train=True)
            )

        indices = torch.randperm(len(dataset)).tolist()

        if self.debug:
            test_cases = 50
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
        logpath = os.path.join(ROOT, "mask.logger.txt")
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
        return
        torch.save(model.state_dict(), self.modelpath)

        logfile.write(str(loss_list))
        logfile.write("\n")
        print("Finished with masks")
        logfile.close()
        print("Creating loss chart")

        fig = plt.figure()
        output_path = os.path.join(ROOT, "loss_plot.png")
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
        self.model = self.get_model_instance_segmentation()
        self.load_machine_learning_model()
        transform = torchvision.transforms.ToTensor()

        files = sorted(os.listdir(self.input))
        for file in tqdm(files):
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
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(img_transformed)

            """
            masks = [(pred[0]["masks"] > 0.5).squeeze().detach().cpu().numpy()]
            #masks = [pred[0]["masks"].squeeze().detach().cpu().numpy()]
            mask = masks[0]
            #ids, counts = np.unique(mask, return_counts=True)
            #print(f'file={file} len masks={len(masks)}')
            #print(f'file={file} dtype={mask.dtype} ids={ids} counts={counts}')
            #return
            dims = mask.ndim
            if dims > 2:
                mask = combine_dims(mask)
            mask = mask.astype(np.uint8)
            mask[mask > 0] = 255
            merged_img = merge_mask(img, mask)
            cv2.imwrite(maskpath, merged_img)
            """
            print(prediction[0]["labels"])

            for i in range(len(prediction[0]["masks"])):
                # iterate over masks
                mask = prediction[0]["masks"][i, 0] > 0.9
                mask = mask.mul(255).byte().cpu().numpy()
                contours, _ = cv2.findContours(
                    mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
                )
                cv2.drawContours(img, contours, -1, 255, 2, cv2.LINE_AA)

            cv2.imwrite(maskpath, img)

    def setup_training_directory(self):
        """Go through the training files, create coco json datasets"""
        animals = ["MD585", "MD589", "MD594"]
        for animal in animals:
            sqlController = SqlController(animal)
            polygon = PolygonSequenceController(animal=animal)

            for _, structure_id in self.structure_ids.items():
                df = polygon.get_volume(animal, self.annotator_id, structure_id)
                z_scale = sqlController.scan_run.zresolution
                sections = []

                for _, row in df.iterrows():
                    z = row["coordinate"][2]
                    section = int(np.round(z / z_scale))
                    sections.append(section)
                    file = str(section).zfill(3) + ".tif"
                    filename = f"{animal}.{file}"
                    inpath = os.path.join(self.input, file)
                    img_outpath = os.path.join(self.training_path, filename)
                    if not os.path.exists(img_outpath):
                        shutil.copyfile(
                            inpath, img_outpath
                        )  # only needs to be done once

    def setup_training(self):
        """Go through the training files, create coco json datasets"""
        id = 0
        animals = ["MD585", "MD589", "MD594"]
        annotations = []
        for animal in animals:
            sqlController = SqlController(animal)
            polygon = PolygonSequenceController(animal=animal)

            for category_id, structure_id in self.structure_ids.items():
                df = polygon.get_volume(animal, self.annotator_id, structure_id)
                scale_xy = sqlController.scan_run.resolution
                z_scale = sqlController.scan_run.zresolution
                polygons = defaultdict(list)

                for _, row in df.iterrows():
                    x = row["coordinate"][0]
                    y = row["coordinate"][1]
                    z = row["coordinate"][2]
                    xy = (x / scale_xy / SCALING_FACTOR, y / scale_xy / SCALING_FACTOR)
                    section = int(np.round(z / z_scale))
                    polygons[section].append(xy)

                for section, points in tqdm(polygons.items()):
                    file = str(section).zfill(3) + ".tif"
                    filename = f"{animal}.{file}"
                    anno_dict = self.construct_annotations(
                        id=id, image_id=self.image_ids[filename],
                        category_id=category_id,
                        anno=points,
                    )
                    annotations.append(anno_dict)
                    id += 1

                    if False:
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

                        img = cv2.imread(inpath, cv2.IMREAD_GRAYSCALE)
                        cv2.rectangle(img, (x1, y1), (x2, y2), 255, 2)
                        points = np.array(points).astype(np.int32)
                        cv2.fillPoly(img, pts=[points], color=255)
                        val_outpath = os.path.join(self.validation, filename)
                        cv2.imwrite(val_outpath, img)

        self.add_images_to_coco(annotations, training=True)

    def get_insert_mask_points(self):
        transform = torchvision.transforms.ToTensor()
        source = "NA"
        files = sorted(os.listdir(self.input))
        xy_resolution = self.sqlController.scan_run.resolution
        z_resolution = self.sqlController.scan_run.zresolution
        for i, file in enumerate(files):
            vlist = []
            filepath = os.path.join(self.input, file)
            section = os.path.splitext(file)[0]
            img8 = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            pimg = Image.fromarray(img8)
            img_transformed = transform(pimg)
            img_transformed = img_transformed.unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                pred = self.model(img_transformed)
            masks = [(pred[0]["masks"] > 0.5).squeeze().detach().cpu().numpy()]
            mask = masks[0]
            dims = mask.ndim
            if dims > 2:
                mask = combine_dims(mask)
            mask = mask.astype(np.uint8)
            mask[mask > 0] = 255
            ids, counts = np.unique(mask, return_counts=True)
            areaArray = []
            point_count = []
            if len(ids) > 1:
                _, thresh = cv2.threshold(mask, 254, 255, 0)
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                for i, c in enumerate(contours):
                    area = cv2.contourArea(c)
                    areaArray.append(area)

                # first sort the array by area
                sorteddata = sorted(
                    zip(areaArray, contours), key=lambda x: x[0], reverse=True
                )
                largest_contour = sorteddata[0][1]
                approx = cv2.approxPolyDP(
                    largest_contour, 0.0009 * cv2.arcLength(largest_contour, True), True
                )
                for j in range(approx.shape[0]):
                    x = approx[j][0][0] * SCALING_FACTOR * xy_resolution
                    y = approx[j][0][1] * SCALING_FACTOR * xy_resolution
                    z = float(section) * z_resolution
                    polygon_index = z
                    point_order = j
                    polygon_sequence = PolygonSequence(
                        x=x,
                        y=y,
                        z=z,
                        source=source,
                        polygon_index=polygon_index,
                        point_order=point_order,
                        FK_session_id=self.annotation_session.id,
                    )
                    vlist.append(polygon_sequence)
                    point_count.append(len(vlist))

                if self.debug:
                    print(f"Finished creating {len(vlist)} points on section={section}")
                else:
                    try:
                        self.brainManager.sqlController.session.bulk_save_objects(vlist)
                        self.brainManager.sqlController.session.commit()
                    except pymysql.err.IntegrityError as e:
                        self.brainManager.sqlController.session.rollback()
                    except exc.IntegrityError as e:
                        self.brainManager.sqlController.session.rollback()
                    except Exception as e:
                        self.brainManager.sqlController.session.rollback()
        if self.debug:
            action = "finding"
        else:
            action = "inserting"
        print(
            f"Finished {action} {sum(point_count)} points for {self.abbreviation} of animal={self.animal} with session ID={self.annotation_session.id}"
        )

    @staticmethod
    def dict_construct(filename, points):
        x, y = zip(*points)
        new_dic = {
            "fileref": "",
            "size": 12345,
            "filename": filename,
            "base64_img_data": "",
            "file_attributes": {},
            "regions": {
                "0": {
                    "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": x,
                        "all_points_y": y,
                    },
                    "region_attributes": {},
                }
            },
        }
        return new_dic

    @staticmethod
    def construct_annotations(id, image_id, category_id, anno):
        """Each dictionary contains a list of every individual object annotation 
        from every image in the dataset. For example, if a brain has 64 SC polygons spread out across 100 images, there will be 64 SC
        annotations (along with a ton of annotations for other object categories). Often there will be multiple structures on a section. 
        This results in a new annotation item for each one.
        Area is measured in pixels (e.g. a 10px by 20px box would have an area of 200).
        Is Crowd specifies whether the segmentation is for a single object or for a group/cluster of objects.
        The image id corresponds to a specific image in the dataset.
        The COCO bounding box format is [top left x position, top left y position, width, height].
        The category id corresponds to a single category specified in the categories section.
        Each annotation also has an id (unique to all other annotations in the dataset).
        """
        px = [a[0] for a in anno]
        py = [a[1] for a in anno]
        poly = [(x, y) for x, y in zip(px, py)]
        poly = [p for x in poly for p in x]
        x1 = np.min(px)
        y1 = np.min(py)
        x2 = np.max(px)
        y2 = np.max(py)
        width = x2 - x1
        height = y2 - y1
        area = width * height
        new_dic = {
            "id": id,
            "category_id": category_id,
            "iscrowd": 0,
            "segmentation": [poly],
            "image_id": image_id,
            "area": area,
            "bbox": [x1, y1, width, height],
        }
        return new_dic

    def add_images_to_coco(self, annotations, training=True):
        if training:
            coco_filename = f"/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/structures/detectron/structure_training.json"
        else:
            coco_filename = f"/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/structures/detectron/structure_testing.json"

        coco = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 0, "name": "SC", "supercategory": "SC"},
                {"id": 1, "name": "IC", "supercategory": "IC"},
            ],
        }

        # images info
        images = []
        for file in self.training_files:
            filename = os.path.join(self.training_path, file)
            im = Image.open(filename)
            width, height = im.size
            image_details = {
                "id": self.image_ids[file],
                "height": height,
                "width": width,
                "file_name": file,
            }
            im.close()
            images.append(image_details)
        coco["images"] = images
        if training:
            coco["annotations"] = annotations

        with open(coco_filename, "w") as coco_file:
            json.dump(coco, coco_file, indent=4)
