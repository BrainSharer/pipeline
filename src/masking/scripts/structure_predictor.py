import argparse
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

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.image_manipulation.filelocation_manager import FileLocationManager
from library.utilities.utilities_mask import combine_dims, merge_mask
from library.utilities.utilities_process import SCALING_FACTOR
from library.controller.sql_controller import SqlController
from library.controller.annotation_session_controller import AnnotationSessionController
from library.controller.structure_com_controller import StructureCOMController
from library.database_model.annotation_points import AnnotationType, PolygonSequence
from library.registration.brain_structure_manager import BrainStructureManager
from library.mask_utilities.mask_class import MaskDataset, StructureDataset, get_transform
from library.mask_utilities.utils import collate_fn
from library.mask_utilities.engine import train_one_epoch

class MaskPrediction():
    def __init__(self, animal, structures, num_classes, epochs, debug=False):
        self.animal = animal
        self.structures = structures
        self.num_classes = num_classes
        self.epochs = epochs
        self.debug = debug
        self.fileLocationManager = FileLocationManager(animal)
        self.input = os.path.join(self.fileLocationManager.prep, 'C1', 'thumbnail_aligned')
        self.output = os.path.join(self.fileLocationManager.masks, 'C1', 'structures')
        os.makedirs(self.output, exist_ok=True)
        self.modelpath = os.path.join("/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/structures/mask.model.pth" )

        if False:
            annotationSessionController = AnnotationSessionController(animal)
            structureController = StructureCOMController(animal)
            self.brainManager = BrainStructureManager(animal)
            self.sqlController = SqlController(animal)
            FK_brain_region_id = structureController.structure_abbreviation_to_id(abbreviation=self.abbreviation)
            self.annotation_session = annotationSessionController.get_annotation_session(self.animal, FK_brain_region_id, 1, AnnotationType.POLYGON_SEQUENCE)


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
            self.model.load_state_dict(torch.load(self.modelpath, map_location=torch.device("cpu")))
        else:
            print("no model to load")
            sys.exit()


    def predict_masks(self):
        self.model = self.get_model_instance_segmentation()
        self.load_machine_learning_model()
        transform = torchvision.transforms.ToTensor()

        files = sorted(os.listdir(self.input))
        for file in tqdm(files[275:300]):
            filepath = os.path.join(self.input, file)
            mask_dest_file = (os.path.splitext(file)[0] + ".tif")  # colored mask images have .tif extension
            maskpath = os.path.join(self.output, mask_dest_file)
            if os.path.exists(maskpath):
                continue
            
            img8 = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            pimg = Image.fromarray(img8)

            # Predict a single image
            #image = pimg
            #image = transform(image)
            #image = image.unsqueeze(0)
            #prediction = self.model(image)
            #prediction = prediction > 0.5
            #print('Prediction: {}'.format(prediction))
            #return

            torch_input = transform(pimg)
            torch_input = torch_input.unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                pred = self.model(torch_input)
            #masks = [(pred[0]["masks"] > 0.05).squeeze().detach().cpu().numpy()]
            masks = [pred[0]["masks"].squeeze().detach().cpu().numpy()]
            mask = masks[0]
            ids, counts = np.unique(mask, return_counts=True)
            print(f'file={file} len masks={len(masks)}')
            print(f'file={file} dtype={mask.dtype} ids={ids} counts={counts}')
            return
            dims = mask.ndim
            if dims > 2:
                mask = combine_dims(mask)
            mask = mask.astype(np.uint8)
            #mask[mask > 0] = 255
            merged_img = merge_mask(img8, mask)
            cv2.imwrite(maskpath, mask)


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
            torch_input = transform(pimg)
            torch_input = torch_input.unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                pred = self.model(torch_input)
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
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for i, c in enumerate(contours):
                    area = cv2.contourArea(c)
                    areaArray.append(area)

                #first sort the array by area
                sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
                largest_contour = sorteddata[0][1]    
                approx = cv2.approxPolyDP(largest_contour, 0.0009 * cv2.arcLength(largest_contour, True), True)
                for j in range(approx.shape[0]):
                    x = approx[j][0][0] * SCALING_FACTOR * xy_resolution
                    y = approx[j][0][1] * SCALING_FACTOR * xy_resolution
                    z = float(section) * z_resolution
                    polygon_index = z
                    point_order = j
                    polygon_sequence = PolygonSequence(x=x, y=y, z=z, source=source, 
                                                    polygon_index=polygon_index, point_order=point_order, FK_session_id=self.annotation_session.id)
                    vlist.append(polygon_sequence)
                    point_count.append(len(vlist))


                if self.debug:
                    print(f'Finished creating {len(vlist)} points on section={section}')
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
        if debug: 
            action = "finding" 
        else: 
            action = "inserting"
        print(f'Finished {action} {sum(point_count)} points for {self.abbreviation} of animal={self.animal} with session ID={self.annotation_session.id}')



    def mask_trainer(self):
        ROOT = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks'
        if self.structures:
            ROOT = os.path.join(ROOT, 'structures')
            dataset = StructureDataset(ROOT, transforms = get_transform(train=True))
        else:
            dataset = MaskDataset(ROOT, animal, transforms = get_transform(train=True))

        indices = torch.randperm(len(dataset)).tolist()

        if self.debug:
            test_cases = 12
            torch.manual_seed(1)
            dataset = torch.utils.data.Subset(dataset, indices[0:test_cases])
        else:
            dataset = torch.utils.data.Subset(dataset, indices)

        workers = 2
        batch_size = 4
        torch.multiprocessing.set_sharing_strategy('file_system')

        if torch.cuda.is_available(): 
            device = torch.device('cuda') 
            print(f'Using Nvidia graphics card GPU with {workers} workers at a batch size of {batch_size}')
        else:
            warnings.filterwarnings("ignore")
            device = torch.device('cpu')
            print(f'Using CPU with {workers} workers at a batch size of {batch_size}')

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
            collate_fn=collate_fn)

        n_files = len(dataset)
        print_freq = 10
        if n_files > 1000:
            print_freq = 100
        print(f"We have: {n_files} images to train and printing loss info every {print_freq} iterations.")
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
        optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler which decreases the learning rate by # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        loss_list = []
        
        # original version with train_one_epoch
        for epoch in range(self.epochs):
            # train for one epoch, printing every 10 iterations
            mlogger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=print_freq)
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
            if not self.debug:
                torch.save(model.state_dict(), self.modelpath)

        logfile.write(str(loss_list))
        logfile.write("\n")
        print('Finished with masks')
        logfile.close()
        print('Creating loss chart')

        fig = plt.figure()
        output_path = os.path.join(ROOT, 'loss_plot.png')
        x = [i for i in range(len(loss_list))]
        l1 = [i[0] for i in loss_list]
        l2 = [i[1] for i in loss_list]
        plt.plot(x, l1,  color='green', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=5, label="Loss")
        plt.plot(x, l2,  color='red', linestyle=':', marker='o', markerfacecolor='yellow', markersize=5, label="Mask loss")
        plt.style.use("ggplot")
        plt.xticks(np.arange(min(x), max(x)+1, 1.0))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss over {len(x)} epochs with {len(dataset)} images')
        plt.legend()
        plt.close()
        fig.savefig(output_path, bbox_inches="tight")
        print('Finished with loss plot')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)
    parser.add_argument('--debug', help='Enter true or false', required=False, default='false', type=str)
    args = parser.parse_args()
    animal = args.animal
    debug = bool({'true': True, 'false': False}[str(args.debug).lower()])
    mask_predictor = MaskPrediction(animal, debug)
    #mask_predictor.get_insert_mask_points()
    mask_predictor.predict_masks()
