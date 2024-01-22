import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
import pymysql
from sqlalchemy import exc


Image.MAX_IMAGE_PIXELS = None
import cv2
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

from pathlib import Path
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

class MaskPrediction():
    def __init__(self, animal, abbreviation, debug):
        self.animal = animal
        self.abbreviation = abbreviation
        self.num_classes = 2
        self.debug = debug
        self.model = self.get_model_instance_segmentation(self.num_classes)
        self.modelpath = os.path.join("/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/tg/mask.model.pth" )
        self.load_machine_learning_model()
        fileLocationManager = FileLocationManager(animal)
        self.input = os.path.join(fileLocationManager.prep, 'C1', 'thumbnail_aligned')
        self.output = os.path.join(fileLocationManager.masks, 'C1', 'tg')
        os.makedirs(self.output, exist_ok=True)
        annotationSessionController = AnnotationSessionController(animal)
        structureController = StructureCOMController(animal)
        self.brainManager = BrainStructureManager(animal)
        self.sqlController = SqlController(animal)

        FK_brain_region_id = structureController.structure_abbreviation_to_id(abbreviation=self.abbreviation)
        self.annotation_session = annotationSessionController.get_annotation_session(self.animal, FK_brain_region_id, 1, AnnotationType.POLYGON_SEQUENCE)


    def get_model_instance_segmentation(self, num_classes):
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


    def load_machine_learning_model(self):
        """Load the CNN model used to generate image masks"""
        if os.path.exists(self.modelpath):
            self.model.load_state_dict(torch.load(self.modelpath, map_location=torch.device("cpu")))
        else:
            print("no model to load")
            sys.exit()


    def predict_mask(self):
        transform = torchvision.transforms.ToTensor()

        files = sorted(os.listdir(self.input))
        for file in tqdm(files):
            filepath = os.path.join(self.input, file)
            mask_dest_file = (os.path.splitext(file)[0] + ".tif")  # colored mask images have .tif extension
            maskpath = os.path.join(self.output, mask_dest_file)
            if os.path.exists(maskpath):
                continue
            
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

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)
    parser.add_argument("--abbreviation", help="Enter the brain region abbreviation", required=True, type=str)
    parser.add_argument('--debug', help='Enter true or false', required=False, default='false', type=str)
    args = parser.parse_args()
    animal = args.animal
    abbreviation = args.abbreviation
    debug = bool({'true': True, 'false': False}[str(args.debug).lower()])
    mask_predictor = MaskPrediction(animal, abbreviation, debug)
    mask_predictor.get_insert_mask_points()
    #mask_predictor.predict_mask()