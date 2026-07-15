from collections import defaultdict
import shutil
import cv2
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path
from PIL import Image

from library.utilities.utilities_mask import merge_mask
Image.MAX_IMAGE_PIXELS = None
import sys
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm
import warnings

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.utilities.utilities_process import M_UM_SCALE, SCALING_FACTOR
from library.controller.sql_controller import SqlController
from library.mask_utilities.utils import collate_fn
from library.mask_utilities.engine import train_one_epoch
from library.image_manipulation.filelocation_manager import FileLocationManager


from library.image_manipulation.mask_manager import SMALL_CONTOUR_AREA
import library.mask_utilities.transforms as T

class MaskPrediction:
    def __init__(self, animal, abbreviation, epochs, annotator_id=1, debug=False):
        self.mask_root = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/"
        self.pipeline_root = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/'
        self.animal = animal
        self.abbreviation = abbreviation
        self.epochs = epochs
        self.debug = debug
        self.num_classes = 2 # 1 class (person) + background. This is different then detectron2!
        self.modelname = "mask.model.pth"
        self.modelpath = os.path.join(self.mask_root, "brain", "models")
        self.annotator_id = annotator_id

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
            self.modelpath = os.path.join(self.mask_root, "models")
            self.fileLocationManager = FileLocationManager('MD585') # hard code an animal
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

        modelpath = os.path.join(self.modelpath, self.modelname)
        if os.path.exists(self.modelpath):
            loaded_model.load_state_dict(torch.load(modelpath, map_location = device))
        else:
            print('No model to load.')
            sys.exit()

        if self.debug:
            print(f'Loading model from: {modelpath}')
            
        return loaded_model

    def mask_trainer(self):

        if self.abbreviation is None:
            dataset = MaskDataset(self.mask_root, transforms=get_transform(), debug=self.debug)
        else:
            dataset = StructureDataset(self.mask_root, transforms=get_transform(), debug=self.debug)


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
        if self.debug:
            print('DEBUG mode, not saving model.')
            print(f'Wrote logs to {logpath}')
        else:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            modelname = f"mask.model_{timestamp}.pth"            
            modelpath = os.path.join(self.modelpath, modelname)
            print(f"Saving model to {modelpath}")
            torch.save(model.state_dict(), modelpath)
        

        logfile.write(str(loss_list))
        logfile.write("\n")
        print("Finished with masks")
        logfile.close()
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
        print(f"Wrote loss plot to {output_path}")

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

    def masks_from_database(self):
        if self.animal is None:
            print("Animal is required")
            exit(0)

        xy_resolution = self.sqlController.scan_run.resolution
        z_resolution = self.sqlController.scan_run.zresolution
        labels = self.sqlController.get_labels(['TG_L', 'TG_R'])
        label_ids = [l.id for l in labels]

        #get_annotation_session(self, prep_id: str, label_ids: list, annotator_id: int, debug: bool = False
        annotation_session = self.sqlController.get_annotation_session(self.animal, label_ids, self.annotator_id)
        if annotation_session is not None:
            print(f'Loaded session with ID={annotation_session.id}')
            annotation = annotation_session.annotation
        else:
            print(f'Could not fine session with {self.animal=}, {self.abbreviation=}, and {self.annotator_id=}')
            exit(0)

        try:
            data = annotation_session.annotation["childJsons"]
        except KeyError:
            print("No childJsons key in data")
            exit(0)

        polygons = defaultdict(list)

        for row in data:
            if 'childJsons' in row:           
                for child in row['childJsons']:
                #points = row['childJsons']
                #for i in range(len(points) - 1):    
                    #x,y,z = points[i]['pointA']
                    if 'type' in child and child['type'] == 'line':
                        x,y,z = child['pointA']
                        x = x * M_UM_SCALE/xy_resolution/SCALING_FACTOR
                        y = y * M_UM_SCALE/xy_resolution/SCALING_FACTOR
                        section = int(np.round((z*M_UM_SCALE/z_resolution) - 0.5))
                        polygons[section].append((x,y))
            
        color = 200

        base_path = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{self.animal}/preps'
        inputpath = os.path.join(base_path, 'C1', 'thumbnail_realigned')
        if not os.path.exists(inputpath) or len(os.listdir(inputpath)) == 0:
            print(f'No input directory found at {inputpath}')
            inputpath = os.path.join(base_path, 'C1', 'thumbnail_aligned')
        
        print(f'Using {inputpath}')

        image_outpath = os.path.join(self.mask_root, 'images')
        mask_outpath = os.path.join(self.mask_root, 'masks')
        merged_outpath = os.path.join(self.mask_root, 'merged')
        os.makedirs(image_outpath, exist_ok=True)
        os.makedirs(mask_outpath, exist_ok=True)
        os.makedirs(merged_outpath, exist_ok=True)
        
        for section, points in tqdm(polygons.items(), desc="Creating masks,merged and copying original"):
            file = str(section).zfill(3) + ".tif"
            inpath = os.path.join(inputpath, file)
            filename = f"{self.animal}.{file}"
            img_file_outpath = os.path.join(image_outpath, filename)
            mask_file_outpath = os.path.join(mask_outpath, filename)
            merged_file_outpath = os.path.join(merged_outpath, filename)
            img = cv2.imread(inpath, cv2.IMREAD_GRAYSCALE)
            #1. Mask file
            mask = np.zeros((img.shape), dtype=np.uint8)
            points = np.array(points).astype(np.int32)
            #cv2.polylines(mask, pts=[points], color=255, isClosed=True, thickness=2, lineType=cv2.LINE_8)
            #contour = cv2.approxPolyDP(points, 0.5, True)
            # Ensure integer coordinates
            #contour = np.round(contour).astype(np.int32) 
            cv2.fillPoly(mask, [points], 255)

            # Fill any remaining holes
            #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
            #                        np.ones((5,5), np.uint8))                       

            cv2.imwrite(mask_file_outpath, mask)
            #2. original image
            if not os.path.exists(img_file_outpath):
                shutil.copyfile(inpath, img_file_outpath) # only needs to be done once
            #3. merged file for sanity check
            merged_image = merge_mask(img, mask)
            cv2.imwrite(merged_file_outpath, merged_image)


        



##### Dataset classes
class MaskDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, animal: str, augment = None, debug=False):
        self.root = root
        self.animal = animal
        self.img_root = os.path.join(self.root, 'images')
        self.mask_root = os.path.join(self.root, 'masks')
        if not os.path.exists(self.img_root):
            print(f'No image directory found at {self.img_root}')
            sys.exit()
        if not os.path.exists(self.mask_root):
            print(f'No mask directory found at {self.mask_root}')
            sys.exit()
        self.imgs = sorted(os.listdir(self.img_root))
        self.masks = sorted(os.listdir(self.mask_root))
        self.augment = augment
        self.debug = debug
                            
    def __getitem__(self, idx):
        # load images and bounding boxes
        img_path = os.path.join(self.img_root, self.imgs[idx])
        mask_path = os.path.join(self.mask_root, self.masks[idx])
        img = Image.open(img_path) # L = grayscale, doesn't work with 16bit images grayscale_img = img.convert('L')
        if img.mode == 'I;16':
            img = img.convert('L')
        img = np.array(img)
        #if img.dtype == np.uint16:
        #    img = (img/256).astype('uint8')
        #pimg8 = Image.fromarray(img)
        mask = Image.open(mask_path) # 
        mask = np.array(mask)
        if self.augment is not None and A is not None:
            augmented = self.augment(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)

        boxes = []
        labels = []
        for i in range(num_objs):
            labels.append(i)
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            # Check if area is larger than a threshold
            check_area = abs((xmax-xmin) * (ymax-ymin)) 
            #print(f"Min area to look for {A}")
            if check_area < 5:
                print('Nr before deletion:', num_objs)
                obj_ids=np.delete(obj_ids, [i])
                # print('Area smaller than 5! Box coordinates:', [xmin, ymin, xmax, ymax])
                print('Nr after deletion:', len(obj_ids))
                continue

            boxes.append([xmin, ymin, xmax, ymax])

        #print('nr boxes is equal to nr ids:', len(boxes)==len(obj_ids))
        num_objs = len(obj_ids)
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64) # just one class
        # there are multiple classes/labels/structures
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if type(img) != Image.Image:
            img = Image.fromarray(img)

        transforms = get_transform()
        img, target = transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class StructureDataset(torch.utils.data.Dataset):
    """TODO, this one needs lots of work"""
    
    def __init__(self, root, animal=None, transforms=None, debug=False):
        self.root = root
        self.transforms = transforms
        self.img_root = os.path.join(root, 'images')
        self.mask_root = os.path.join(root, 'masks')
        self.imgs = sorted(os.listdir(self.img_root))
        self.masks = sorted(os.listdir(self.mask_root))
        self.debug = debug

        if animal is not None:
            self.imgs = [img for img in self.imgs if animal in img]
            self.masks = [mask for mask in self.masks if animal in mask]

        if len(self.imgs) != len(self.masks):
            print('Number of images and masks is not equal')
            sys.exit()

        if len(self.imgs) == 0:
            print('No images found')
            sys.exit()

        if self.debug:
            print(f'Root dir is {root}')
            print(f'Image dir is {self.img_root}')
            print(f'Mask dir is {self.mask_root}')



    def __getitem__(self, idx):
        # load images and bounding boxes
        img_path = os.path.join(self.img_root, self.imgs[idx])
        mask_path = os.path.join(self.mask_root, self.masks[idx])
        img = Image.open(img_path) # L = grayscale, doesn't work with 16bit images
        img = np.array(img)
        if img.dtype == np.uint16:
            img = (img/256).astype('uint8')
        pimg8 = Image.fromarray(img)
        mask = Image.open(mask_path) # 
        mask = np.array(mask)

        ret, thresh = cv2.threshold(mask, 200, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for i, contour in enumerate(contours):
            x,y,w,h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area > SMALL_CONTOUR_AREA:
                xmin = int(round(x))
                ymin = int(round(y))
                xmax = int(round(x+w))
                ymax = int(round(y+h))
                color = (i+10) * 10
                cv2.fillPoly(mask, [contour], color);
                #print(f'Area: {area}, Box: {xmin, ymin, xmax, ymax}')
                boxes.append([xmin, ymin, xmax, ymax])
        
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        try:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        except Exception as e:
            print(f'Error: {e} boxes has shape {boxes.shape}')
            area = torch.zeros((1, 4), dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["masks"] = masks

        if type(pimg8) != Image.Image:
            pimg8 = Image.fromarray(pimg8)

        transforms = get_transform()
        pimg8, target = transforms(pimg8, target)

        return pimg8, target

    def __len__(self):
        return len(self.imgs)
    

def get_transform():
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    return T.Compose(transforms)
    

