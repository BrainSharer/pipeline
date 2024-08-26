import os
import math
import sys
import numpy as np
import torch
from PIL import Image
import cv2
import torchvision
from library.mask_utilities.utils import reduce_dict, collate_fn
import library.mask_utilities.transforms as T

class MaskDataset(torch.utils.data.Dataset):
    def __init__(self, root, animal=None, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root, 'normalized')))
        self.masks = sorted(os.listdir(os.path.join(root, 'thumbnail_masked')))
        if animal is not None:
            self.imgs = [img for img in self.imgs if animal in img]
            self.masks = [mask for mask in self.masks if animal in mask]

        self.img_root = os.path.join(self.root, 'normalized')
        self.mask_root = os.path.join(self.root, 'thumbnail_masked') 

    def __getitem__(self, idx):
        # load images and bounding boxes
        img_path = os.path.join(self.img_root, self.imgs[idx])
        mask_path = os.path.join(self.mask_root, self.masks[idx])
        img = Image.open(img_path).convert("L") # L = grayscale
        mask = Image.open(mask_path) # 
        mask = np.array(mask)
        mask[mask > 0] = 255
        ret, thresh = cv2.threshold(mask, 200, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for i, contour in enumerate(contours):
            x,y,w,h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area > 1500:
                xmin = int(round(x))
                ymin = int(round(y))
                xmax = int(round(x+w))
                ymax = int(round(y+h))
                #color = (i+10) * 10
                color = 255
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

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            return img, target

        return img, target

    def __len__(self):
        return len(self.imgs)


class StructureDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgdir = 'thumbnail_aligned'
        self.maskdir = 'thumbnail_masked'
        self.imgs = sorted(os.listdir(os.path.join(root, self.imgdir)))
        self.masks = sorted(os.listdir(os.path.join(root, self.maskdir)))
    def __getitem__(self, idx):
        # load images and bounding boxes
        img_path = os.path.join(self.root, self.imgdir, self.imgs[idx])
        mask_path = os.path.join(self.root, self.maskdir, self.masks[idx])
        img = Image.open(img_path) # L = grayscale, doesn't work with 16bit images
        img = np.array(img)
        if img.dtype == np.uint16:
            img = (img/256).astype('uint8')
        pimg8 = Image.fromarray(img)

        mask = Image.open(mask_path) # 
        mask = np.array(mask)
        # instances are encoded as different grayscale values (structure_id)
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
            A = abs((xmax-xmin) * (ymax-ymin)) 
            #print(f"Min area to look for {A}")
            if A < 5:
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
        #labels = torch.ones((num_objs,), dtype=torch.int64) # just one class
        # there are multiple classes/labels/structures
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64) 
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            pimg8, target = self.transforms(pimg8, target)
            return pimg8, target

        return pimg8, target

    def __len__(self):
        return len(self.imgs)



class TrigeminalDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgdir = 'thumbnail_aligned'
        self.maskdir = 'thumbnail_masked'
        self.imgs = sorted(os.listdir(os.path.join(root, self.imgdir)))
        self.masks = sorted(os.listdir(os.path.join(root, self.maskdir)))

    def __getitem__(self, idx):
        # load images and bounding boxes
        img_path = os.path.join(self.root, self.imgdir, self.imgs[idx])
        mask_path = os.path.join(self.root, self.maskdir, self.masks[idx])
        img = Image.open(img_path) # L = grayscale, doesn't work with 16bit images
        img16 = np.array(img)
        img8 = (img16/256).astype('uint8')
        pimg8 = Image.fromarray(img8)

        mask = Image.open(mask_path) # 
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        # print(num_objs)
        boxes = []
        for i in range(num_objs):
          pos = np.where(masks[i])
          xmin = np.min(pos[1])
          xmax = np.max(pos[1])
          ymin = np.min(pos[0])
          ymax = np.max(pos[0])
          # Check if area is larger than a threshold
          A = abs((xmax-xmin) * (ymax-ymin)) 
          #print(f"Min area to look for {A}")
          if A < 5:
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
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
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

        if self.transforms is not None:
            pimg8, target = self.transforms(pimg8, target)
            return pimg8, target

        return pimg8, target

    def __len__(self):
        return len(self.imgs)



def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def test_model(ROOT, animal):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    dataset = MaskDataset(ROOT, animal, transforms = get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=1,
    collate_fn=collate_fn)
    # For Training
    images,targets = next(iter(data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = model(images,targets)   # Returns losses and detections
    print('Output')
    print(output)
    # For inference
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)           # Returns predictions
    print()
    print('Predictions')
    print(predictions)
    print()

def IoU(y_real, y_pred):
  # Intersection over Union loss function
  intersection = y_real*y_pred
  #not_real = 1 - y_real
  #union = y_real + (not_real*y_pred)
  union = (y_real+y_pred)-(y_real*y_pred)
  return np.sum(intersection)/np.sum(union)

def dice_coef(y_real, y_pred, smooth=1):
  intersection = y_real*y_pred
  union = (y_real+y_pred)-(y_real*y_pred)
  return np.mean((2*intersection+smooth)/(union+smooth))

def confusion_matrix(y_true, y_pred):
  y_true= y_true.flatten()
  y_pred = y_pred.flatten()*2
  cm = y_true+y_pred
  cm = np.bincount(cm, minlength=4)
  tn, fp, fn, tp = cm
  return tp, fp, tn, fn

def get_f1_score(y_true, y_pred):
    """Return f1 score covering edge cases"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    f1_score = (2 * tp) / ((2 * tp) + fp + fn)

    return f1_score     


def train_an_epoch(model, optimizer, data_loader, device, epoch, scaler=None):
    model.train()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    loss_list = []
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        loss_list.append(loss_value)
    return np.mean(loss_list)

