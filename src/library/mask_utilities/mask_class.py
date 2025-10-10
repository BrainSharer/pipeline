import os
import sys
import numpy as np
import torch
from PIL import Image
import cv2
from typing import List, Tuple, Optional
try:
    import albumentations as A
except ImportError:
    print("Albumentations not found, please install it with 'pip install albumentations'")
    A = None

from library.image_manipulation.mask_manager import SMALL_CONTOUR_AREA
import library.mask_utilities.transforms as T

class StructureDataset(torch.utils.data.Dataset):
    def __init__(self, root, animal=None, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root, 'normalized')))
        self.masks = sorted(os.listdir(os.path.join(root, 'thumbnail_masked')))
        if animal is not None:
            self.imgs = [img for img in self.imgs if animal in img]
            self.masks = [mask for mask in self.masks if animal in mask]

        if len(self.imgs) != len(self.masks):
            print('Number of images and masks is not equal')
            sys.exit()

        if len(self.imgs) == 0:
            print('No images found')
            sys.exit()

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

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            return img, target

        return img, target

    def __len__(self):
        return len(self.imgs)


class MaskDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, animal: str, augment: Optional[A.BasicTransform] = None):
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
        if self.augment is not None and A is not None:
            augmented = self.augment(image=img, mask=mask)
            pimg8 = augmented['image']
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

        if type(pimg8) != Image.Image:
            pimg8 = Image.fromarray(pimg8)

        transforms = self.get_transform()
        pimg8, target = transforms(pimg8, target)

        return pimg8, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def get_transform():
        transforms = []
        transforms.append(T.PILToTensor())
        transforms.append(T.ConvertImageDtype(torch.float))
        return T.Compose(transforms)
    

