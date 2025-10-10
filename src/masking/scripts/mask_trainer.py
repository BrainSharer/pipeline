import argparse
from datetime import datetime
import os
import sys
from pathlib import Path
import torch
import torch.utils.data
import torch.multiprocessing
import numpy as np
from matplotlib import pyplot as plt
import warnings
try:
    import albumentations as A
except ImportError:
    print("Albumentations not found, please install it with 'pip install albumentations'")
    A = None

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.mask_utilities.mask_class import MaskDataset, StructureDataset
from library.image_manipulation.mask_manager import MaskManager
from library.mask_utilities.utils import collate_fn
from library.mask_utilities.engine import train_one_epoch

class MaskTrainer():

    def __init__(self, animal, structure, epochs, num_classes, debug=False):
        data_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks'
        self.workers = 2
        self.batch_size = 4
        self.animal = animal
        self.structure = structure
        self.epochs = epochs
        self.num_classes = num_classes
        self.debug = debug
        if torch.cuda.is_available(): 
            self.device = torch.device('cuda') 
            print('Using Nvidia graphics card GPU.')
        else:
            warnings.filterwarnings("ignore")
            self.device = torch.device('cpu')
            print('No Nvidia card found, using CPU.')
        self.created = datetime.now().strftime("%Y-%m-%d-%H:%M")
        self.root = os.path.join(data_path, self.structure)
        if os.path.exists(self.root):
            print(f"Using root directory: {self.root}")
        else:
            raise FileNotFoundError(f"Root directory not found: {self.root}")
        
        # Augmentations
        augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.8, 1.2), translate_percent=0.1, rotate=(-15, 15), shear=(-10, 10), p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ])

        if self.structure == 'structures':
            self.dataset = StructureDataset(self.root)
        else:
            self.dataset = MaskDataset(self.root, animal, augment=augmentations if A is not None else None)


    def train(self):

        indices = torch.randperm(len(self.dataset)).tolist()

        if self.debug:
            test_cases = 12
            torch.manual_seed(1)
            torch_dataset = torch.utils.data.Subset(self.dataset, indices[0:test_cases])
        else:
            torch_dataset = torch.utils.data.Subset(self.dataset, indices)

        ## the line below is very important for data on an NFS file system!
        torch.multiprocessing.set_sharing_strategy('file_system')


        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            torch_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers,
            collate_fn=collate_fn)

        n_files = len(torch_dataset)
        print_freq = 100
        if n_files > 1000:
            print_freq = 1000
        print(f"We have: {n_files} images to train from {self.dataset.img_root} and printing loss info every {print_freq} iterations.")
        # our dataset has two classs, tissue or 'not tissue'
        model_train_name = 'mask.model.train.pth'
        model_final_name = 'mask.model.pth'
        model_dir = os.path.join(self.root, 'models')
        model_train_path = os.path.join(model_dir, model_train_name)
        model_final_path = os.path.join(model_dir, model_final_name)
        # get the model using our helper function
        mask_manager = MaskManager()
        model = mask_manager.get_model_instance_segmentation(self.num_classes)

        # load model dictionary if it exists
        if os.path.exists(model_train_path):
            print(f"Loading model dictionary weights from {model_train_path}")
            model.load_state_dict(torch.load(model_train_path, map_location = self.device, weights_only=True))
        else:
            print(f"Model training model not found at {model_train_path}")

        # move model to the right device
        model.to(self.device)
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler which decreases the learning rate by # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        loss_list = []
        
        best_val = float('inf')
        for epoch in range(epochs):
            # train for one epoch, printing every 10 iterations
            mlogger = train_one_epoch(model, optimizer, data_loader, self.device, epoch, print_freq=print_freq)
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

            model.eval()
            if loss < best_val:
                best_val = loss
                torch.save(model.state_dict(), model_final_path)
                print(f"Saved new best model (val_loss={best_val:.4f} at epoch {epoch}) to {model_final_path}")

        print('Creating loss chart')
        fig = plt.figure()
        output_path = os.path.join(str(self.root), f'loss_plot.{self.created}.png')
        x = [i for i in range(len(loss_list))]
        l1 = [i[0] for i in loss_list]
        l2 = [i[1] for i in loss_list]
        plt.plot(x, l1,  color='green', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=5, label="Loss")
        plt.plot(x, l2,  color='red', linestyle=':', marker='o', markerfacecolor='yellow', markersize=5, label="Mask loss")
        plt.style.use("ggplot")
        plt.xticks(np.arange(min(x), max(x)+1, 1.0))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss over {len(x)} epochs with {len(self.dataset)} images')
        plt.legend()
        plt.close()
        fig.savefig(output_path, bbox_inches="tight")
        print(f'Saving loss plot to {output_path}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on masking')
    parser.add_argument('--animal', help='specify animal', required=False, type=str)
    parser.add_argument('--debug', help='test model', required=False, default='false', type=str)
    parser.add_argument('--structure', help='Use TG or structure masking', required=False, default='brain', choices=['brain', 'TG', 'structures'], type=str)
    parser.add_argument('--epochs', help='# of epochs', required=False, default=2, type=int)
    parser.add_argument('--num_classes', help='# of structures', required=False, default=2, type=int)
    
    args = parser.parse_args()
    structure = args.structure.strip()
    debug = bool({'true': True, 'false': False}[args.debug.lower()])
    animal = args.animal
    epochs = args.epochs
    num_classes = args.num_classes
    mask_trainer = MaskTrainer(animal, structure, epochs, num_classes, debug)
    mask_trainer.train()




