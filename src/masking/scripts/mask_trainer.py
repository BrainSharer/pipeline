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

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.mask_utilities.mask_class import MaskDataset, StructureDataset, get_transform
from library.image_manipulation.mask_manager import MaskManager
from library.mask_utilities.utils import collate_fn
from library.mask_utilities.engine import train_one_epoch, evaluate

class MaskTrainer():

    def __init__(self, animal, structures, epochs, num_classes, debug):
        self.root = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks'
        self.workers = 2
        self.batch_size = 4
        self.animal = animal
        self.structures = structures
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


    def train_and_test(self):
        # our dataset has two classes only - background and person
        # use our dataset and defined transformations
        dataset = MaskDataset(self.root, animal=None, transforms = get_transform(train=True))
        dataset_test = MaskDataset(self.root, animal=None, transforms = get_transform(train=False))

        # split the dataset in train and test set
        indices = torch.randperm(len(dataset)).tolist()
        if self.debug:
            test_cases = 20
            torch.manual_seed(1)
            dataset = torch.utils.data.Subset(dataset, indices[0:test_cases])
            dataset_test = torch.utils.data.Subset(dataset_test, indices[test_cases:test_cases+2])
        else:            
            dataset = torch.utils.data.Subset(dataset, indices[:-50])
            dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])


        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            collate_fn=collate_fn
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn
        )

        # get the model using our helper function
        mask_manager = MaskManager()
        model = mask_manager.get_model_instance_segmentation(num_classes)
        # move model to the right device
        model.to(self.device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler which decreases the learning rate by # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        n_files = len(dataset)
        print_freq = 10
        if n_files > 1000:
            print_freq = 100

        print(f"We have: {len(dataset)} images to train and {len(dataset_test)} images to test over {self.epochs} epochs.")

        for epoch in range(epochs):
            train_one_epoch(model, optimizer, data_loader, self.device, epoch, print_freq=print_freq)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, data_loader_test, device=self.device)

        print("That's it!")    

    def train(self):

        if self.structures:
            structure_root = os.path.join(self.root, 'structures')
            dataset = StructureDataset(structure_root, transforms = get_transform(train=True))
        else:
            dataset = MaskDataset(self.root, animal, transforms = get_transform(train=True))

        indices = torch.randperm(len(dataset)).tolist()

        if self.debug:
            test_cases = 12
            torch.manual_seed(1)
            torch_dataset = torch.utils.data.Subset(dataset, indices[0:test_cases])
        else:
            torch_dataset = torch.utils.data.Subset(dataset, indices)

        ## the line below is very important for data on an NFS file system!
        torch.multiprocessing.set_sharing_strategy('file_system')


        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            torch_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers,
            collate_fn=collate_fn)

        n_files = len(torch_dataset)
        print_freq = 10
        if n_files > 1000:
            print_freq = 100
        print(f"We have: {n_files} images to train from {dataset.img_root} and printing loss info every {print_freq} iterations.")
        # our dataset has two classs, tissue or 'not tissue'
        modelpath = os.path.join(self.root, 'mask.model.pth')
        # create logging file
        logpath = os.path.join(self.root, "mask.logger.txt")
        logfile = open(logpath, "w")
        logheader = f"Masking {datetime.now()} with {epochs} epochs from {dataset.img_root} with {n_files} files.\n"
        logfile.write(logheader)
        # get the model using our helper function
        mask_manager = MaskManager()
        model = mask_manager.get_model_instance_segmentation(num_classes)
        # move model to the right device
        model.to(self.device)
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler which decreases the learning rate by # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        loss_list = []
        
        # original version with train_one_epoch
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
            if not self.debug:
                torch.save(model.state_dict(), modelpath)


        logfile.write(str(loss_list))
        logfile.write("\n")
        print('Finished with masks')
        logfile.close()
        print('Creating loss chart')

        fig = plt.figure()
        output_path = os.path.join(self.root, 'loss_plot.png')
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='specify animal', required=False, type=str)
    parser.add_argument('--debug', help='test model', required=False, default='false', type=str)
    parser.add_argument('--test', help='train or test model', required=False, default='false', type=str)
    parser.add_argument('--structures', help='Use TG or structure masking', required=False, default='false', type=str)
    parser.add_argument('--epochs', help='# of epochs', required=False, default=2, type=int)
    parser.add_argument('--num_classes', help='# of structures', required=False, default=2, type=int)
    
    args = parser.parse_args()
    structures = bool({'true': True, 'false': False}[args.structures.lower()])
    debug = bool({'true': True, 'false': False}[args.debug.lower()])
    test = bool({'true': True, 'false': False}[args.test.lower()])
    animal = args.animal
    epochs = args.epochs
    num_classes = args.num_classes
    mask_trainer = MaskTrainer(animal, structures, epochs, num_classes, debug)
    if test:
        mask_trainer.train_and_test()
    else:
        mask_trainer.train()





