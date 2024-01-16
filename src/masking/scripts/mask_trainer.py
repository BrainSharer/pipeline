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

from library.mask_utilities.mask_class import MaskDataset, TrigeminalDataset, get_model_instance_segmentation, get_transform
from library.mask_utilities.utils import collate_fn
from library.mask_utilities.engine import train_one_epoch

ROOT = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='specify animal', required=False)
    parser.add_argument('--debug', help='test model', required=False, default='false')
    parser.add_argument('--tg', help='Use TG masks', required=False, default='false')
    parser.add_argument('--epochs', help='# of epochs', required=False, default=2)
    
    args = parser.parse_args()
    tg = bool({'true': True, 'false': False}[args.tg.lower()])
    debug = bool({'true': True, 'false': False}[args.debug.lower()])
    animal = args.animal
    epochs = int(args.epochs)

    if tg:
        ROOT = os.path.join(ROOT, 'tg')
        dataset = TrigeminalDataset(ROOT, transforms = get_transform(train=True))
    else:
        dataset = MaskDataset(ROOT, animal, transforms = get_transform(train=True))

    indices = torch.randperm(len(dataset)).tolist()

    if debug:
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
    num_classes = 2
    modelpath = os.path.join(ROOT, 'mask.model.pth')
    # create logging file
    logpath = os.path.join(ROOT, "mask.logger.txt")
    logfile = open(logpath, "w")
    logheader = f"Masking {datetime.now()} with {epochs} epochs\n"
    logfile.write(logheader)
    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    # move model to the right device
    model.to(device)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler which decreases the learning rate by # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    loss_list = []
    
    # original version with train_one_epoch
    for epoch in range(epochs):
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
        if not debug:
            torch.save(model.state_dict(), modelpath)

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




