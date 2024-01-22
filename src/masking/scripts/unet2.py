import tensorflow as tf
import os
import random
import numpy as np
 
from tqdm import tqdm 

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
    
seed = 42
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1

IMAGE_PATH = '/home/eddyod/tmp/images/'
MASK_PATH = '/home/eddyod/tmp/masks/'
files = sorted(os.listdir(IMAGE_PATH))

X = np.zeros((len(files), IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
y = np.zeros((len(files), IMG_HEIGHT, IMG_WIDTH), dtype=bool)


print('Resizing training images and masks')
for n, file in enumerate(tqdm(files)):
    filepath = IMAGE_PATH + file
    maskpath = MASK_PATH + file
    img = imread(filepath)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    mask = imread(maskpath)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X[n] = img  #Fill empty X_train with values from img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=bool)            
    y[n] = mask 
    
