import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
from skimage.io import imread
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x

def encoder_block(x, filters):
    f = conv_block(x, filters)
    p = layers.MaxPooling2D((2, 2))(f)
    return f, p

def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def build_unet(input_shape=(256, 256, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    f1, p1 = encoder_block(inputs, 64)
    f2, p2 = encoder_block(p1, 128)
    f3, p3 = encoder_block(p2, 256)
    f4, p4 = encoder_block(p3, 512)

    # Bottleneck
    bottleneck = conv_block(p4, 1024)

    # Decoder
    d1 = decoder_block(bottleneck, f4, 512)
    d2 = decoder_block(d1, f3, 256)
    d3 = decoder_block(d2, f2, 128)
    d4 = decoder_block(d3, f1, 64)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model

data_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/structures/TG"
image_dir = os.path.join(data_path, "thumbnail_aligned")
mask_dir = os.path.join(data_path, "thumbnail_masked")
image_paths = sorted(glob(os.path.join(image_dir, "*.tif")))
mask_paths = sorted(glob(os.path.join(mask_dir, "*.tif")))

print(f"Found {len(image_paths)} images and {len(mask_paths)} masks.")

# Load and preprocess (resize to uniform shape)
IMG_HEIGHT, IMG_WIDTH = 256, 256

def load_data(img_paths, mask_paths):
    X, Y = [], []
    for img_p, mask_p in zip(img_paths, mask_paths):
        img = imread(img_p)
        mask = imread(mask_p)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        mask = (mask > 0).astype(np.float32)
        X.append(img)
        Y.append(np.expand_dims(mask, -1))
    X = np.array(X, dtype=np.float32) / 255.0
    Y = np.array(Y, dtype=np.float32)
    return X, Y

def train_unet(model, X_train, Y_train, X_val, Y_val, epochs=20, batch_size=8):
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size
    )

    return history

def run_model(epochs=10):
    
    X, Y = load_data(image_paths, mask_paths)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("Train shape:", X_train.shape, Y_train.shape)
    print("Validation shape:", X_val.shape, Y_val.shape)

    model = build_unet(input_shape=X_train.shape[1:])

    history = train_unet(model, X_train, Y_train, X_val, Y_val, epochs=epochs)
    model_path = os.path.join(data_path, "unet_model.h5")
    model.save(model_path)
    print(f"Model saved as {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--epochs', help='# of epochs', required=False, default=2, type=int)
    
    args = parser.parse_args()
    epochs = args.epochs

    run_model(epochs=epochs)


