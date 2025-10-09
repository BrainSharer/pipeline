"""
unet_brain_segmentation.py

Requirements:
    pip install torch torchvision tifffile opencv-python scikit-image albumentations matplotlib
"""

import os
import glob
import random
from typing import List, Tuple, Optional

import numpy as np
import tifffile
import cv2
from skimage import exposure

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------------------------
# Utilities
# -------------------------
def read_tif_gray(path: str) -> np.ndarray:
    """Read a grayscale tiff and return as float32 image 0..1"""
    img = tifffile.imread(path)
    if img.ndim == 3:
        # if it's (channels, h, w) or (h, w, c) - try to collapse channels by averaging
        if img.shape[0] <= 4 and img.shape[0] != img.shape[-1]:
            img = np.mean(img, axis=0)
        else:
            img = np.mean(img, axis=-1)
    img = img.astype(np.float32)
    # normalize contrast using percentile stretch then scale to 0-1
    p1, p99 = np.percentile(img, (1, 99))
    if p99 > p1:
        img = np.clip((img - p1) / (p99 - p1), 0.0, 1.0)
    else:
        # fallback normalization
        img = (img - img.min()) / max(1e-8, img.max() - img.min())
    return img


def save_tif(img: np.ndarray, path: str):
    tifffile.imwrite(path, (img * 255).astype(np.uint8))


# -------------------------
# Contour drawing utilities
# -------------------------
def mask_to_contours(mask: np.ndarray, threshold: float = 0.5) -> List[np.ndarray]:
    """
    Convert model probability mask (float 0..1) to OpenCV contours.
    Returns list of contours; each contour is an Nx1x2 int32 array (OpenCV format).
    """
    if mask.dtype != np.uint8:
        bin_mask = (mask >= threshold).astype(np.uint8) * 255
    else:
        bin_mask = (mask > 0).astype(np.uint8) * 255

    # findContours expects uint8
    contours_info = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    return contours


def draw_contours_on_image(image: np.ndarray, contours: List[np.ndarray],
                           color: Tuple[int, int, int] = (255, 0, 0),
                           thickness: int = 2) -> np.ndarray:
    """
    Draw contours on a grayscale image. Returns an RGB image (uint8).
    - image: float 0..1 or uint8 h x w
    - color: BGR tuple for OpenCV drawing (we pass as RGB but cv2 expects BGR - below we convert)
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        base = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        base = image.copy().astype(np.uint8)

    if base.ndim == 2:
        rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        rgb = base

    # OpenCV uses BGR order
    bgr_color = (int(color[2]), int(color[1]), int(color[0]))
    cv2.drawContours(rgb, contours, -1, bgr_color, thickness)
    return rgb


# -------------------------
# Dataset
# -------------------------
class TiffMaskDataset(Dataset):
    """
    Dataset for grayscale TIFF images and binary mask TIFFs.
    images: list of image paths
    masks: list of mask paths (same order). masks should be single-channel (0 or 255 or 0..1)
    patch_size: size to resize to (square). aspect preserved by simple resizing.
    augment: albumentations transform or None
    """
    def __init__(self, images: List[str], masks: List[str], patch_size: int = 512,
                 augment: Optional[A.BasicTransform] = None):
        assert len(images) == len(masks)
        self.images = images
        self.masks = masks
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = read_tif_gray(self.images[idx])  # float32 0..1
        mask = tifffile.imread(self.masks[idx]).astype(np.float32)
        if mask.ndim == 3:
            mask = np.mean(mask, axis=0)
        # normalize mask to 0..1
        if mask.max() > 1:
            mask = mask / 255.0
        mask = (mask > 0.5).astype(np.float32)

        # resize both to patch_size
        img_resized = cv2.resize(img, (self.patch_size, self.patch_size),
                                 interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (self.patch_size, self.patch_size),
                                  interpolation=cv2.INTER_NEAREST)

        if self.augment is not None:
            augmented = self.augment(image=img_resized, mask=mask_resized)
            img_resized = augmented['image']
            mask_resized = augmented['mask']

        # to tensor: (C,H,W)
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0).float()  # 1xHxW
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float()
        return img_tensor, mask_tensor


# -------------------------
# U-Net model (PyTorch)
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super().__init__()
        f = base_filters
        self.inc = DoubleConv(in_channels, f)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f, f*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f*2, f*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f*4, f*8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(f*8, f*8))

        self.up1 = nn.ConvTranspose2d(f*8, f*8, 2, stride=2)
        self.conv_up1 = DoubleConv(f*16, f*8)
        self.up2 = nn.ConvTranspose2d(f*8, f*4, 2, stride=2)
        self.conv_up2 = DoubleConv(f*8, f*4)
        self.up3 = nn.ConvTranspose2d(f*4, f*2, 2, stride=2)
        self.conv_up3 = DoubleConv(f*4, f*2)
        self.up4 = nn.ConvTranspose2d(f*2, f, 2, stride=2)
        self.conv_up4 = DoubleConv(f*2, f)

        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.inc(x)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        c5 = self.down4(c4)

        u1 = self.up1(c5)
        u1 = torch.cat([u1, c4], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, c3], dim=1)
        u2 = self.conv_up2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, c2], dim=1)
        u3 = self.conv_up3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, c1], dim=1)
        u4 = self.conv_up4(u4)

        out = self.outc(u4)
        return out  # logits


# -------------------------
# Training function
# -------------------------
def train_unet(train_images: List[str], train_masks: List[str],
               val_images: List[str], val_masks: List[str],
               out_dir: str,
               patch_size: int = 512,
               batch_size: int = 8,
               epochs: int = 50,
               lr: float = 1e-3,
               base_filters: int = 32,
               device: Optional[str] = None,
               save_every: int = 5):
    """
    Train a UNet model. Saves best model (by val loss) to out_dir/best_model.pth and last model to last_model.pth.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Augmentations
    train_aug = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.8, 1.2), translate_percent=0.1, rotate=(-15, 15), shear=(-10, 10), p=0.5),
        A.RandomBrightnessContrast(p=0.3),
    ])
    val_aug = None

    train_ds = TiffMaskDataset(train_images, train_masks, patch_size, augment=train_aug)
    val_ds = TiffMaskDataset(val_images, val_masks, patch_size, augment=val_aug)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True if device=='cuda' else False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True if device=='cuda' else False)

    model = UNet(in_channels=1, out_channels=1, base_filters=base_filters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    best_val = float('inf')

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item() * imgs.size(0)
        val_loss = val_loss / len(val_loader.dataset)

        scheduler.step(val_loss)
        print(f"Epoch {epoch}/{epochs}  TrainLoss: {train_loss:.4f}  ValLoss: {val_loss:.4f}")

        # save last
        torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict()}, os.path.join(out_dir, 'last_model.pth'))

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'optim_state': optimizer.state_dict()}, os.path.join(out_dir, 'best_model.pth'))

        if epoch % save_every == 0:
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'optim_state': optimizer.state_dict()}, os.path.join(out_dir, f"model_epoch_{epoch}.pth"))

    print("Training finished. Best val loss:", best_val)
    return os.path.join(out_dir, 'best_model.pth')


# -------------------------
# Prediction / Inference
# -------------------------
def load_model(weights_path: str, device: Optional[str] = None, base_filters: int = 32) -> Tuple[nn.Module, str]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1, base_filters=base_filters)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device).eval()
    return model, device


def predict_and_draw(input_tif: str, model: nn.Module, device: str,
                     patch_size: int = 512, threshold: float = 0.5,
                     contour_color: Tuple[int, int, int] = (1.0, 0.0, 0.0),
                     min_contour_area: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load input_tif, run model, return (pred_mask, overlay_rgb)
    - pred_mask: float32 resized to original image size (0..1)
    - overlay_rgb: uint8 RGB image with contours drawn on original image
    """
    orig = read_tif_gray(input_tif)  # 0..1
    h0, w0 = orig.shape[:2]
    img_resized = cv2.resize(orig, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # HxW

    # upsample mask back to original size
    probs_up = cv2.resize(probs, (w0, h0), interpolation=cv2.INTER_LINEAR)
    # threshold
    contours = mask_to_contours((probs_up * 255).astype(np.uint8), threshold=threshold)

    # filter small contours
    filtered = []
    for c in contours:
        if cv2.contourArea(c) >= min_contour_area:
            filtered.append(c)

    # contour_color in floats 0..1 to ints 0..255
    color_rgb = tuple(int(255 * float(c)) for c in contour_color)

    overlay = draw_contours_on_image(orig, filtered, color=color_rgb, thickness=2)
    return probs_up, overlay


# -------------------------
# Example usage
# -------------------------
# --------------------------- Example CLI --------------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['train','predict'], default='train', required=False, type=str)
    parser.add_argument('--epochs', help='# of epochs', required=False, default=2, type=int)
    parser.add_argument('--samples', help='# of samples for debugging', required=False, default=10, type=int)
    parser.add_argument('--debug', help='test model', required=False, default='false', type=str)
    args = parser.parse_args()

    task = args.task
    epochs = args.epochs
    samples = args.samples
    debug = bool({'true': True, 'false': False}[args.debug.lower()])
    input_tif = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK78/preps/C1/thumbnail_aligned/101.tif"
    all_imgs = []
    all_masks = []

    data_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/structures/TG"
    img_root = os.path.join(data_path, 'thumbnail_aligned')
    mask_root = os.path.join(data_path, 'thumbnail_masked')
    model_root = os.path.join(data_path, 'models')
    out_dir = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK78/preps/predictions"
    os.makedirs(out_dir, exist_ok=True)
    # example file lists (replace with your paths)
    all_imgs = sorted(os.listdir(img_root))
    all_masks = sorted(os.listdir(mask_root))
    for img, mask in zip(all_imgs, all_masks):
        assert img.split('.')[0] == mask.split('.')[0], f"Image and mask filenames do not match: {img} vs {mask}"
    all_imgs = sorted([os.path.join(img_root, f) for f in all_imgs])
    all_masks = sorted([os.path.join(mask_root, f) for f in all_masks])
    if debug:
        all_imgs = all_imgs[:samples]
        all_masks = all_masks[:samples]
    combined = list(zip(all_imgs, all_masks))
    random.shuffle(combined)
    split = int(0.8 * len(combined))
    train_pairs = combined[:split]
    val_pairs = combined[split:]
    train_images, train_masks = zip(*train_pairs)
    val_images, val_masks = zip(*val_pairs)

    print(f"Training on {len(train_images)} images, validating on {len(val_images)} images.")

    best_path = train_unet(list(train_images), list(train_masks),
                           list(val_images), list(val_masks),
                           model_root,
                           patch_size=512, batch_size=8, epochs=epochs, lr=1e-3)

    # inference example
    model, device = load_model(best_path)
    pred_mask, overlay = predict_and_draw(input_tif, model, device, patch_size=512, threshold=0.5)

    # save outputs
    save_tif(pred_mask, os.path.join(out_dir, "pred_mask_example.tif"))
    tifffile.imwrite(os.path.join(out_dir, "overlay_example.tif"), overlay)
    print("Saved example outputs to", out_dir)


