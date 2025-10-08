"""
UNet training and inference script for contouring specific brain areas from grayscale TIFF sections.

Features:
- PyTorch U-Net implementation (single-channel input, single-channel sigmoid output)
- Dataset class that loads TIFF grayscale images and binary masks
- Training function with Dice+BCE loss, checkpointing, and simple metric logging
- Inference function that supports arbitrary-size TIFFs using tiled sliding-window inference with overlap and Gaussian blending
- Contour extraction using skimage.measure.find_contours (returns subpixel contours)
- Example usage at the bottom

Dependencies:
- torch, torchvision
- numpy, tifffile, Pillow
- scikit-image (skimage), opencv-python (optional), tqdm

Run example:
    python unet_brain_contour.py --train --data_csv my_dataset.csv --epochs 40

where my_dataset.csv has two columns: "image_path","mask_path" (absolute or relative paths)

"""

import os
import math
import random
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from PIL import Image
import tifffile as tiff

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms

from skimage import measure

# -----------------------------
# Model: U-Net (small, configurable)
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, mid_ch=in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY != 0 or diffX != 0:
            x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, base_c=32, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.outc = nn.Conv2d(base_c, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# -----------------------------
# Dataset
# -----------------------------
class BrainSectionDataset(Dataset):
    def __init__(self, transform=None, target_transform=None, patch_size=None):
        self.samples = []
        data_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/structures/TG"
        images_path = os.path.join(data_path, 'thumbnail_aligned')
        masks_path = os.path.join(data_path, 'thumbnail_masked')
        images = sorted(os.listdir(images_path))
        masks = sorted(os.listdir(masks_path))

        for image_path, mask_path in zip(images, masks):
                img = os.path.join(images_path, image_path)
                msk = os.path.join(masks_path, mask_path)
                if img and msk:
                    self.samples.append((img, msk))

        self.transform = transform
        self.target_transform = target_transform
        self.patch_size = patch_size

    def __len__(self):
        l = len(self.samples)
        return l

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = tiff.imread(img_path)
        mask = tiff.imread(mask_path)
        # ensure 2D
        if img.ndim == 3:
            # take first channel if multi-channel
            img = img[..., 0]
        if mask.ndim == 3:
            mask = mask[..., 0]

        img = img.astype(np.float32)
        mask = (mask > 0).astype(np.float32)

        # normalize image to 0-1
        if img.max() > 0:
            img = img / img.max()

        
        if self.patch_size is not None:
            h, w = img.shape
            ph, pw = self.patch_size
            if h >= ph and w >= pw:
                top = random.randint(0, h - ph)
                left = random.randint(0, w - pw)
                img = img[top:top + ph, left:left + pw]
                mask = mask[top:top + ph, left:left + pw]
            else:
                # pad
                pad_h = max(0, ph - h)
                pad_w = max(0, pw - w)
                img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant')
                mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')
                img = img[:ph, :pw]
                mask = mask[:ph, :pw]
        
        # transforms
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).unsqueeze(0)  # 1,H,W
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = torch.from_numpy(mask).unsqueeze(0)

        return img.float(), mask.float()


# -----------------------------
# Loss & metric
# -----------------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2, 3)) + self.smooth
        den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth
        dice = (num / den).mean()
        return 1 - dice


# -----------------------------
# Training function
# -----------------------------

def train_unet(model_save_dir: str,
               epochs: int = 30,
               batch_size: int = 8,
               patch_size: Tuple[int, int] = (512, 512),
               lr: float = 1e-3,
               device: str = None):
    
    if torch.cuda.is_available(): 
        device = torch.device('cuda') 
        print('Using Nvidia graphics card GPU.')
    else:
        device = torch.device('cpu')
        print('No Nvidia card found, using CPU.')

    os.makedirs(model_save_dir, exist_ok=True)

    # basic transforms: random flips/rotations
    def aug(img):
        # img is numpy HxW
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
        if random.random() > 0.5:
            img = np.flipud(img).copy()
        # small rotations
        k = random.choice([0, 1, 2, 3])
        img = np.rot90(img, k).copy()
        return torch.from_numpy(img).unsqueeze(0).float()

    train_ds = BrainSectionDataset(transform=aug, target_transform=lambda m: torch.from_numpy(m).unsqueeze(0).float(), patch_size=patch_size)
    val_ds = BrainSectionDataset(transform=lambda x: torch.from_numpy(x).unsqueeze(0).float(), target_transform=lambda m: torch.from_numpy(m).unsqueeze(0).float(), patch_size=patch_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = UNet(n_channels=1, n_classes=1, base_c=32)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    best_val_loss = 1e9

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            loss = 0.5 * bce(logits, masks) + 0.5 * dice(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                imgs = imgs.to(device)
                masks = masks.to(device)
                logits = model(imgs)
                loss = 0.5 * bce(logits, masks) + 0.5 * dice(logits, masks)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}: train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        # save checkpoint
        ckpt_path = os.path.join(model_save_dir, f"unet_epoch{epoch:03d}_vl{val_loss:.4f}.pth")
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(model_save_dir, 'best_unet.pth')
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, best_path)
            print(f"Saved best model: {best_path}")

    return model


# -----------------------------
# Inference: tiled sliding window with blending
# -----------------------------

def _make_gaussian_weight(h, w, sigma_scale=0.25):
    """Create a 2D Gaussian weight window for blending tiles"""
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    xv, yv = np.meshgrid(x, y)
    sigma = sigma_scale
    gauss = np.exp(-((xv ** 2 + yv ** 2) / (2 * sigma ** 2)))
    return gauss.astype(np.float32)


def predict_large_tif(image_path: str,
                      model: torch.nn.Module,
                      device: str = None,
                      tile_size: int = 512,
                      overlap: int = 64,
                      batch_size: int = 4):
    """
    Predict probability map for a possibly very large grayscale TIFF using tiled inference.
    Returns a float32 numpy array with values [0,1].
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    img = tiff.imread(image_path)
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.float32)
    if img.max() > 0:
        img = img / img.max()
    H, W = img.shape

    stride = tile_size - overlap
    ys = list(range(0, max(H - tile_size + 1, 1), stride))
    xs = list(range(0, max(W - tile_size + 1, 1), stride))
    if ys[-1] + tile_size < H:
        ys.append(H - tile_size)
    if xs[-1] + tile_size < W:
        xs.append(W - tile_size)

    output = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    wtile = _make_gaussian_weight(tile_size, tile_size)

    tiles = []
    coords = []
    for y in ys:
        for x in xs:
            tile = img[y:y + tile_size, x:x + tile_size]
            if tile.shape != (tile_size, tile_size):
                # pad
                th, tw = tile.shape
                pad_h = tile_size - th
                pad_w = tile_size - tw
                tile = np.pad(tile, ((0, pad_h), (0, pad_w)), mode='constant')
            tiles.append(tile)
            coords.append((y, x))

    # batch inference
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i + batch_size]
            batch_coords = coords[i:i + batch_size]
            batch_arr = np.stack(batch_tiles, axis=0)  # B,H,W
            batch_arr = torch.from_numpy(batch_arr).unsqueeze(1).to(device)
            logits = model(batch_arr)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # B,H,W
            for p, (y, x) in zip(probs, batch_coords):
                ph, pw = p.shape
                if ph != tile_size or pw != tile_size:
                    p = p[:tile_size, :tile_size]
                output[y:y + tile_size, x:x + tile_size] += (p * wtile)
                weight[y:y + tile_size, x:x + tile_size] += wtile

    # avoid div by zero
    nz = weight > 0
    output[nz] = output[nz] / weight[nz]
    output[~nz] = 0.0
    return output


# -----------------------------
# Contour extraction
# -----------------------------

def probmap_to_contours(probmap: np.ndarray, threshold: float = 0.5) -> List[np.ndarray]:
    """Convert a probability map to a list of contours (Nx2 arrays in image coordinates).
    Uses skimage.measure.find_contours which returns subpixel coordinates in (row, col) format.
    We'll convert to (x,y) as (col, row) for typical plotting.
    """
    mask = (probmap >= threshold).astype(np.uint8)
    contours = []
    # find_contours on the binary mask gives coordinates for each isocontour
    found = measure.find_contours(mask, 0.5)
    for c in found:
        # c is array shape (N,2) with (row, col)
        coords = np.stack([c[:, 1], c[:, 0]], axis=1)  # (x,y)
        print(coords)
        contours.append(coords)
    return contours


# Optionally draw contours onto an image using PIL
from PIL import ImageDraw

def draw_contours_on_image(image: np.ndarray, contours: List[np.ndarray], line_width: int = 2) -> Image.Image:
    """Return a PIL image (RGB) with contours overlaid on the grayscale image."""
    if image.max() > 0:
        norm = (image / image.max() * 255).astype(np.uint8)
    else:
        norm = (image * 255).astype(np.uint8)
    if norm.ndim == 2:
        im = Image.fromarray(norm).convert('RGB')
    else:
        im = Image.fromarray(norm)
    draw = ImageDraw.Draw(im)
    for c in contours:
        # convert to list of tuples
        pts = [(float(x), float(y)) for x, y in c]
        draw.line(pts + [pts[0]], width=line_width, fill=(255, 0, 0))
    return im

# -----------------------------
# CLI & example usage
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--predict', type=str, help='Path to a TIFF to run inference on')
    parser.add_argument('--out_mask', type=str, help='Path to save predicted mask (.tif)')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    data_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/structures/TG"
    model_dir = os.path.join(data_path, 'models')
    os.makedirs(model_dir, exist_ok=True)
    if args.train:
        print('Starting training...')
        train_unet(model_dir, epochs=args.epochs)
        print('Training done.')

    if args.predict:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = UNet(n_channels=1, n_classes=1, base_c=32)
        model_path = os.path.join(model_dir, 'best_unet.pth')
        if os.path.isfile(model_path):
            print(f'Loading model from {model_path}')
        else:
            print(f'Model file not found: {model_path}')
            exit(1)        
        output_dir = os.path.join(data_path, 'predictions')
        os.makedirs(output_dir, exist_ok=True)
        ck = torch.load(model_path, map_location=device)
        model.load_state_dict(ck['model_state'] if 'model_state' in ck else ck)
        infile = os.path.join(data_path, 'thumbnail_aligned', args.predict)
        if os.path.isfile(infile):
            print(f'Predicting: {infile}')
        else:
            print(f'Input file not found: {infile}')
            exit(1)
        prob = predict_large_tif(infile, model, device=device, tile_size=512, overlap=64)
        
        # convert to uint8 mask
        m = (prob >= args.threshold).astype(np.uint8) * 255
        m = m.astype(np.uint8)
        ids, counts = np.unique(m, return_counts=True)
        print(f'Unique mask ids: {ids}')
        print(f'Unique mask counts: {counts}')
        exit(1)
        outputmask_path = os.path.join(output_dir, f'mask_{args.predict}')
        tiff.imwrite(outputmask_path, m.astype(np.uint8))
        print(f'Saved mask to {outputmask_path}')
        # extract contours and save overlay
        contours = probmap_to_contours(prob, threshold=args.threshold)
        print(f'Found {len(contours)} contours')
        img = tiff.imread(infile).astype(np.float32)
        if img.ndim == 3:
            img = img[..., 0]
        overlay = draw_contours_on_image(img, contours)
        output_path = os.path.join(output_dir, f'overlay_{args.predict}')
        #overlay.save(output_path)
        tiff.imwrite(output_path, prob)
        print(f'Saved overlay to {output_path}')

    if not (args.train or args.predict):
        print('No action requested. Use --train to train or --predict  to predict.')
