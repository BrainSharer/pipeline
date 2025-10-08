import os
from glob import glob
from pathlib import Path
import warnings


import numpy as np
from PIL import Image


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


from tqdm import tqdm
import cv2
import argparse



# Dataset class
class BrainMaskDataset(Dataset):
    """Dataset that returns (image_tensor, mask_tensor). Expects matching filenames in two folders."""
    def __init__(self, images_dir, masks_dir, transform=None, resize=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.resize = resize

        # Build list of files by matching basenames
        image_files = sorted(self.images_dir.glob('*'))
        # Keep only files that have a matching mask
        self.files = []
        for img_path in image_files:
            mask_path = self.masks_dir / img_path.name
            if mask_path.exists():
                self.files.append((img_path, mask_path))
        if len(self.files) == 0:
            raise RuntimeError(f'No paired images/masks found in {images_dir} and {masks_dir}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, mask_path = self.files[idx]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.resize is not None:
            img = img.resize(self.resize, resample=Image.BILINEAR)
            mask = mask.resize(self.resize, resample=Image.NEAREST)

        img = np.array(img).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32) / 255.0
        # if mask is 0/255, dividing by 255 converts to 0/1
        if mask.max() > 1.0:
            mask = (mask > 127).astype(np.float32)

        # HWC -> CHW
        img = np.transpose(img, (2,0,1))
        mask = np.expand_dims(mask, 0)

        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(mask)

        if self.transform is not None:
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

        return img_tensor, mask_tensor

# Basic U-Net implementation
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part
        for feat in features:
            self.downs.append(DoubleConv(in_channels, feat))
            in_channels = feat

        # Up part
        for feat in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feat*2, feat, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feat*2, feat))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            up_transpose = self.ups[idx]
            up_double = self.ups[idx+1]
            x = up_transpose(x)
            skip = skip_connections[idx//2]
            # pad if needed
            if x.shape != skip.shape:
                x = TF.resize(x, skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = up_double(x)

        return torch.sigmoid(self.final_conv(x))

def dice_coeff(pred, target, eps=1e-7):
    # pred and target are tensors with shape [B,1,H,W]
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2*inter + eps) / (union + eps)
    return dice.mean().item()

class DiceBCELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        # soft dice
        smooth = 1.
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(1)
        dice_score = (2.*intersection + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.mean()
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

def train_unet(
    train_loader,
    val_loader,
    device,
    epochs=30,
    lr=1e-3,
    model_save_path='unet_model.pt',
    in_channels=3,
    out_channels=1,
    features=[32,64,128],
    print_every=1
):
    model = UNet(in_channels=in_channels, out_channels=out_channels, features=features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = DiceBCELoss(alpha=0.5)

    best_val_dice = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f'Epoch {epoch} Train'):
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss = train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                preds = model(imgs)
                val_loss += criterion(preds, masks).item() * imgs.size(0)
                val_dice += dice_coeff(preds, masks) * imgs.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        val_dice = val_dice / len(val_loader.dataset)

        if epoch % print_every == 0:
            print(f'Epoch {epoch}/{epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}')

        # Save best
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({'model_state_dict': model.state_dict(), 'features': features, 'in_channels': in_channels, 'out_channels': out_channels}, model_save_path)

    print('Training finished. Best val Dice:', best_val_dice)
    return model

def predict_image(model, device, img_np, resize=None):
    # img_np: HxWxC float [0..1]
    model.eval()
    img = img_np.copy()
    if resize is not None:
        img = cv2.resize(img, tuple(resize[::-1]))
    tensor = torch.from_numpy(np.transpose(img, (2,0,1))).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor)
    pred_np = pred.squeeze().cpu().numpy()  # [H,W]
    return pred_np


def extract_contours_from_mask(bin_mask, min_area=10):
    # bin_mask expected binary 0/1 or 0/255, uint8
    if bin_mask.dtype != np.uint8:
        bin_mask = (bin_mask > 0).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            filtered.append(c)
    return filtered


# Example: set up data loaders and run training (adjust paths and hyperparams)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--epochs', help='# of epochs', required=False, default=2, type=int)
    args = parser.parse_args()
    epochs = args.epochs
    # Paths - change to your dataset location
    data_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/structures/TG"
    images_dir = os.path.join(data_path, "thumbnail_aligned")
    masks_dir = os.path.join(data_path, "thumbnail_masked")
    image_paths = sorted(glob(os.path.join(images_dir, "*.tif")))
    mask_paths = sorted(glob(os.path.join(masks_dir, "*.tif")))

    # Create dataset and split
    full_ds = BrainMaskDataset(images_dir, masks_dir, resize=(256,256))
    n = len(full_ds)
    n_val = max(1, int(0.1 * n))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])
    print(f'Total images: {n}, train: {n_train}, val: {n_val}')

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    if torch.cuda.is_available(): 
        device = torch.device('cuda') 
        print('Using Nvidia graphics card GPU.')
    else:
        warnings.filterwarnings("ignore")
        device = torch.device('cpu')
        print('No Nvidia card found, using CPU.')


    # Train (this will save best model to unet_model.pt)
    model_path = os.path.join(data_path, "unet_model.h5")
    _ = train_unet(train_loader, val_loader, device, epochs=epochs, lr=1e-3, model_save_path=model_path, in_channels=3, out_channels=1, features=[32,64,128])

# Inference + contour drawing example (run after training or if you have a saved model)
# Load model checkpoint and run inference on a single image, then draw contours and show/save the result.


# Example usage:
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = load_model_from_checkpoint('unet_model.pt', device)
# img_path = 'dataset/images/sample_001.png'
# img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32) / 255.0
# pred_mask = predict_image(model, device, img, resize=(256,256))
# bin_mask = (pred_mask > 0.5).astype(np.uint8) * 255
# contours = extract_contours_from_mask(bin_mask, min_area=20)
# # draw contours on original (resized) image
# disp = (cv2.resize((img*255).astype(np.uint8), bin_mask.shape[::-1]))
# cv2.drawContours(disp, contours, -1, (0,255,0), 2)
# plt.imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
# plt.axis('off')

import json

def predict_folder_and_save_polygons(model, device, images_folder, out_json='polygons.json', resize=(256,256), threshold=0.5, min_area=50):
    images = sorted(glob(os.path.join(images_folder, '*')))
    records = []
    for p in images:
        orig = np.array(Image.open(p).convert('RGB')).astype(np.float32) / 255.0
        pred = predict_image(model, device, orig, resize=resize)
        bin_mask = (pred > threshold).astype(np.uint8) * 255
        contours = extract_contours_from_mask(bin_mask, min_area=min_area)
        # convert contours to simplified polygon list in original image coordinates
        h0, w0 = orig.shape[:2]
        h1, w1 = bin_mask.shape
        scale_x = w0 / w1
        scale_y = h0 / h1
        polys = []
        for c in contours:
            pts = c.squeeze().tolist()
            if len(pts) == 0:
                continue
            # scale to original coordinates
            scaled = [[int(pt[0]*scale_x), int(pt[1]*scale_y)] for pt in pts]
            polys.append(scaled)
        records.append({'image': os.path.basename(p), 'polygons': polys})
    with open(out_json, 'w') as f:
        json.dump(records, f)
    print('Wrote polygons to', out_json)
