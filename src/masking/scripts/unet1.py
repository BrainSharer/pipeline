import torch
from torch.utils.data import DataLoader
from unet import UNet
from tqdm import tqdm
import sys
from pathlib import Path

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.mask_utilities.mask_class import EllipseDataset, MaskDataset, TrigeminalDataset, get_model_instance_segmentation, get_transform
from library.mask_utilities.utils import collate_fn

dtype = torch.float32
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print('using device:', device)

if __name__ == "__main__":
    model = UNet(in_chans=1, depth=1, layers=1, skip_connection=True)
    model.to(device, dtype=dtype)
    
    batch_size = 4
    lr = 1e-3
    dataset = EllipseDataset()
    #loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    criterion = torch.nn.BCEWithLogitsLoss()
    sigmoid = torch.nn.Sigmoid()

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    epochs = 5
    model.train()
    for e in range(epochs):
        print("Epoch {0} out of {1}".format(e+1, epochs))
        print("_"*10)
        epoch_loss = 0.0

        for t, (x, y) in enumerate(loader):
            x = x.to(device, dtype=dtype)
            scores = model(x)
            y = y.to(device).type_as(scores)
            loss = criterion(scores, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                epoch_loss += loss.item()
        print(f"epoch loss: {epoch_loss}")

    # save model
    torch.save(model.state_dict(), "unet.pth")