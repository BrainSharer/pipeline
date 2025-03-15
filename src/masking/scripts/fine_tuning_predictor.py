import torch
from torchvision import models


model_path = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks/fine_tuned_model.pth'

model = models.squeezenet1_0(weights=None)
model.load_state_dict(torch.load(model_path))
model.eval()


print(model)