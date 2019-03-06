from constants import *
from data_loading import *
from unet import UNet

import torch

thunks = loadThunks(mri_data)

net = UNet(1,1)

rawImg = thunks[0][0][0]
rawImgTensor = torch.from_numpy(rawImg).float().unsqueeze(1)

print(rawImgTensor.type())

label = net(rawImgTensor)
