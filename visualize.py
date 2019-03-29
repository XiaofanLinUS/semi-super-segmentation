from config import *

from mri_dataloader import ColonRawDataset, NumpyToTensor, NumpyRot, NumpyFlip
from util.data_io import save_object, load_object
from util.data_process import stich_pair
from util.data_visual import play_img_sequence, save_img_sequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torchvision.transforms import Compose
from unet import UNet


net = UNet(1, 1).to(device)
net.load_state_dict(torch.load('./model_25random_lrelu/model_weight'))
net.eval()

tsfms = Compose([NumpyRot(25,0.5,True), NumpyFlip(0.5), NumpyFlip(0.5, False),NumpyToTensor()])

test_set = ColonRawDataset('./mri_data/unlabeled', 921, tsfms)

predict_seq = []
for img, mask in test_set:
    img = img.to(device)
    img = img.unsqueeze(0)
    with torch.no_grad():
        predict = net(img)
        predict_seq.append(stich_pair((img.cpu().numpy()[
                           0, 0], predict.cpu().numpy()[0, 0])))

# play_img_sequence(predict_seq, .5)
save_img_sequence(predict_seq, 'result_25random_lrelu', True, 1)