from config import *

from mri_dataloader import ColonSegDataset, NumpyToTensor, NumpyRot, NumpyFlip
from util.data_io import save_object, load_object

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torchvision.transforms import Compose
from unet import UNet

import pdb

net = UNet(1, 1).to(device)
tsfms = Compose([NumpyRot(90, 0.5),
                 NumpyFlip(0.5),
                 NumpyFlip(0.5, False),
                 NumpyToTensor()])

train_pairs = ColonSegDataset('./mri_data/labeled', 2012, tsfms, False)
valid_pairs = ColonSegDataset(
    './mri_data/labeled', 2012, NumpyToTensor(), True)

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters()) # optim.Adamax(net.parameters())
# optim.Adam(net.parameters())


def train_model(model, criterion, optmizer, epoches, train_set, valid_set):
    import copy

    history = {'train': [], 'valid': []}
    data_load = {
        'train': DataLoader(train_set, **loader_config),
        'valid': DataLoader(valid_set, **loader_config)}

    best_wts = copy.deepcopy(model.state_dict())
    best_loss = 111111

    for epoch in range(epoches):
        print(f"Epoch: {epoch}")
        for phase in ['train', 'valid']:
            print(f"Phase: {phase}")
            accum_loss = 0
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for inputs, mask in data_load[phase]:

                inputs = inputs.to(device)
                mask = mask.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    outputs_flat = outputs.view(-1)
                    mask_flat = mask.view(-1)

                    loss = criterion(outputs_flat, mask_flat)

                    if phase == 'train':

                        optmizer.zero_grad()
                        loss.backward()
                        optmizer.step()
                    accum_loss = accum_loss + loss.item() * inputs.shape[0]
                    # accum_loss = accum_loss / data_size[phase]
                    history[phase].append(loss.item())
                    print(f"{phase} Loss: {loss.item():.4f}")
            print(f'-------total loss: {accum_loss}')
            if phase == 'valid' and accum_loss < best_loss:
                best_loss = accum_loss
                best_wts = copy.deepcopy(model.state_dict())
                print("saving new weights")
            if phase == 'valid':
                print("--------------------Finish One Epoch.--------------\n")

    print("Training Completed")
    print(f"Best Loss: {best_loss}")

    torch.save(best_wts, SAVING_PATH+'/model_weight')
    save_object(history, SAVING_PATH+'/model_history')


train_model(net, criterion, optimizer, epoches, train_pairs, valid_pairs)
