from mri_dataloader import ColonSegDataset

import torch
import torch.nn as nn
import torch.utils.data.DataLoader as Dataloader
from torch import optim

from config import *
from unet import UNet




net = UNet(1,1)

train_pairs = ColonSegDataset('./mri_data/labeled',921, None, True)
valid_pairs = ColonSegDataset('./mri_data/labeled',921, None, False)
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters())
train_model(net, criterion, optimizer, epoches, train_pairs, valid_pairs)

def train_model(model, criterion, optmizer, epoches, train_set, valid_set):
    import copy

    history = {'train':[], 'valid':[]}
    data_load = {'train': DataLoader(train_set, *loader_config), 'valid': DataLoader(valid_set, *loader_config)}
    train_size = len(train_set)
    valid_size = len(valid_set)
    data_size = {'train': train_size, 'valid': valid_size}
    
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
                    accum_loss = accum_loss / data_size[phase]
                    history[phase].append(accum_loss)
                    print(f"{phase} Loss: {accum_loss:.4f}")

                    if phase == 'valid' and accum_loss < best_loss:
                        best_loss = accum_loss
                        best_wts = copy.deepcopy(model.state_dict())
                        print("saving new weights")
    
    print("Training Completed")
    print(f"Best Loss: {best_loss}")

    torch.save(best_wts, SAVING_PATH+'/model_weight')
    save_object(history, SAVING_PATH+'/model_history')
