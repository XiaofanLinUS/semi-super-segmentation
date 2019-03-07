import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
import pydicom

from util.data_loading import getSubdirs, save_object, load_object

from config import *

def normalize(img):
    img = img - img.min()
    img = img / img.max()
    return img

class NumpyRot(object):
    def __init__(self, degree, probability = 0.5):
        self.d = degree
        self.p = probability
    def __call__(self, imgLabelPair):
        from scipy import ndimage
        img, mask = imgLabelPair

        if (np.random.rand(1).item() > 1 - self.p):
            img = ndimage.rotate(img, self.d, reshape=False)
            mask = ndimage.rotate(mask, self.d, reshape=False)
        return img, mask
     
class NumpyFlip(object):
    def __init__(self, probablity=0.5, horizonotal=True):
        # probablity
        self.p = probablity
        self.h = horizonotal
    
    def __call__(self, imgLabelPair):
        img, mask = imgLabelPair

        if (np.random.rand(1).item() > 1 - self.p):
            if(self.h):
                img = np.flip(img, axis=1)
                mask = np.flip(mask, axis=1)
            else:
                img = np.flip(img, axis=0)
                mask = np.flip(img, axis=0)
        
        return (img, mask)

class NumpyToTensor(object):
    def __call__(self, imgLabelPair):
        img, mask = imgLabelPair

        return (torch.from_numpy(img).float(), torch.from_numpy(mask).float())


def get_PIL_image(img):
    img = normalize(img)
    img = np.uint8(img * 255)
    return Image.fromarray(img)

'''
    Retrive the information of filename from the directory and store it
    as a bunch of pairs image and segmentaion mask files

'''

class ColonSegDataset(Dataset):
    """Colon Half-Segmentation dataset."""

    def __init__(self, root_dir, seed=921, transform=None, valid=False, valid_portion=0.1):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data_info = getSubdirs(root_dir)
        self.indices = np.random.RandomState(seed).permutation(len(self.data_info))
        if(valid):
            self.indices = self.indices[int((1-valid_portion)*len(self.data_info)):]
        else:
            self.indices = self.indices[:int((1-valid_portion)*len(self.data_info))]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img = normalize(pydicom.dcmread(self.data_info[idx][self.indices]).pixel_array)
        mask = normalize(pydicom.dcmread(self.data_info[idx][self.indices]).pixel_array)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return (img, mask)








            


                    