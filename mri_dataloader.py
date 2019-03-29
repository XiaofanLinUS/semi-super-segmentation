import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
import pydicom
from util.data_io import get_subdirs, save_object, load_object
from util.data_process import stich_pair, match_size, normalize
from PIL import Image
from torchvision.transforms import Compose


class NumpyRot(object):
    def __init__(self, degree, probability=0.5, between=False):
        self.d = degree
        self.p = probability
        self.between = between

    def __call__(self, imgLabelPair):
        from scipy import ndimage
        img, mask = imgLabelPair
        degree = self.d
        if(self.between):
            degree = np.random.randint(self.d)
        if (np.random.rand(1).item() > 1 - self.p):
            img[0] = ndimage.rotate(img[0], degree, reshape=False)
            if mask is not None:
                mask[0] = ndimage.rotate(mask[0], degree, reshape=False)
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
                img = np.flip(img, axis=2)
                if mask is not None:
                    mask = np.flip(mask, axis=2)
            else:
                img = np.flip(img, axis=1)
                if mask is not None:
                    mask = np.flip(mask, axis=1)

        return (img, mask)


class NumpyToTensor(object):
    def __call__(self, imgLabelPair):
        img, mask = imgLabelPair

        if mask is None:
            return (torch.from_numpy(img.copy()).float(),
                    None)

        return \
            (torch.from_numpy(img.copy()).float(),
             torch.from_numpy(mask.copy()).float())


def get_PIL_image(img):
    print(f'img type: {img.shape}')
    img = img[0]
    img = np.uint8(img * 255)
    return Image.fromarray(img)


'''
    Retrive the information of filename from the directory and store it
    as a bunch of pairs image and segmentaion mask files

'''


class ColonRawDataset(Dataset):
    """Colon Half-Segmentation dataset."""

    def __init__(self, root_dir, seed=921, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data_info = get_subdirs(root_dir)
        self.indices = np.random.RandomState(
            seed).permutation(len(self.data_info))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # We need to first check what format it is
        if self.data_info[self.indices[idx]][0].endswith('png'):
            img = np.array(Image.open(self.data_info[self.indices[idx]][0]))
        else:
            img = pydicom.dcmread(self.data_info[self.indices[idx]][0])\
                .pixel_array
        img = np.expand_dims(img, axis=0)
        img = normalize(img)

        pair = (img, None)
        if img.shape != (512, 512):
            pair = match_size(pair)

        if self.transform:
            pair = self.transform(pair)

        return pair


class ColonSegDataset(Dataset):
    """Colon Half-Segmentation dataset."""

    def __init__(self, root_dir, seed=921, transform=None, valid=False,
                 valid_portion=0.1):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data_info = get_subdirs(root_dir)
        self.indices = np.random.RandomState(
            seed).permutation(len(self.data_info))
        if(valid):
            self.indices = self.indices[int(
                (1-valid_portion)*len(self.data_info)):]
        else:
            self.indices = self.indices[:int(
                (1-valid_portion)*len(self.data_info))]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # We need to first check what format it is
        if self.data_info[self.indices[idx]][0].endswith('png'):
            img = np.array(Image.open(self.data_info[self.indices[idx]][0]))
            mask = np.array(Image.open(self.data_info[self.indices[idx]][1]))
        else:
            img = pydicom.dcmread(self.data_info[self.indices[idx]][0])\
                .pixel_array
            mask = pydicom.dcmread(self.data_info[self.indices[idx]][1])\
                .pixel_array
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)
        img = normalize(img)
        mask = normalize(mask)

        pair = (img, mask)
        if mask.shape != (512, 512) and img.shape != (512, 512):
            pair = match_size(pair)

        if self.transform:
            pair = self.transform(pair)

        return pair
