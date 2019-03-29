from PIL import Image
import numpy as np


def normalize(img):
    img = img.astype('float')
    img = img - img.min()
    if img.max() != 0:
        img = img / img.max()
    return img


def match_size(imgLabel, targetSize=(512, 512)):
    img, label = imgLabel

    if label is not None:
        label = label[0]
        pic = Image.fromarray((label * 255).astype('uint8'))\
            .resize(targetSize)
        label = np.array(pic)
        label = label.astype('float') / 255
        label = np.expand_dims(label, 0)
    # Resize Image
    img = img[0]
    pic = Image.fromarray((img * 255).astype('uint8'))\
        .resize(targetSize)
    img = np.array(pic)
    img = img.astype('float') / 255
    img = np.expand_dims(img, 0)

    return (img, label)


def stich_pair(imgLabel):
    stiched = np.hstack(imgLabel)
    return stiched
