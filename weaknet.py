import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Given an input of 512x512 images
trying to predict its segmentation mask of size 512x512
'''
class SegNet(nn.Module):

    def __init__(self):
        super(SegNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # pooling / unpooling
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        # kernel
        # input, output, size
        self.conv1 = nn.Conv2d(1, 3, 7, padding=3)        
        self.conv2 = nn.Conv2d(3, 5, 5, padding=2)
        self.conv3 = nn.Conv2d(5, 7, 3, padding=1)

        self.conv4 = nn.Conv2d(7, 9, 1)
        self.deconv1 = nn.ConvTranspose2d(9, 7, 1)
        self.deconv2 = nn.ConvTranspose2d(7, 5, 3)
        self.deconv3 = nn.ConvTranspose2d(5, 3, 5)
        self.deconv4 = nn.ConvTranspose2d(3, 1, 7)

    def forward(self, x):
        def layer(conv, x):
            return F.selu(conv(x))

        x, idx1 = self.pool(layer(self.conv1, x))
        print(x.shape)
        x, idx2 = self.pool(layer(self.conv2, x))
        print(x.shape)
        x, idx3 = self.pool(layer(self.conv3, x))
        print(x.shape)
        x = layer(self.conv4, x)
        print(x.shape)
        x = layer(self.deconv1, x)
        print(x.shape)
        x = layer(self.deconv2, x)
        print(x.shape)
        x = layer(self.deconv3, x)
        print(x.shape)
        x = layer(self.deconv4, x)
        print(x.shape)
        return x