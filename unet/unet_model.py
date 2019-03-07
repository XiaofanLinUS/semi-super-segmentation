# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        #self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        # 128
        x2 = self.down1(x1)
        # 256
        x3 = self.down2(x2)
        # 256
        x4 = self.down3(x3)
        # 512 -> 128
        x = self.up2(x4, x3)
        # 128 + 128 ->64
        x = self.up3(x, x2)
        # 64 + 64 -> 128
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)
