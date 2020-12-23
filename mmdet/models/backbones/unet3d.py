# based on implementation: https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py

from ..registry import BACKBONES

import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

@BACKBONES.register_module
class UNet3D(nn.Module):

    def __init__(self):
        super(UNet3D, self).__init__()
                
        self.dconv_down1 = double_conv(3, 16)
        self.dconv_down2 = double_conv(16, 32)
        self.dconv_down3 = double_conv(32, 64)
        self.dconv_down4 = double_conv(64, 128)        

        self.maxpool = nn.MaxPool3d(2)
        # self.upsample = nn.functional.interpolate(scale_factor=2, mode='trilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(64 + 128, 64)
        self.dconv_up2 = double_conv(32 + 64, 32)
        self.dconv_up1 = double_conv(32 + 16, 16)
        
        # self.conv_last = nn.Conv2d(64, n_class, 1)
        
    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)       
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)         
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)      
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        return x