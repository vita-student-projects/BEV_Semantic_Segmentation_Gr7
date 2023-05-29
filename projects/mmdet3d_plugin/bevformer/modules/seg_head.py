import torch
import torch.nn as nn

from torchvision.models.resnet import resnet18
from .builder import SEG_ENCODER
import torch.nn.functional as F
class DoubleConv(nn.Sequential):
    """
    A double convolution block: double (Conv -> BatchNorm2D -> ReLu)
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels

        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    """
    A downsampling block: pooling + double convolution
    """

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    """
    A upsampling block: upsampling + concatenating + double convolution
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels//2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_x):
        # upsampling
        x = self.up(x)

        # in case of unmatched size
        # size = (B, C, H, W)
        diff_height = skip_x.size()[2] - x.size()[2]
        diff_width = skip_x.size()[3] - x.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x = F.pad(x, pad=[diff_width//2, diff_width - diff_width//2, 
                          diff_height//2, diff_height - diff_height//2])
        
        # concatenation
        x = torch.cat([skip_x, x], dim=1)

        x = self.conv(x)

        return x 


#TODO: convert BN to syncBN for multi-gpu training
# model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

@SEG_ENCODER.register_module()
class SegUnet(nn.Module):
    def __init__(self, in_channels, num_classes, bilinear: bool = True, base_channels: int = 64):
        super(SegUnet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # Input conv
        self.in_conv = DoubleConv(in_channels, base_channels)

        # Downsampling
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        # Upsampling
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        
        # Out conv
        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.out_conv(x)

        return x