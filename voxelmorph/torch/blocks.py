from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Code from Yitong's master thesis: RESNET Feature extractor
def conv3d(
    in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
) -> nn.Module:
    if kernel_size != 1:
        padding = 1
    else:
        padding = 0
    return nn.Conv3d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False,
    )


class ConvBnReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float = 0.05,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bn_track_running_stats: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn_track_running_stats = bn_track_running_stats
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum, track_running_stats = self.bn_track_running_stats)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bn_momentum: float = 0.05, stride: int = 1, 
                bn_track_running_stats: bool = True, no_downsample: bool = False):
        super().__init__()
        self.bn_track_running_stats = bn_track_running_stats
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum, track_running_stats = self.bn_track_running_stats)
        self.dropout1 = nn.Dropout(p=0.2, inplace=True)

        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum, track_running_stats = self.bn_track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.no_downsample = no_downsample

        if not no_downsample:
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm3d(out_channels, momentum=bn_momentum, track_running_stats = self.bn_track_running_stats)
                )
            else:
                self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if not self.no_downsample:
            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
        
        out = self.relu(out)

        return out