import torch
import torch.nn as nn

# Original implementation from ConfidNet:
# https://github.com/valeoai/ConfidNet/blob/master/confidnet/models/small_convnet_svhn.py 
class Conv2dSame(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d
    ):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias),
        )

    def forward(self, x):
        return self.net(x)
