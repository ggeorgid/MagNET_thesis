import torch
import torch.nn as nn

class WaveletModel(nn.Module):
    def __init__(self):
        super(WaveletModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # nn.Dropout(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0),-1)
        x = self.fc_layers(x)
        return x

# Explanation why nn.Linerar(256, ...) -> the 256
#Why Assume H=W=64H=W=64?
#
#    Standard Practice:
#        Many datasets (e.g., CIFAR-10, CelebA, or resized image datasets) use square input images 
#        of sizes like 32×3232×32, 64×6464×64, 128×128128×128, etc.
#        64×6464×64 is a reasonable starting point for feature maps in smaller networks like this one.
#
#    No Explicit Input Size Provided:
#        The code snippet doesn’t specify the dimensions of the input image. Since pooling and convolutional 
#        operations reduce spatial dimensions, it’s common to assume a square size like 64×6464×64 unless otherwise stated.
#
#    Flattening to Match 256:
#        The Linear(256, 8) layer implies that the convolutional layers produce a flattened feature size of 256.
#        Using H=W=64H=W=64 and calculating the feature map dimensions step by step matches the flattened size logically.
#
# From ChatGTP Wavelet Model Class Creation Archived Discussion





