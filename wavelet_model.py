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
# From ChatGPT Wavelet Model Class Creation Archived Discussion
#
# From ChatGPT on potenital issues with the model:

# 2. Review of wavelet_model.py (Potential Issues/Optimizations)

# Here are some observations:

#     Padding Issue: You're using padding=2 with kernel_size=3. 
#          This causes the feature maps to expand slightly, which might be unintentional. 
#          Typically, padding = 1 keeps the dimensions stable with 3x3 kernels.
#     Dropout Layer (Commented Out): Adding a dropout layer (after activations in the fully connected layers) could help with regularization, 
#           especially if you see overfitting.
#     Adaptive Pooling: Consider nn.AdaptiveAvgPool2d((1, 1)) instead of fixed AvgPool2d. 
#           This allows flexibility if the input size changes, making the model more robust.
#     Flattening Method: Using x.view(x.size(0), -1) is fine, but nn.Flatten() is cleaner 
#           and avoids potential shape mismatch issues.

# I’m not suggesting changes now, but if you encounter performance issues, these could be areas to revisit.








