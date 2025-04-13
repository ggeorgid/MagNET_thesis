import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class TransferModel(nn.Module):
    def __init__(self, use_pretrained=True, freeze_layers=False):
        super(TransferModel, self).__init__()
        # Initialize ResNet18 with pre-trained weights or random weights
        if use_pretrained:
            self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.resnet = resnet18(weights=None)
        
        # No modification to conv1; it remains nn.Conv2d(3, 64, ...)
        
        # Replace fc for regression output (1 output, randomly initialized)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        
        # Freeze layers if specified
        if freeze_layers:
            for name, param in self.resnet.named_parameters():
                # Keep only fc trainable; freeze everything else (including conv1)
                if not name.startswith("fc."):
                    param.requires_grad = False

    def forward(self, x):
        return self.resnet(x)