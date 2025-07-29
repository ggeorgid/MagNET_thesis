import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class TransferModel(nn.Module):
    def __init__(self, use_pretrained=True, fine_tune_layers=["fc"]):
        super(TransferModel, self).__init__()

        # Print for debugging
        print(f"🔹 USE_PRETRAINED: {use_pretrained}")

        # Initialize ResNet18
        if use_pretrained:
            self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.resnet = resnet18(weights=None)  # Random weights

        # Replace fc layer for regression (1 output)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

        if use_pretrained:
            # Freeze all layers initially
            for param in self.resnet.parameters():
                param.requires_grad = False

            # Unfreeze specified layers
            for name, param in self.resnet.named_parameters():
                if any(name.startswith(layer + ".") for layer in fine_tune_layers):
                    print(f"Unfreezing: {name}")
                    param.requires_grad = True
        else:
            # Ensure all layers are trainable for training from scratch
            for param in self.resnet.parameters():
                param.requires_grad = True
            print("🔹 Training all layers from scratch.")

    def forward(self, x):
        return self.resnet(x)

