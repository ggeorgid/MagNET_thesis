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

# import torch.nn as nn
# from torchvision.models import resnet18, ResNet18_Weights

# # New transfer model with fc and layer4 unfrozen 
# class TransferModel(nn.Module):
#     def __init__(self, use_pretrained=True, fine_tune_layers=["fc"]):
#         super(TransferModel, self).__init__()

#         # Print the value of use_pretrained
#         print(f"🔹 [DEBUG] USE_PRETRAINED in TransferModel: {use_pretrained}")

#         # Initialize ResNet18 with pre-trained weights or random weights
#         if use_pretrained:
#             self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
#         else:
#             self.resnet = resnet18(weights=None)
        
#         # No modification to conv1; it remains nn.Conv2d(3, 64, ...)
        
#         # Replace fc for regression output (1 output, randomly initialized)
#         self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        
#         # Freeze all layers initially
#         for param in self.resnet.parameters():
#             param.requires_grad = False
        
#         # Unfreeze specified layers
#         for name, param in self.resnet.named_parameters():
#             if any(name.startswith(layer + ".") for layer in fine_tune_layers):
#                 print(f"Unfreezing: {name}")
#                 param.requires_grad = True

#     def forward(self, x):
#         return self.resnet(x)


# import torch.nn as nn
# from torchvision.models import resnet18, ResNet18_Weights

# # original transfer model with everything frozen
# class TransferModel(nn.Module):
#     def __init__(self, use_pretrained=True, freeze_layers=False):
#         super(TransferModel, self).__init__()
#         # Initialize ResNet18 with pre-trained weights or random weights
#         if use_pretrained:
#             self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
#         else:
#             self.resnet = resnet18(weights=None)
        
#         # No modification to conv1; it remains nn.Conv2d(3, 64, ...)
        
#         # Replace fc for regression output (1 output, randomly initialized)
#         self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        
#         # Freeze layers if specified
#         if freeze_layers:
#             for name, param in self.resnet.named_parameters():
#                 # Keep only fc trainable; freeze everything else (including conv1)
#                 if not name.startswith("fc."):
#                     param.requires_grad = False

#     def forward(self, x):
#         return self.resnet(x)