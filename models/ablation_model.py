import torch
import torch.nn as nn

class VoltageModel(nn.Module):
    def __init__(self):
        super(VoltageModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(16),  # Reduces to fixed length 16
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 16, 8),  # 16 channels * 16 length = 256
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)  # Input: [batch_size, 1, 8000]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 256]
        x = self.fc_layers(x)
        return x