import torch
import torch.nn as nn

class LSTMVoltageModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(LSTMVoltageModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fnn = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        # Input x: [batch_size, 1, sequence_length]
        x = x.permute(0, 2, 1)  # Reshape to [batch_size, sequence_length, 1]
        _, (h_n, _) = self.lstm(x)  # h_n: [num_layers, batch_size, hidden_size]
        out = self.fnn(h_n[-1])  # Last hidden state: [batch_size, hidden_size] -> [batch_size, 1]
        return out


