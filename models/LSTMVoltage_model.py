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


""" Sequence Length: Your voltage time series have 8000 samples per sequence, as confirmed by your terminal output (Voltage Shape: [1, 8000] in 
commented debug code). Given a 2µs sampling period (from the MagNet context), this corresponds to 16ms of data:

    For 1kHz waveforms: 16 cycles (1ms per cycle).
    For 5kHz waveforms: 80 cycles (0.2ms per cycle).

Consideration: The LSTM should handle these long sequences, but ensure it’s learning general features across multiple cycles. If training results 
show poor generalization, you might experiment with shorter sequences later.
Waveform Types: Your dataset includes sinusoidal, triangular, and trapezoidal waveforms. To ensure the model doesn’t overfit to one type, you could 
later evaluate performance on each type separately using the test set.
Action: No immediate change needed, but keep this in mind when analyzing results. """