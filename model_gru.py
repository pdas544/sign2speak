import torch
import torch.nn as nn


class ASLKeypointGRU(nn.Module):
    """Bidirectional GRU baseline — same harness as ASLKeypointLSTM (model_lstm.py)."""

    def __init__(self, input_size, hidden_size=256, num_layers=3, num_classes=15, dropout=0.3):
        super(ASLKeypointGRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # GRU forward pass
        gru_out, _ = self.gru(x)

        # Take the last time step output
        last_output = gru_out[:, -1, :]

        # Fully connected layers
        out = torch.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)

        return out
