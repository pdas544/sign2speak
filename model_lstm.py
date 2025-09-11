import torch
import torch.nn as nn

class ASLKeypointLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, num_classes=100, dropout=0.3):
        super(ASLKeypointLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = torch.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
