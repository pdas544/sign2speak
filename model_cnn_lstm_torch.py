import torch
import torch.nn as nn


class CNNKeypointLSTM(nn.Module):
    """
    PyTorch port of the TF CNN+LSTM hybrid (action_recognition_model.py).

    Conv1D stack extracts local temporal features, then a 2-layer LSTM
    models long-range dependencies. Input: (batch, 30, 1629).
    """

    def __init__(self, input_size, hidden_size=256, num_layers=2, num_classes=15, dropout=0.3):
        super(CNNKeypointLSTM, self).__init__()

        # CNN feature extractor (time-major convs over the keypoint axis)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 30 -> 15
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 15 -> 7 (floor)
            nn.Dropout(dropout),
        )

        # LSTM over CNN features
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Classifier
        self.fc1 = nn.Linear(hidden_size, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, time, features) -> conv expects (batch, features, time)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]

        out = torch.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)

        return out
