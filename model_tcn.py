import torch
import torch.nn as nn


class _Chomp1d(nn.Module):
    """Trim right padding so convolutions stay causal."""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        if self.chomp == 0:
            return x
        return x[:, :, :-self.chomp]


class _TemporalBlock(nn.Module):
    """Dilated causal conv block with residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class KeypointTCN(nn.Module):
    """
    Temporal Convolutional Network for keypoint sequences.

    Dilated causal convolutions give exponential receptive-field growth with
    depth while staying fully parallel (fast inference). Input: (batch, 30, F).
    """

    def __init__(self, input_size, num_classes=15, channels=(128, 128, 128, 128),
                 kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        in_ch = input_size
        for i, out_ch in enumerate(channels):
            layers.append(_TemporalBlock(in_ch, out_ch, kernel_size,
                                         dilation=2 ** i, dropout=dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        # x: (batch, time, features) -> conv expects (batch, features, time)
        x = x.transpose(1, 2)
        features = self.network(x)            # (batch, channels, time)
        pooled = features.mean(dim=2)         # temporal average pooling
        return self.classifier(pooled)
