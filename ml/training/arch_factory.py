"""
ml/training/arch_factory.py
───────────────────────────
Single model-construction factory shared by the trainer
(ml.training.train_torch) and serving (TorchBackend).

Every arch maps (batch, seq_len, input_features) -> class logits.
"""

from __future__ import annotations

from ml.training.datasets.video_keypoints import VIDEO_LABELS


ARCHES = ("lstm", "gru", "cnn_lstm", "tcn", "transformer")

ARCH_DEFAULTS = {
    "lstm": {"lr": 1e-3, "hidden_size": 256, "num_layers": 3, "dropout": 0.3},
    "gru": {"lr": 1e-3, "hidden_size": 256, "num_layers": 3, "dropout": 0.3},
    "cnn_lstm": {"lr": 1e-3, "hidden_size": 256, "num_layers": 2, "dropout": 0.3},
    "tcn": {"lr": 1e-3, "hidden_size": 128, "num_layers": 4, "dropout": 0.2},
    "transformer": {"lr": 1e-4, "hidden_size": 256, "num_layers": 3, "dropout": 0.1},
}


def build_model(arch: str, *, hidden_size: int, num_layers: int, dropout: float,
                nhead: int = 8, num_classes: int = 15, input_features: int = 1629,
                labels: list[str] | None = None):
    """Instantiate one comparison arch. All take (batch, T, F) -> logits."""
    if arch == "lstm":
        from model_lstm import ASLKeypointLSTM

        return ASLKeypointLSTM(input_size=input_features, hidden_size=hidden_size,
                               num_layers=num_layers, num_classes=num_classes,
                               dropout=dropout)
    if arch == "gru":
        from model_gru import ASLKeypointGRU

        return ASLKeypointGRU(input_size=input_features, hidden_size=hidden_size,
                              num_layers=num_layers, num_classes=num_classes,
                              dropout=dropout)
    if arch == "cnn_lstm":
        from model_cnn_lstm_torch import CNNKeypointLSTM

        return CNNKeypointLSTM(input_size=input_features, hidden_size=hidden_size,
                               num_layers=num_layers, num_classes=num_classes,
                               dropout=dropout)
    if arch == "tcn":
        from model_tcn import KeypointTCN

        channels = tuple([hidden_size] * num_layers)
        return KeypointTCN(input_size=input_features, num_classes=num_classes,
                           channels=channels, dropout=dropout)
    if arch == "transformer":
        from model_transformer import Config, SignLanguageTransformer

        label_list = list(labels or VIDEO_LABELS)
        config = Config()
        config.input_features = input_features
        config.num_classes = num_classes
        config.d_model = hidden_size
        config.num_encoder_layers = num_layers
        config.nhead = nhead
        config.dropout = dropout
        config.gloss_to_idx = {g: i for i, g in enumerate(label_list)}
        config.idx_to_gloss = {i: g for g, i in config.gloss_to_idx.items()}
        return SignLanguageTransformer(config)
    raise ValueError(f"Unknown arch '{arch}'. Choices: {ARCHES}")
