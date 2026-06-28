import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DenseNetBiLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        densenet = models.densenet121(weights=None)
        self.backbone = densenet.features
        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, channels, height, width)
        batch_size, seq_len, _, h, w = x.shape
        x = x.view(batch_size * seq_len, 3, h, w)

        features = self.backbone(x)
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1))

        features = torch.flatten(features, 1)

        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)

        # Use mean pooling over temporal frames.
        pooled = torch.mean(lstm_out, dim=1)
        logits = self.classifier(pooled)
        return logits
