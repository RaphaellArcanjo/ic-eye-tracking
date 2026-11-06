"""Definição da arquitetura neural multimodal BenabderrahmaneNet em PyTorch."""

from typing import Optional

import torch
import torch.nn as nn


class BenabderrahmaneNet(nn.Module):
    """Rede Neural Multimodal em PyTorch baseada em Benabderrahmane et al. (2024).

    Integra quatro ramos especializados com fusão tardia (concatenação):
    1. Ramo CNN 2D para mapas espaciais de scan-path `(1, 400, 600)`.
    2. Ramo GRU 1 para sequências de distâncias euclidianas com `AvgPool1d(kernel_size=5, stride=5)`.
    3. Ramo GRU 2 para sequências de durações de eventos oculares (fixações e sacadas).
    4. Ramo MLP para variáveis escalares (`Fixation Rate` e `Stimulus ID`).
    """

    def __init__(
        self,
        cnn_channels: Optional[list[int]] = None,
        cnn_latent_dim: int = 128,
        gru_hidden_size: int = 64,
        gru_num_layers: int = 2,
        bidirectional: bool = False,
        mlp_latent_dim: int = 16,
        fusion_hidden_size: int = 128,
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [8, 16, 32, 64]

        self.bidirectional = bidirectional
        self.gru_hidden_size = gru_hidden_size

        # a) Ramo CNN 2D (Espacial: Mapa de Scan-path 2D)
        cnn_layers: list[nn.Module] = []
        in_c = 1
        for out_c in cnn_channels:
            cnn_layers.extend([
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
            in_c = out_c

        cnn_layers.extend([
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(cnn_channels[-1] * 4 * 4, cnn_latent_dim),
            nn.BatchNorm1d(cnn_latent_dim),
            nn.ReLU(inplace=True),
        ])
        self.cnn_branch = nn.Sequential(*cnn_layers)
        self.cnn_latent_dim = cnn_latent_dim

        # b) Ramo GRU 1 (Distâncias: AvgPool1d kernel=5, stride=5 -> 2.074 timesteps)
        self.pool_distances = nn.AvgPool1d(kernel_size=5, stride=5)
        self.gru_distances = nn.GRU(
            input_size=1,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.gru_dist_out_dim = gru_hidden_size * (2 if bidirectional else 1)

        # c) Ramo GRU 2 (Durações de Eventos: 1.313 timesteps)
        self.gru_durations = nn.GRU(
            input_size=1,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.gru_dur_out_dim = gru_hidden_size * (2 if bidirectional else 1)

        # d) Ramo MLP (Escalares: Fixation Rate e Stimulus ID)
        self.mlp_scalars = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, mlp_latent_dim),
            nn.ReLU(inplace=True),
        )
        self.mlp_latent_dim = mlp_latent_dim

        # e) Fusão Tardia e Classificador MLP
        fused_dim = (
            self.cnn_latent_dim
            + self.gru_dist_out_dim
            + self.gru_dur_out_dim
            + self.mlp_latent_dim
        )
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        scanpath: torch.Tensor,
        distances: torch.Tensor,
        durations: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        """Executa o forward pass multimodal integrado com downsampling temporal 1D."""
        # a) Ramo CNN: (B, 1, 400, 600) -> (B, 128)
        feat_cnn = self.cnn_branch(scanpath)

        # b) Ramo GRU 1 (Distâncias): Downsampling 1D de 10.370 para 2.074 timesteps
        if distances.dim() == 2:
            distances = distances.unsqueeze(1)
        elif distances.dim() == 3 and distances.size(-1) == 1:
            distances = distances.transpose(1, 2)
        dist_pooled = self.pool_distances(distances).transpose(1, 2)
        _, h_dist = self.gru_distances(dist_pooled)
        if self.bidirectional:
            feat_dist = torch.cat([h_dist[-2], h_dist[-1]], dim=-1)
        else:
            feat_dist = h_dist[-1]

        # c) Ramo GRU 2 (Durações): (B, 1313, 1)
        if durations.dim() == 2:
            durations = durations.unsqueeze(-1)
        _, h_dur = self.gru_durations(durations)
        if self.bidirectional:
            feat_dur = torch.cat([h_dur[-2], h_dur[-1]], dim=-1)
        else:
            feat_dur = h_dur[-1]

        # d) Ramo MLP (Escalares): (B, 2) -> (B, 16)
        feat_scalars = self.mlp_scalars(scalars)

        # e) Fusão Tardia e Classificação
        fused = torch.cat([feat_cnn, feat_dist, feat_dur, feat_scalars], dim=1)
        return self.classifier(fused)
