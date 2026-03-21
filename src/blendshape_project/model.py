from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.norm(tensor.transpose(1, 2)).transpose(1, 2)


class CausalGatedResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_norm = ChannelLayerNorm(channels)
        self.conv = nn.Conv1d(
            channels,
            channels * 2,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        residual = tensor
        tensor = self.in_norm(tensor)
        padding = (self.kernel_size - 1) * self.dilation
        tensor = F.pad(tensor, (padding, 0))
        tensor = self.conv(tensor)
        filter_tensor, gate_tensor = tensor.chunk(2, dim=1)
        tensor = torch.tanh(filter_tensor) * torch.sigmoid(gate_tensor)
        tensor = self.out_proj(tensor)
        tensor = self.dropout(tensor)
        return residual + tensor


class BlendshapeRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_blendshapes: int,
        num_speakers: int,
        num_phonemes: int,
        hidden_size: int = 256,
        dropout: float = 0.1,
        speaker_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.speaker_embedding = nn.Embedding(num_speakers, speaker_embed_dim)
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim + speaker_embed_dim, hidden_size, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        dilations = [1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32]
        self.blocks = nn.ModuleList(
            [CausalGatedResidualBlock(hidden_size, dilation=dilation, dropout=dropout) for dilation in dilations]
        )
        self.final_norm = ChannelLayerNorm(hidden_size)
        self.regression_head = nn.Conv1d(hidden_size, num_blendshapes, kernel_size=1)
        self.phoneme_head = nn.Conv1d(hidden_size, num_phonemes, kernel_size=1)

    def forward(
        self,
        features: torch.Tensor,
        speaker_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tensor = features.transpose(1, 2)
        speaker_embedding = self.speaker_embedding(speaker_ids).unsqueeze(-1).expand(-1, -1, tensor.shape[-1])
        tensor = self.input_proj(torch.cat([tensor, speaker_embedding], dim=1))
        for block in self.blocks:
            tensor = block(tensor)
        tensor = self.final_norm(tensor)
        return {
            "blendshapes": self.regression_head(tensor).transpose(1, 2),
            "phonemes": self.phoneme_head(tensor).transpose(1, 2),
        }

