from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        num_chars: int = 0,
        hidden_size: int = 256,
        dropout: float = 0.1,
        speaker_embed_dim: int = 16,
        char_embed_dim: int = 64,
        text_hidden_size: int | None = None,
        use_text_conditioning: bool = False,
        temporal_encoder: str = "causal_tcn",
        num_attention_heads: int = 4,
        num_gru_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.temporal_encoder = temporal_encoder
        self.use_text_conditioning = use_text_conditioning and num_chars > 0
        self.speaker_embedding = nn.Embedding(num_speakers, speaker_embed_dim)
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim + speaker_embed_dim, hidden_size, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.use_text_conditioning:
            text_hidden_size = text_hidden_size or hidden_size // 2
            self.char_embedding = nn.Embedding(num_chars, char_embed_dim, padding_idx=0)
            self.text_encoder = nn.GRU(
                input_size=char_embed_dim,
                hidden_size=text_hidden_size,
                batch_first=True,
                bidirectional=True,
            )
            self.text_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.text_fusion = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        if temporal_encoder == "causal_tcn":
            dilations = [1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32]
            self.blocks = nn.ModuleList(
                [CausalGatedResidualBlock(hidden_size, dilation=dilation, dropout=dropout) for dilation in dilations]
            )
            self.final_norm = ChannelLayerNorm(hidden_size)
            self.regression_head = nn.Conv1d(hidden_size, num_blendshapes, kernel_size=1)
            self.phoneme_head = nn.Conv1d(hidden_size, num_phonemes, kernel_size=1)
        elif temporal_encoder == "bgru":
            gru_hidden_size = hidden_size // 2
            self.sequence_norm = nn.LayerNorm(hidden_size)
            self.temporal_gru = nn.GRU(
                input_size=hidden_size,
                hidden_size=gru_hidden_size,
                num_layers=num_gru_layers,
                batch_first=True,
                dropout=dropout if num_gru_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.sequence_ffn = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Dropout(dropout),
            )
            self.sequence_regression_head = nn.Linear(hidden_size, num_blendshapes)
            self.sequence_phoneme_head = nn.Linear(hidden_size, num_phonemes)
        else:
            raise ValueError(f"Unsupported temporal encoder: {temporal_encoder}")

    def _apply_text_conditioning(
        self,
        tensor: torch.Tensor,
        text_ids: torch.Tensor | None,
        text_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.use_text_conditioning or text_ids is None or text_lengths is None:
            return tensor

        embedded = self.char_embedding(text_ids)
        packed = pack_padded_sequence(
            embedded,
            text_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_states, _ = self.text_encoder(packed)
        text_states, _ = pad_packed_sequence(packed_states, batch_first=True, total_length=text_ids.shape[1])
        key_padding_mask = text_ids.eq(0)
        attended, _ = self.text_attention(
            query=tensor,
            key=text_states,
            value=text_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return tensor + self.text_fusion(torch.cat([tensor, attended], dim=-1))

    def forward(
        self,
        features: torch.Tensor,
        speaker_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        text_ids: torch.Tensor | None = None,
        text_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        tensor = features.transpose(1, 2)
        speaker_embedding = self.speaker_embedding(speaker_ids).unsqueeze(-1).expand(-1, -1, tensor.shape[-1])
        tensor = self.input_proj(torch.cat([tensor, speaker_embedding], dim=1))
        sequence = tensor.transpose(1, 2)
        sequence = self._apply_text_conditioning(sequence, text_ids=text_ids, text_lengths=text_lengths)

        if self.temporal_encoder == "causal_tcn":
            tensor = sequence.transpose(1, 2)
            for block in self.blocks:
                tensor = block(tensor)
            tensor = self.final_norm(tensor)
            return {
                "blendshapes": self.regression_head(tensor).transpose(1, 2),
                "phonemes": self.phoneme_head(tensor).transpose(1, 2),
            }

        if lengths is None:
            lengths = torch.full(
                (sequence.shape[0],),
                fill_value=sequence.shape[1],
                dtype=torch.long,
                device=sequence.device,
            )
        packed = pack_padded_sequence(sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.temporal_gru(packed)
        sequence_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=sequence.shape[1])
        sequence_output = sequence_output + self.sequence_ffn(sequence_output)
        return {
            "blendshapes": self.sequence_regression_head(sequence_output),
            "phonemes": self.sequence_phoneme_head(sequence_output),
        }
