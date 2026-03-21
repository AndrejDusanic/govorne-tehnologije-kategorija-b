from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .constants import SPEAKER_ORDER
from .data import AudioFeatureExtractor, DatasetStats, unnormalize_targets
from .model import BlendshapeRegressor


@dataclass
class ModelBundle:
    checkpoint_path: Path
    model: BlendshapeRegressor
    stats: DatasetStats
    speaker_to_id: dict[str, int]
    phoneme_vocab: dict[str, int]
    char_vocab: dict[str, int]
    blendshape_names: list[str]
    config: dict[str, Any]
    feature_mean: torch.Tensor
    feature_std: torch.Tensor


def load_model_bundle(
    checkpoint_path: str | Path,
    device: torch.device,
    feature_dim: int | None = None,
) -> ModelBundle:
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    stats = DatasetStats.from_json(checkpoint["stats"])
    speaker_to_id = checkpoint.get("speaker_to_id", {speaker: idx for idx, speaker in enumerate(SPEAKER_ORDER)})
    phoneme_vocab = checkpoint.get("phoneme_vocab", {"<pad>": 0, "<unk>": 1})
    char_vocab = checkpoint.get("char_vocab", {"<pad>": 0, "<unk>": 1})
    config = checkpoint.get("config", {})

    model = BlendshapeRegressor(
        input_dim=feature_dim or AudioFeatureExtractor().feature_dim,
        num_blendshapes=len(checkpoint["blendshape_names"]),
        num_speakers=len(speaker_to_id),
        num_phonemes=len(phoneme_vocab),
        num_chars=len(char_vocab),
        hidden_size=config.get("hidden_size", 256),
        dropout=config.get("dropout", 0.12),
        char_embed_dim=config.get("char_embed_dim", 64),
        text_hidden_size=config.get("text_hidden_size", 128),
        use_text_conditioning=config.get("use_text_conditioning", False),
        temporal_encoder=config.get("temporal_encoder", "causal_tcn"),
        num_attention_heads=config.get("num_attention_heads", 4),
        num_gru_layers=config.get("num_gru_layers", 2),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    return ModelBundle(
        checkpoint_path=path,
        model=model,
        stats=stats,
        speaker_to_id=speaker_to_id,
        phoneme_vocab=phoneme_vocab,
        char_vocab=char_vocab,
        blendshape_names=checkpoint["blendshape_names"],
        config=config,
        feature_mean=torch.tensor(stats.feature_mean, dtype=torch.float32, device=device),
        feature_std=torch.tensor(stats.feature_std, dtype=torch.float32, device=device),
    )


def predict_raw_blendshapes(
    bundle: ModelBundle,
    features: torch.Tensor,
    speaker_ids: torch.Tensor,
    lengths: torch.Tensor,
    text_ids: torch.Tensor,
    text_lengths: torch.Tensor,
) -> torch.Tensor:
    normalized_features = (features - bundle.feature_mean) / bundle.feature_std
    outputs = bundle.model(
        normalized_features,
        speaker_ids,
        lengths=lengths,
        text_ids=text_ids,
        text_lengths=text_lengths,
    )
    return unnormalize_targets(outputs["blendshapes"], bundle.stats)
