from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class FaceRefiner:
    coefficients: torch.Tensor
    intercept: torch.Tensor
    feature_mode: str
    default_strength: float
    metadata: dict[str, Any]


def build_face_refiner_features(values: torch.Tensor) -> torch.Tensor:
    if values.dim() != 3:
        raise ValueError(f"Expected [batch, time, dim] tensor, got shape {tuple(values.shape)}")
    delta = torch.cat([torch.zeros_like(values[:, :1]), values[:, 1:] - values[:, :-1]], dim=1)
    return torch.cat([values, delta, values.square()], dim=-1)


def apply_face_refiner(
    values: torch.Tensor,
    refiner: FaceRefiner,
    strength: float | None = None,
    clamp: bool = True,
) -> torch.Tensor:
    strength = refiner.default_strength if strength is None else strength
    features = build_face_refiner_features(values)
    refined = F.linear(
        features,
        refiner.coefficients.to(device=values.device, dtype=values.dtype),
        refiner.intercept.to(device=values.device, dtype=values.dtype),
    )
    blended = torch.lerp(values, refined, weight=float(strength))
    return blended.clamp(0.0, 1.0) if clamp else blended


def save_face_refiner(
    path: str | Path,
    coefficients: np.ndarray,
    intercept: np.ndarray,
    feature_mode: str,
    default_strength: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_mode": feature_mode,
        "default_strength": float(default_strength),
        "metadata": metadata or {},
    }
    np.savez_compressed(
        output_path,
        coefficients=coefficients.astype(np.float32),
        intercept=intercept.astype(np.float32),
        payload_json=json.dumps(payload, ensure_ascii=False),
    )


def load_face_refiner(path: str | Path, device: torch.device | None = None) -> FaceRefiner:
    archive = np.load(Path(path), allow_pickle=False)
    payload = json.loads(str(archive["payload_json"]))
    coefficients = torch.tensor(archive["coefficients"], dtype=torch.float32, device=device)
    intercept = torch.tensor(archive["intercept"], dtype=torch.float32, device=device)
    return FaceRefiner(
        coefficients=coefficients,
        intercept=intercept,
        feature_mode=payload.get("feature_mode", "current_delta_square"),
        default_strength=float(payload.get("default_strength", 1.0)),
        metadata=payload.get("metadata", {}),
    )
