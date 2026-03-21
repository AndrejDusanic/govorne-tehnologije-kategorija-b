from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .constants import BLENDSHAPE_NAMES, blendshape_priority_weights
from .data import DatasetStats, unnormalize_targets


def lengths_to_mask(lengths: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    max_length = int(max_length or lengths.max().item())
    steps = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return steps < lengths.unsqueeze(1)


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    phoneme_loss_weight: float = 0.2,
    temporal_loss_weight: float = 0.1,
    coefficient_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    predictions = outputs["blendshapes"]
    targets = batch["targets"]
    mask = batch["target_mask"].unsqueeze(-1).float()
    coeffs = coefficient_weights
    if coeffs is None:
        coeffs = torch.tensor(blendshape_priority_weights(), device=predictions.device, dtype=predictions.dtype)
    coeffs = coeffs.view(1, 1, -1)

    reg_residual = (predictions - targets).abs() * coeffs * mask
    reg_denom = (mask.sum() * coeffs.sum()).clamp_min(1.0)
    regression_loss = reg_residual.sum() / reg_denom

    if predictions.shape[1] > 1:
        delta_pred = predictions[:, 1:] - predictions[:, :-1]
        delta_target = targets[:, 1:] - targets[:, :-1]
        delta_mask = (batch["target_mask"][:, 1:] & batch["target_mask"][:, :-1]).unsqueeze(-1).float()
        temporal_residual = (delta_pred - delta_target).abs() * coeffs * delta_mask
        temporal_denom = (delta_mask.sum() * coeffs.sum()).clamp_min(1.0)
        temporal_loss = temporal_residual.sum() / temporal_denom
    else:
        temporal_loss = predictions.new_tensor(0.0)

    phoneme_logits = outputs["phonemes"].reshape(-1, outputs["phonemes"].shape[-1])
    phoneme_targets = batch["phoneme_ids"].reshape(-1)
    phoneme_loss = F.cross_entropy(phoneme_logits, phoneme_targets, ignore_index=-100)

    total_loss = regression_loss + temporal_loss_weight * temporal_loss + phoneme_loss_weight * phoneme_loss
    return {
        "loss": total_loss,
        "regression_loss": regression_loss.detach(),
        "temporal_loss": temporal_loss.detach(),
        "phoneme_loss": phoneme_loss.detach(),
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    stats: DatasetStats,
) -> dict[str, Any]:
    model.eval()
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_points = 0
    per_dim_abs = torch.zeros(len(BLENDSHAPE_NAMES), dtype=torch.float64)
    per_dim_count = 0
    sample_payloads: list[dict[str, Any]] = []

    mouth_indices = [idx for idx, name in enumerate(BLENDSHAPE_NAMES) if name.startswith(("jaw", "mouth", "tongue"))]
    mouth_abs_error = 0.0
    mouth_points = 0

    jaw_index = BLENDSHAPE_NAMES.index("jawOpen")
    jaw_abs_error = 0.0
    jaw_points = 0

    for batch in dataloader:
        features = batch["features"].to(device)
        speaker_ids = batch["speaker_ids"].to(device)
        lengths = batch["lengths"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["target_mask"].to(device)

        outputs = model(features, speaker_ids)
        predictions = outputs["blendshapes"]

        predictions_raw = unnormalize_targets(predictions, stats)
        targets_raw = unnormalize_targets(targets, stats)

        valid = mask.unsqueeze(-1).expand_as(predictions_raw)
        abs_error = (predictions_raw - targets_raw).abs()
        sq_error = (predictions_raw - targets_raw) ** 2

        total_abs_error += abs_error[valid].sum().item()
        total_sq_error += sq_error[valid].sum().item()
        total_points += valid.sum().item()
        per_dim_abs += (abs_error * valid).sum(dim=(0, 1)).cpu().double()
        per_dim_count += int(mask.sum().item())

        mouth_mask = mask.unsqueeze(-1).expand(-1, -1, len(mouth_indices))
        mouth_abs_error += abs_error[:, :, mouth_indices][mouth_mask].sum().item()
        mouth_points += int(mask.sum().item()) * len(mouth_indices)

        jaw_abs_error += abs_error[:, :, jaw_index][mask].sum().item()
        jaw_points += int(mask.sum().item())

        for batch_index, sample_id in enumerate(batch["sample_ids"]):
            valid_length = int(lengths[batch_index].item())
            sample_payloads.append(
                {
                    "sample_id": sample_id,
                    "prediction": predictions_raw[batch_index, :valid_length].cpu().numpy(),
                    "target": targets_raw[batch_index, :valid_length].cpu().numpy(),
                }
            )

    rmse = math.sqrt(total_sq_error / max(total_points, 1))
    per_dim_mae = (per_dim_abs / max(per_dim_count, 1)).numpy()
    return {
        "mae": total_abs_error / max(total_points, 1),
        "rmse": rmse,
        "mouth_mae": mouth_abs_error / max(mouth_points, 1),
        "jaw_open_mae": jaw_abs_error / max(jaw_points, 1),
        "per_dim_mae": per_dim_mae.tolist(),
        "samples": sample_payloads,
    }


def save_history_plot(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_mae = [item["val_mae"] for item in history]
    val_rmse = [item["val_rmse"] for item in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train loss", linewidth=2)
    plt.plot(epochs, val_mae, label="Val MAE", linewidth=2)
    plt.plot(epochs, val_rmse, label="Val RMSE", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title("Training curves")
    plt.grid(alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_per_blendshape_plot(per_dim_mae: list[float], output_path: Path) -> None:
    order = np.argsort(per_dim_mae)[::-1]
    names = [BLENDSHAPE_NAMES[index] for index in order]
    values = [per_dim_mae[index] for index in order]
    plt.figure(figsize=(14, 7))
    plt.bar(range(len(values)), values, color="#2563eb")
    plt.xticks(range(len(values)), names, rotation=80, ha="right")
    plt.ylabel("MAE")
    plt.title("Validation MAE per blendshape")
    plt.grid(axis="y", alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_overlay_plot(
    prediction: np.ndarray,
    target: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    focus_names = ["jawOpen", "mouthClose", "mouthFunnel", "mouthPucker"]
    plt.figure(figsize=(12, 8))
    for plot_index, name in enumerate(focus_names, start=1):
        idx = BLENDSHAPE_NAMES.index(name)
        plt.subplot(len(focus_names), 1, plot_index)
        plt.plot(target[:, idx], label=f"{name} target", linewidth=1.8, color="#111827")
        plt.plot(prediction[:, idx], label=f"{name} pred", linewidth=1.2, color="#dc2626", alpha=0.85)
        plt.ylabel(name)
        plt.grid(alpha=0.25)
        if plot_index == 1:
            plt.title(title)
            plt.legend(loc="upper right")
    plt.xlabel("Frame")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
