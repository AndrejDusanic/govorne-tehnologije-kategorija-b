from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blendshape_project.checkpoint_utils import load_model_bundle, predict_raw_blendshapes  # noqa: E402
from blendshape_project.constants import BLENDSHAPE_NAMES  # noqa: E402
from blendshape_project.data import AudioFeatureExtractor, BlendshapeDataset, collate_batch  # noqa: E402
from blendshape_project.face_refiner import build_face_refiner_features, save_face_refiner  # noqa: E402
from blendshape_project.io_utils import load_json, save_json  # noqa: E402


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def parse_ensemble_weights(raw: str | None, ensemble_size: int) -> list[float]:
    if raw is None:
        return [1.0 / ensemble_size] * ensemble_size
    weights = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(weights) != ensemble_size:
        raise ValueError(f"Expected {ensemble_size} ensemble weights, got {len(weights)}.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Ensemble weights must sum to a positive value.")
    return [weight / total for weight in weights]


def collect_predictions(
    frame: pd.DataFrame,
    bundles: list,
    ensemble_weights: list[float],
    feature_extractor: AudioFeatureExtractor,
    device: torch.device,
    batch_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    loaders = []
    for bundle in bundles:
        dataset = BlendshapeDataset(
            frame,
            feature_extractor,
            stats=None,
            phoneme_vocab=bundle.phoneme_vocab,
            char_vocab=bundle.char_vocab,
            speaker_to_id=bundle.speaker_to_id,
        )
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch))

    features_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []
    weight_tensor = torch.tensor(ensemble_weights, device=device, dtype=torch.float32).view(-1, 1, 1, 1)
    with torch.no_grad():
        for batches in tqdm(zip(*loaders), total=len(loaders[0]), desc="Collecting predictions", leave=False):
            predictions = []
            for bundle, batch in zip(bundles, batches):
                prediction = predict_raw_blendshapes(
                    bundle,
                    features=batch["features"].to(device),
                    speaker_ids=batch["speaker_ids"].to(device),
                    lengths=batch["lengths"].to(device),
                    text_ids=batch["text_ids"].to(device),
                    text_lengths=batch["text_lengths"].to(device),
                )
                predictions.append(prediction)

            ensemble_prediction = (torch.stack(predictions, dim=0) * weight_tensor).sum(dim=0)
            refiner_features = build_face_refiner_features(ensemble_prediction).cpu().numpy()
            targets = batches[0]["targets"].cpu().numpy()
            mask = batches[0]["target_mask"].cpu().numpy()

            for batch_index in range(refiner_features.shape[0]):
                valid_length = int(mask[batch_index].sum())
                features_list.append(refiner_features[batch_index, :valid_length])
                targets_list.append(targets[batch_index, :valid_length])

    return np.concatenate(features_list, axis=0), np.concatenate(targets_list, axis=0)


def evaluate_strength(
    base_prediction: np.ndarray,
    refined_prediction: np.ndarray,
    targets: np.ndarray,
    strength: float,
) -> dict[str, float]:
    blended = np.clip((1.0 - strength) * base_prediction + strength * refined_prediction, 0.0, 1.0)
    mouth_indices = [idx for idx, name in enumerate(BLENDSHAPE_NAMES) if name.startswith(("jaw", "mouth", "tongue"))]
    jaw_index = BLENDSHAPE_NAMES.index("jawOpen")
    return {
        "mae": float(np.mean(np.abs(blended - targets))),
        "rmse": float(np.sqrt(np.mean((blended - targets) ** 2))),
        "mouth_mae": float(np.mean(np.abs(blended[:, mouth_indices] - targets[:, mouth_indices]))),
        "jaw_open_mae": float(np.mean(np.abs(blended[:, jaw_index] - targets[:, jaw_index]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight face-refiner on top of one or more checkpoints.")
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "natural_samples.csv")
    parser.add_argument("--split-json", type=Path, default=ROOT / "data" / "manifests" / "split.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "refiners" / "ensemble_face_refiner_v1.npz")
    parser.add_argument("--metrics-json", type=Path, default=ROOT / "artifacts" / "refiners" / "ensemble_face_refiner_v1_metrics.json")
    parser.add_argument("--ensemble-weights", type=str, default=None)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--ridge-alpha-grid", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--strength-grid", type=str, default="0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    args = parser.parse_args()

    device = select_device(args.device)
    feature_extractor = AudioFeatureExtractor()
    bundles = [load_model_bundle(checkpoint, device=device, feature_dim=feature_extractor.feature_dim) for checkpoint in args.checkpoint]
    ensemble_weights = parse_ensemble_weights(args.ensemble_weights, len(bundles))

    frame = pd.read_csv(args.manifest)
    split = load_json(args.split_json)
    train_df = frame[frame["sample_id"].isin(split["train"])].copy()
    val_df = frame[frame["sample_id"].isin(split["val"])].copy()

    train_x, train_y = collect_predictions(
        train_df,
        bundles,
        ensemble_weights=ensemble_weights,
        feature_extractor=feature_extractor,
        device=device,
        batch_size=args.batch_size,
    )
    val_x, val_y = collect_predictions(
        val_df,
        bundles,
        ensemble_weights=ensemble_weights,
        feature_extractor=feature_extractor,
        device=device,
        batch_size=args.batch_size,
    )

    alpha_grid = (
        [float(item.strip()) for item in args.ridge_alpha_grid.split(",") if item.strip()]
        if args.ridge_alpha_grid
        else [args.ridge_alpha]
    )
    grid = [float(item.strip()) for item in args.strength_grid.split(",") if item.strip()]
    feature_dim = len(BLENDSHAPE_NAMES)
    base_val = val_x[:, :feature_dim]

    best_alpha = alpha_grid[0]
    best_strength = grid[0]
    best_metrics = None
    best_mae = float("inf")
    best_model = None
    best_strength_logs = []
    alpha_logs = []
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(train_x, train_y)
        refined_val = np.clip(model.predict(val_x), 0.0, 1.0)
        strength_logs = []
        alpha_best_metrics = None
        alpha_best_strength = grid[0]
        alpha_best_mae = float("inf")
        for strength in grid:
            metrics = evaluate_strength(base_val, refined_val, val_y, strength)
            strength_logs.append({"strength": strength, **metrics})
            if metrics["mae"] < alpha_best_mae:
                alpha_best_mae = metrics["mae"]
                alpha_best_strength = strength
                alpha_best_metrics = metrics
        alpha_logs.append(
            {
                "ridge_alpha": alpha,
                "best_strength": alpha_best_strength,
                "best_metrics": alpha_best_metrics,
                "strength_grid": strength_logs,
            }
        )
        if alpha_best_mae < best_mae:
            best_mae = alpha_best_mae
            best_alpha = alpha
            best_strength = alpha_best_strength
            best_metrics = alpha_best_metrics
            best_model = model
            best_strength_logs = strength_logs

    save_face_refiner(
        args.output,
        coefficients=best_model.coef_,
        intercept=best_model.intercept_,
        feature_mode="current_delta_square",
        default_strength=best_strength,
        metadata={
            "ridge_alpha": best_alpha,
            "source_checkpoints": [str(path) for path in args.checkpoint],
            "ensemble_weights": ensemble_weights,
            "feature_dim": feature_dim,
            "tuned_on": "validation_split",
        },
    )

    payload = {
        "best_ridge_alpha": best_alpha,
        "best_strength": best_strength,
        "best_metrics": best_metrics,
        "base_metrics": evaluate_strength(base_val, np.clip(best_model.predict(val_x), 0.0, 1.0), val_y, 0.0),
        "fully_refined_metrics": evaluate_strength(base_val, np.clip(best_model.predict(val_x), 0.0, 1.0), val_y, 1.0),
        "strength_grid": best_strength_logs,
        "alpha_grid": alpha_logs,
        "source_checkpoints": [str(path) for path in args.checkpoint],
        "ensemble_weights": ensemble_weights,
    }
    save_json(args.metrics_json, payload)
    print(json.dumps(payload["best_metrics"] | {"best_strength": best_strength}, indent=2))


if __name__ == "__main__":
    main()
