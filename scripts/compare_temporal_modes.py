from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate import evaluate_bundles, parse_ensemble_weights, select_device  # noqa: E402
from train_face_refiner import collect_predictions  # noqa: E402

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader

from blendshape_project.checkpoint_utils import load_model_bundle  # noqa: E402
from blendshape_project.data import AudioFeatureExtractor, BlendshapeDataset, collate_batch  # noqa: E402
from blendshape_project.face_refiner import load_face_refiner, save_face_refiner  # noqa: E402
from blendshape_project.io_utils import load_json, save_json  # noqa: E402


def ensure_single_checkpoint_refiner(
    checkpoint: Path,
    manifest: Path,
    split_json: Path,
    device: torch.device,
    feature_extractor: AudioFeatureExtractor,
    output_dir: Path,
    batch_size: int,
    ridge_alpha: float,
    strength_grid: list[float],
) -> tuple[Path, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    refiner_path = output_dir / f"{checkpoint.parent.name}_solo_refiner.npz"
    metrics_path = output_dir / f"{checkpoint.parent.name}_solo_refiner_metrics.json"
    if refiner_path.exists() and metrics_path.exists():
        return refiner_path, json.loads(metrics_path.read_text(encoding="utf-8"))

    bundle = load_model_bundle(checkpoint, device=device, feature_dim=feature_extractor.feature_dim)
    frame = pd.read_csv(manifest)
    split = load_json(split_json)
    train_df = frame[frame["sample_id"].isin(split["train"])].copy()
    val_df = frame[frame["sample_id"].isin(split["val"])].copy()
    train_x, train_y = collect_predictions(
        train_df,
        [bundle],
        ensemble_weights=[1.0],
        feature_extractor=feature_extractor,
        device=device,
        batch_size=batch_size,
    )
    val_x, val_y = collect_predictions(
        val_df,
        [bundle],
        ensemble_weights=[1.0],
        feature_extractor=feature_extractor,
        device=device,
        batch_size=batch_size,
    )

    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    model.fit(train_x, train_y)
    base_val = val_x[:, : train_y.shape[1]]
    refined_val = np.clip(model.predict(val_x), 0.0, 1.0)

    best_strength = strength_grid[0]
    best_metrics = None
    best_mae = float("inf")
    strength_logs = []
    mouth_indices = None
    jaw_index = None
    from blendshape_project.constants import BLENDSHAPE_NAMES  # local import keeps script focused

    mouth_indices = [idx for idx, name in enumerate(BLENDSHAPE_NAMES) if name.startswith(("jaw", "mouth", "tongue"))]
    jaw_index = BLENDSHAPE_NAMES.index("jawOpen")

    for strength in strength_grid:
        blended = np.clip((1.0 - strength) * base_val + strength * refined_val, 0.0, 1.0)
        metrics = {
            "mae": float(np.mean(np.abs(blended - val_y))),
            "rmse": float(np.sqrt(np.mean((blended - val_y) ** 2))),
            "mouth_mae": float(np.mean(np.abs(blended[:, mouth_indices] - val_y[:, mouth_indices]))),
            "jaw_open_mae": float(np.mean(np.abs(blended[:, jaw_index] - val_y[:, jaw_index]))),
        }
        strength_logs.append({"strength": strength, **metrics})
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_strength = strength
            best_metrics = metrics

    save_face_refiner(
        refiner_path,
        coefficients=model.coef_,
        intercept=model.intercept_,
        feature_mode="current_delta_square",
        default_strength=best_strength,
        metadata={
            "ridge_alpha": ridge_alpha,
            "source_checkpoints": [str(checkpoint)],
            "ensemble_weights": [1.0],
            "feature_dim": train_y.shape[1],
            "tuned_on": "validation_split",
        },
    )
    payload = {
        "checkpoint": str(checkpoint.relative_to(ROOT) if checkpoint.is_absolute() else checkpoint),
        "best_strength": best_strength,
        "best_metrics": best_metrics,
        "strength_grid": strength_logs,
    }
    save_json(metrics_path, payload)
    return refiner_path, payload


def build_loader(bundle, frame: pd.DataFrame, feature_extractor: AudioFeatureExtractor, batch_size: int) -> DataLoader:
    dataset = BlendshapeDataset(
        frame,
        feature_extractor,
        stats=None,
        phoneme_vocab=bundle.phoneme_vocab,
        char_vocab=bundle.char_vocab,
        speaker_to_id=bundle.speaker_to_id,
        aux_target_type=bundle.config.get("aux_target_type", "phoneme"),
        viseme_variant=bundle.config.get("viseme_variant", "viseme_balanced_10"),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare single-checkpoint causal TCN + refiner vs offline BiGRU + refiner.")
    parser.add_argument("--causal-checkpoint", type=Path, required=True)
    parser.add_argument("--offline-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "natural_samples.csv")
    parser.add_argument("--split-json", type=Path, default=ROOT / "data" / "manifests" / "split.json")
    parser.add_argument("--output-json", type=Path, default=ROOT / "reports" / "figures" / "temporal_mode_comparison.json")
    parser.add_argument("--refiner-dir", type=Path, default=ROOT / "artifacts" / "refiners" / "solo_modes")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--strength-grid", type=str, default="0.15,0.20,0.25,0.30,0.35")
    args = parser.parse_args()

    device = select_device(args.device)
    feature_extractor = AudioFeatureExtractor()
    strength_grid = [float(item.strip()) for item in args.strength_grid.split(",") if item.strip()]

    causal_refiner_path, causal_refiner_meta = ensure_single_checkpoint_refiner(
        args.causal_checkpoint,
        manifest=args.manifest,
        split_json=args.split_json,
        device=device,
        feature_extractor=feature_extractor,
        output_dir=args.refiner_dir,
        batch_size=args.batch_size,
        ridge_alpha=args.ridge_alpha,
        strength_grid=strength_grid,
    )
    offline_refiner_path, offline_refiner_meta = ensure_single_checkpoint_refiner(
        args.offline_checkpoint,
        manifest=args.manifest,
        split_json=args.split_json,
        device=device,
        feature_extractor=feature_extractor,
        output_dir=args.refiner_dir,
        batch_size=args.batch_size,
        ridge_alpha=args.ridge_alpha,
        strength_grid=strength_grid,
    )

    frame = pd.read_csv(args.manifest)
    split = load_json(args.split_json)
    val_df = frame[frame["sample_id"].isin(split["val"])].copy()

    causal_bundle = load_model_bundle(args.causal_checkpoint, device=device, feature_dim=feature_extractor.feature_dim)
    offline_bundle = load_model_bundle(args.offline_checkpoint, device=device, feature_dim=feature_extractor.feature_dim)
    causal_loader = build_loader(causal_bundle, val_df, feature_extractor, args.batch_size)
    offline_loader = build_loader(offline_bundle, val_df, feature_extractor, args.batch_size)

    causal_metrics = evaluate_bundles(
        [causal_bundle],
        [causal_loader],
        device,
        ensemble_weights=[1.0],
        face_refiner=load_face_refiner(causal_refiner_path, device=device),
    )
    offline_metrics = evaluate_bundles(
        [offline_bundle],
        [offline_loader],
        device,
        ensemble_weights=[1.0],
        face_refiner=load_face_refiner(offline_refiner_path, device=device),
    )

    payload = {
        "causal": {
            "checkpoint": str(args.causal_checkpoint.relative_to(ROOT) if args.causal_checkpoint.is_absolute() else args.causal_checkpoint),
            "temporal_encoder": causal_bundle.config.get("temporal_encoder", "causal_tcn"),
            "use_speaker_embedding": causal_bundle.config.get("use_speaker_embedding", True),
            "aux_target_type": causal_bundle.config.get("aux_target_type", "phoneme"),
            "viseme_variant": causal_bundle.config.get("viseme_variant"),
            "refiner": str(causal_refiner_path.relative_to(ROOT)),
            "refiner_meta": causal_refiner_meta,
            "metrics": {key: causal_metrics[key] for key in ("mae", "rmse", "mouth_mae", "jaw_open_mae")},
        },
        "offline": {
            "checkpoint": str(args.offline_checkpoint.relative_to(ROOT) if args.offline_checkpoint.is_absolute() else args.offline_checkpoint),
            "temporal_encoder": offline_bundle.config.get("temporal_encoder", "causal_tcn"),
            "use_speaker_embedding": offline_bundle.config.get("use_speaker_embedding", True),
            "aux_target_type": offline_bundle.config.get("aux_target_type", "phoneme"),
            "viseme_variant": offline_bundle.config.get("viseme_variant"),
            "refiner": str(offline_refiner_path.relative_to(ROOT)),
            "refiner_meta": offline_refiner_meta,
            "metrics": {key: offline_metrics[key] for key in ("mae", "rmse", "mouth_mae", "jaw_open_mae")},
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output_json, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
