from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blendshape_project.checkpoint_utils import load_model_bundle, predict_raw_blendshapes  # noqa: E402
from blendshape_project.constants import BLENDSHAPE_NAMES  # noqa: E402
from blendshape_project.data import AudioFeatureExtractor, BlendshapeDataset, collate_batch  # noqa: E402
from blendshape_project.io_utils import load_json, save_json  # noqa: E402


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def evaluate_weight_grid(
    bundles: list,
    loaders: list[DataLoader],
    device: torch.device,
    weight_grid: list[float],
) -> list[dict[str, float]]:
    if len(bundles) != 2:
        raise ValueError("Weight search currently supports exactly two checkpoints.")

    mouth_indices = [idx for idx, name in enumerate(BLENDSHAPE_NAMES) if name.startswith(("jaw", "mouth", "tongue"))]
    jaw_index = BLENDSHAPE_NAMES.index("jawOpen")
    stats = {
        weight_a: {
            "abs_error": 0.0,
            "sq_error": 0.0,
            "points": 0,
            "mouth_abs_error": 0.0,
            "mouth_points": 0,
            "jaw_abs_error": 0.0,
            "jaw_points": 0,
        }
        for weight_a in weight_grid
    }

    with torch.no_grad():
        for batch_a, batch_b in zip(*loaders):
            pred_a = predict_raw_blendshapes(
                bundles[0],
                features=batch_a["features"].to(device),
                speaker_ids=batch_a["speaker_ids"].to(device),
                lengths=batch_a["lengths"].to(device),
                text_ids=batch_a["text_ids"].to(device),
                text_lengths=batch_a["text_lengths"].to(device),
            )
            pred_b = predict_raw_blendshapes(
                bundles[1],
                features=batch_b["features"].to(device),
                speaker_ids=batch_b["speaker_ids"].to(device),
                lengths=batch_b["lengths"].to(device),
                text_ids=batch_b["text_ids"].to(device),
                text_lengths=batch_b["text_lengths"].to(device),
            )
            targets = batch_a["targets"].to(device)
            mask = batch_a["target_mask"].to(device)
            valid = mask.unsqueeze(-1).expand_as(pred_a)
            mouth_mask = mask.unsqueeze(-1).expand(-1, -1, len(mouth_indices))

            for weight_a in weight_grid:
                prediction = pred_a * weight_a + pred_b * (1.0 - weight_a)
                abs_error = (prediction - targets).abs()
                sq_error = (prediction - targets) ** 2
                bucket = stats[weight_a]
                bucket["abs_error"] += abs_error[valid].sum().item()
                bucket["sq_error"] += sq_error[valid].sum().item()
                bucket["points"] += int(valid.sum().item())
                bucket["mouth_abs_error"] += abs_error[:, :, mouth_indices][mouth_mask].sum().item()
                bucket["mouth_points"] += int(mask.sum().item()) * len(mouth_indices)
                bucket["jaw_abs_error"] += abs_error[:, :, jaw_index][mask].sum().item()
                bucket["jaw_points"] += int(mask.sum().item())

    results = []
    for weight_a in weight_grid:
        bucket = stats[weight_a]
        points = max(bucket["points"], 1)
        mouth_points = max(bucket["mouth_points"], 1)
        jaw_points = max(bucket["jaw_points"], 1)
        results.append(
            {
                "weight_a": weight_a,
                "weight_b": 1.0 - weight_a,
                "mae": bucket["abs_error"] / points,
                "rmse": math.sqrt(bucket["sq_error"] / points),
                "mouth_mae": bucket["mouth_abs_error"] / mouth_points,
                "jaw_open_mae": bucket["jaw_abs_error"] / jaw_points,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Search scalar ensemble weights for exactly two checkpoints.")
    parser.add_argument("--checkpoint", type=Path, nargs=2, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "natural_samples.csv")
    parser.add_argument("--split-json", type=Path, default=ROOT / "data" / "manifests" / "split.json")
    parser.add_argument("--weight-grid", type=str, default="0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    device = select_device(args.device)
    feature_extractor = AudioFeatureExtractor()
    bundles = [load_model_bundle(checkpoint, device=device, feature_dim=feature_extractor.feature_dim) for checkpoint in args.checkpoint]

    frame = pd.read_csv(args.manifest)
    split = load_json(args.split_json)
    val_df = frame[frame["sample_id"].isin(split["val"])].copy()

    loaders = []
    for bundle in bundles:
        dataset = BlendshapeDataset(
            val_df,
            feature_extractor,
            stats=None,
            phoneme_vocab=bundle.phoneme_vocab,
            char_vocab=bundle.char_vocab,
            speaker_to_id=bundle.speaker_to_id,
        )
        loaders.append(DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_batch))

    weight_grid = [float(item.strip()) for item in args.weight_grid.split(",") if item.strip()]
    results = evaluate_weight_grid(bundles, loaders, device, weight_grid)
    best = min(results, key=lambda item: item["mae"])
    payload = {
        "checkpoints": [str(path) for path in args.checkpoint],
        "results": results,
        "best": best,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.output_json, payload)
    print(payload["best"])


if __name__ == "__main__":
    main()
