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
from blendshape_project.data import AudioFeatureExtractor, BlendshapeDataset, collate_batch  # noqa: E402
from blendshape_project.face_refiner import apply_face_refiner, load_face_refiner  # noqa: E402
from blendshape_project.io_utils import load_json, save_json  # noqa: E402
from blendshape_project.train_utils import save_overlay_plot, save_per_blendshape_plot  # noqa: E402


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


def evaluate_bundles(
    bundles: list,
    loaders: list[DataLoader],
    device: torch.device,
    ensemble_weights: list[float],
    face_refiner=None,
    face_refiner_strength: float | None = None,
) -> dict[str, object]:
    blendshape_names = bundles[0].blendshape_names
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_points = 0
    per_dim_abs = torch.zeros(len(blendshape_names), dtype=torch.float64)
    per_dim_count = 0
    sample_payloads: list[dict[str, object]] = []

    mouth_indices = [idx for idx, name in enumerate(blendshape_names) if name.startswith(("jaw", "mouth", "tongue"))]
    jaw_index = blendshape_names.index("jawOpen")
    mouth_abs_error = 0.0
    mouth_points = 0
    jaw_abs_error = 0.0
    jaw_points = 0

    with torch.no_grad():
        for batches in zip(*loaders):
            reference_batch = batches[0]
            mask = reference_batch["target_mask"].to(device)
            lengths = reference_batch["lengths"].to(device)
            predictions_raw = []
            for bundle, batch in zip(bundles, batches):
                prediction = predict_raw_blendshapes(
                    bundle,
                    features=batch["features"].to(device),
                    speaker_ids=batch["speaker_ids"].to(device),
                    lengths=batch["lengths"].to(device),
                    text_ids=batch["text_ids"].to(device),
                    text_lengths=batch["text_lengths"].to(device),
                )
                predictions_raw.append(prediction)

            weight_tensor = torch.tensor(
                ensemble_weights,
                device=device,
                dtype=predictions_raw[0].dtype,
            ).view(-1, 1, 1, 1)
            ensemble_prediction = (torch.stack(predictions_raw, dim=0) * weight_tensor).sum(dim=0)
            if face_refiner is not None:
                ensemble_prediction = apply_face_refiner(
                    ensemble_prediction,
                    face_refiner,
                    strength=face_refiner_strength,
                    clamp=True,
                )
            targets_raw = reference_batch["targets"].to(device)

            valid = mask.unsqueeze(-1).expand_as(ensemble_prediction)
            abs_error = (ensemble_prediction - targets_raw).abs()
            sq_error = (ensemble_prediction - targets_raw) ** 2

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

            for batch_index, sample_id in enumerate(reference_batch["sample_ids"]):
                valid_length = int(lengths[batch_index].item())
                sample_payloads.append(
                    {
                        "sample_id": sample_id,
                        "prediction": ensemble_prediction[batch_index, :valid_length].cpu().numpy(),
                        "target": targets_raw[batch_index, :valid_length].cpu().numpy(),
                    }
                )

    rmse = math.sqrt(total_sq_error / max(total_points, 1))
    return {
        "mae": total_abs_error / max(total_points, 1),
        "rmse": rmse,
        "mouth_mae": mouth_abs_error / max(mouth_points, 1),
        "jaw_open_mae": jaw_abs_error / max(jaw_points, 1),
        "per_dim_mae": (per_dim_abs / max(per_dim_count, 1)).numpy().tolist(),
        "samples": sample_payloads,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one checkpoint or an ensemble on the validation split.")
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "natural_samples.csv")
    parser.add_argument("--split-json", type=Path, default=ROOT / "data" / "manifests" / "split.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "figures")
    parser.add_argument("--ensemble-weights", type=str, default=None)
    parser.add_argument("--face-refiner", type=Path, default=None)
    parser.add_argument("--face-refiner-strength", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    feature_extractor = AudioFeatureExtractor()
    bundles = [load_model_bundle(checkpoint, device=device, feature_dim=feature_extractor.feature_dim) for checkpoint in args.checkpoint]
    ensemble_weights = parse_ensemble_weights(args.ensemble_weights, len(bundles))
    face_refiner = load_face_refiner(args.face_refiner, device=device) if args.face_refiner is not None else None

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

    metrics = evaluate_bundles(
        bundles,
        loaders,
        device,
        ensemble_weights=ensemble_weights,
        face_refiner=face_refiner,
        face_refiner_strength=args.face_refiner_strength,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_dir / "validation_metrics.json",
        metrics
        | {
            "samples": [],
            "checkpoints": [str(bundle.checkpoint_path) for bundle in bundles],
            "ensemble_size": len(bundles),
            "ensemble_weights": ensemble_weights,
            "face_refiner": str(args.face_refiner) if args.face_refiner is not None else None,
            "face_refiner_strength": (
                args.face_refiner_strength
                if args.face_refiner_strength is not None
                else (face_refiner.default_strength if face_refiner is not None else None)
            ),
        },
    )
    save_per_blendshape_plot(metrics["per_dim_mae"], args.output_dir / "validation_per_blendshape_mae.png")
    if metrics["samples"]:
        sample_payload = metrics["samples"][0]
        prefix = "ensemble" if len(bundles) > 1 else "single"
        save_overlay_plot(
            prediction=sample_payload["prediction"],
            target=sample_payload["target"],
            output_path=args.output_dir / f"{prefix}_{sample_payload['sample_id']}_overlay.png",
            title=f"Validation overlay for {sample_payload['sample_id']}",
        )

    label = "Ensemble" if len(bundles) > 1 else "Validation"
    print(f"{label} MAE: {metrics['mae']:.6f}")


if __name__ == "__main__":
    main()
