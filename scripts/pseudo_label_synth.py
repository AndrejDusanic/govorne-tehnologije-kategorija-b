from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blendshape_project.checkpoint_utils import load_model_bundle, predict_raw_blendshapes  # noqa: E402
from blendshape_project.data import AudioFeatureExtractor, load_waveform, text_to_char_ids  # noqa: E402
from blendshape_project.face_refiner import apply_face_refiner, load_face_refiner  # noqa: E402
from blendshape_project.io_utils import load_json, save_json, write_blendshape_csv  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-label synth samples and build a mixed training manifest.")
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--ensemble-weights", type=str, default=None)
    parser.add_argument("--face-refiner", type=Path, default=None)
    parser.add_argument("--face-refiner-strength", type=float, default=None)
    parser.add_argument("--natural-manifest", type=Path, default=ROOT / "data" / "manifests" / "natural_samples.csv")
    parser.add_argument("--synth-manifest", type=Path, default=ROOT / "data" / "manifests" / "synth_samples.csv")
    parser.add_argument("--split-json", type=Path, default=ROOT / "data" / "manifests" / "split.json")
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synth-sample-weight", type=float, default=0.35)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    feature_extractor = AudioFeatureExtractor(fps=args.fps)
    bundles = [load_model_bundle(checkpoint, device=device, feature_dim=feature_extractor.feature_dim) for checkpoint in args.checkpoint]
    ensemble_weights = parse_ensemble_weights(args.ensemble_weights, len(bundles))
    weight_tensor = torch.tensor(ensemble_weights, dtype=torch.float32, device=device).view(-1, 1, 1)
    face_refiner = load_face_refiner(args.face_refiner, device=device) if args.face_refiner is not None else None

    natural_df = pd.read_csv(args.natural_manifest)
    synth_df = pd.read_csv(args.synth_manifest)
    split = load_json(args.split_json)

    pseudo_dir = args.output_dir / "csv"
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    synth_outputs = []

    with torch.no_grad():
        for record in tqdm(synth_df.to_dict("records"), desc="Pseudo-label synth", leave=False):
            waveform = load_waveform(record["audio_path"], feature_extractor.sample_rate)
            target_frames = int(record.get("n_frames") or max(1, round(float(record["duration_sec"]) * args.fps)))
            features = feature_extractor(waveform, target_frames=target_frames).float().to(device)
            lengths = torch.tensor([target_frames], dtype=torch.long, device=device)

            raw_predictions = []
            text = str(record.get("text", "") or "")
            for bundle in bundles:
                speaker_ids = torch.tensor([bundle.speaker_to_id[record["speaker"]]], dtype=torch.long, device=device)
                text_ids = text_to_char_ids(text, bundle.char_vocab).unsqueeze(0).to(device)
                text_lengths = torch.tensor([text_ids.shape[1]], dtype=torch.long, device=device)
                raw_predictions.append(
                    predict_raw_blendshapes(
                        bundle,
                        features=features.unsqueeze(0),
                        speaker_ids=speaker_ids,
                        lengths=lengths,
                        text_ids=text_ids,
                        text_lengths=text_lengths,
                    ).squeeze(0)
                )

            prediction = (torch.stack(raw_predictions, dim=0) * weight_tensor).sum(dim=0).unsqueeze(0)
            if face_refiner is not None:
                prediction = apply_face_refiner(
                    prediction,
                    face_refiner,
                    strength=args.face_refiner_strength,
                    clamp=True,
                )
            prediction_np = prediction.squeeze(0).cpu().numpy()
            output_path = pseudo_dir / f"{record['sample_id']}.csv"
            write_blendshape_csv(output_path, prediction_np)
            synth_outputs.append(str(output_path.resolve()))

    mixed_split = {
        "train": sorted(split["train"] + synth_df["sample_id"].astype(str).tolist()),
        "val": sorted(split["val"]),
    }

    natural_df = natural_df.copy()
    natural_df["sample_weight"] = 1.0

    synth_df = synth_df.copy()
    synth_df["blendshape_path"] = synth_outputs
    synth_df["sample_weight"] = float(args.synth_sample_weight)
    synth_df["split"] = "train"

    mixed_df = pd.concat([natural_df, synth_df], ignore_index=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_split.parent.mkdir(parents=True, exist_ok=True)
    mixed_df.to_csv(args.output_manifest, index=False)
    save_json(args.output_split, mixed_split)
    save_json(
        args.output_dir / "meta.json",
        {
            "checkpoints": [str(path) for path in args.checkpoint],
            "ensemble_weights": ensemble_weights,
            "face_refiner": str(args.face_refiner) if args.face_refiner is not None else None,
            "face_refiner_strength": (
                args.face_refiner_strength
                if args.face_refiner_strength is not None
                else (face_refiner.default_strength if face_refiner is not None else None)
            ),
            "natural_manifest": str(args.natural_manifest),
            "synth_manifest": str(args.synth_manifest),
            "output_manifest": str(args.output_manifest),
            "output_split": str(args.output_split),
            "n_synth": int(len(synth_df)),
            "synth_sample_weight": float(args.synth_sample_weight),
        },
    )
    print(f"Wrote pseudo labels for {len(synth_df)} synth files.")


if __name__ == "__main__":
    main()
