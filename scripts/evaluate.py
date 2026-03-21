from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blendshape_project.constants import SPEAKER_ORDER  # noqa: E402
from blendshape_project.data import AudioFeatureExtractor, BlendshapeDataset, DatasetStats, collate_batch  # noqa: E402
from blendshape_project.io_utils import load_json, save_json  # noqa: E402
from blendshape_project.model import BlendshapeRegressor  # noqa: E402
from blendshape_project.train_utils import evaluate_model, save_overlay_plot, save_per_blendshape_plot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on the validation split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "natural_samples.csv")
    parser.add_argument("--split-json", type=Path, default=ROOT / "data" / "manifests" / "split.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "figures")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stats = DatasetStats.from_json(checkpoint["stats"])
    speaker_to_id = checkpoint.get("speaker_to_id", {speaker: idx for idx, speaker in enumerate(SPEAKER_ORDER)})
    phoneme_vocab = checkpoint["phoneme_vocab"]

    frame = pd.read_csv(args.manifest)
    split = load_json(args.split_json)
    val_df = frame[frame["sample_id"].isin(split["val"])].copy()

    feature_extractor = AudioFeatureExtractor()
    dataset = BlendshapeDataset(val_df, feature_extractor, stats=stats, phoneme_vocab=phoneme_vocab, speaker_to_id=speaker_to_id)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)

    model = BlendshapeRegressor(
        input_dim=feature_extractor.feature_dim,
        num_blendshapes=len(checkpoint["blendshape_names"]),
        num_speakers=len(speaker_to_id),
        num_phonemes=len(phoneme_vocab),
        hidden_size=checkpoint["config"].get("hidden_size", 256),
        dropout=checkpoint["config"].get("dropout", 0.12),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    metrics = evaluate_model(model, loader, device, stats)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "validation_metrics.json", metrics | {"samples": []})
    save_per_blendshape_plot(metrics["per_dim_mae"], args.output_dir / "validation_per_blendshape_mae.png")
    if metrics["samples"]:
        sample_payload = metrics["samples"][0]
        save_overlay_plot(
            prediction=sample_payload["prediction"],
            target=sample_payload["target"],
            output_path=args.output_dir / f"{sample_payload['sample_id']}_overlay.png",
            title=f"Validation overlay for {sample_payload['sample_id']}",
        )
    print(f"Validation MAE: {metrics['mae']:.6f}")


if __name__ == "__main__":
    main()
