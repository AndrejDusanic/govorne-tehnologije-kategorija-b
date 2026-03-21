from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blendshape_project.constants import DEFAULT_FPS, SPEAKER_ORDER  # noqa: E402
from blendshape_project.data import AudioFeatureExtractor, DatasetStats, load_waveform, unnormalize_targets  # noqa: E402
from blendshape_project.io_utils import save_json, write_blendshape_csv  # noqa: E402
from blendshape_project.model import BlendshapeRegressor  # noqa: E402


def infer_speaker_id(filename: str, speaker_to_id: dict[str, int], default_speaker: str) -> int:
    prefix = filename.split("_")[0]
    if prefix in speaker_to_id:
        return speaker_to_id[prefix]
    return speaker_to_id[default_speaker]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference over a folder of WAV files.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "predictions")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--default-speaker", type=str, choices=SPEAKER_ORDER, default="spk08")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stats = DatasetStats.from_json(checkpoint["stats"])
    speaker_to_id = checkpoint.get("speaker_to_id", {speaker: idx for idx, speaker in enumerate(SPEAKER_ORDER)})

    feature_extractor = AudioFeatureExtractor(fps=args.fps)
    model = BlendshapeRegressor(
        input_dim=feature_extractor.feature_dim,
        num_blendshapes=len(checkpoint["blendshape_names"]),
        num_speakers=len(speaker_to_id),
        num_phonemes=len(checkpoint["phoneme_vocab"]),
        hidden_size=checkpoint["config"].get("hidden_size", 256),
        dropout=checkpoint["config"].get("dropout", 0.12),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    feature_mean = torch.tensor(stats.feature_mean, dtype=torch.float32, device=device)
    feature_std = torch.tensor(stats.feature_std, dtype=torch.float32, device=device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wav_paths = sorted(args.input_dir.glob("*.wav"))
    meta = {
        "system": {
            "lookahead_ms": 0,
            "fps_out": args.fps,
        },
        "files": {},
    }

    with torch.no_grad():
        for wav_path in wav_paths:
            waveform = load_waveform(wav_path, feature_extractor.sample_rate)
            target_frames = max(1, int(round(waveform.shape[-1] / feature_extractor.sample_rate * args.fps)))
            features = feature_extractor(waveform, target_frames=target_frames).to(device)
            features = (features - feature_mean) / feature_std
            speaker_id = infer_speaker_id(wav_path.stem, speaker_to_id, args.default_speaker)

            start_time = time.perf_counter()
            outputs = model(features.unsqueeze(0), torch.tensor([speaker_id], device=device))
            elapsed = time.perf_counter() - start_time

            prediction = unnormalize_targets(outputs["blendshapes"], stats).squeeze(0).cpu().numpy()
            prediction = np.clip(prediction, 0.0, 1.0)
            output_csv = args.output_dir / f"{wav_path.stem}.csv"
            write_blendshape_csv(output_csv, prediction)

            duration_sec = waveform.shape[-1] / feature_extractor.sample_rate
            meta["files"][wav_path.name] = {
                "csv_path": output_csv.name,
                "inference_time_sec": elapsed,
                "rtf": elapsed / max(duration_sec, 1e-6),
            }

    save_json(args.output_dir / "meta.json", meta)
    print(f"Inference completed for {len(wav_paths)} files.")


if __name__ == "__main__":
    main()
