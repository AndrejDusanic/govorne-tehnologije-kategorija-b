from __future__ import annotations

import argparse
import sys
import time
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blendshape_project.checkpoint_utils import load_model_bundle, predict_raw_blendshapes  # noqa: E402
from blendshape_project.blink_postprocess import apply_random_blinks  # noqa: E402
from blendshape_project.constants import DEFAULT_FPS, SPEAKER_ORDER  # noqa: E402
from blendshape_project.data import AudioFeatureExtractor, load_waveform, text_to_char_ids  # noqa: E402
from blendshape_project.face_refiner import apply_face_refiner, load_face_refiner  # noqa: E402
from blendshape_project.io_utils import save_json, write_blendshape_csv  # noqa: E402


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def infer_speaker_id(filename: str, speaker_to_id: dict[str, int], default_speaker: str) -> int:
    prefix = filename.split("_")[0]
    if prefix in speaker_to_id:
        return speaker_to_id[prefix]
    return speaker_to_id[default_speaker]


def read_text_for_audio(wav_path: Path, text_dir: Path | None, default_text: str) -> str:
    if text_dir is None:
        return default_text
    text_path = text_dir / f"{wav_path.stem}.txt"
    if text_path.exists():
        return text_path.read_text(encoding="utf-8").strip()
    return default_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference over a folder of WAV files.")
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "predictions")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--default-speaker", type=str, choices=SPEAKER_ORDER, default="spk08")
    parser.add_argument("--text-dir", type=Path, default=None)
    parser.add_argument("--default-text", type=str, default="")
    parser.add_argument("--face-refiner", type=Path, default=None)
    parser.add_argument("--face-refiner-strength", type=float, default=None)
    parser.add_argument("--random-blinks", action="store_true")
    parser.add_argument("--blink-strength", type=float, default=1.0)
    parser.add_argument("--blink-seed", type=int, default=1337)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    args = parser.parse_args()

    device = select_device(args.device)
    feature_extractor = AudioFeatureExtractor(fps=args.fps)
    bundles = [load_model_bundle(checkpoint, device=device, feature_dim=feature_extractor.feature_dim) for checkpoint in args.checkpoint]
    face_refiner = load_face_refiner(args.face_refiner, device=device) if args.face_refiner is not None else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_paths = sorted(
        path
        for pattern in ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")
        for path in args.input_dir.glob(pattern)
    )
    meta = {
        "system": {
            "lookahead_ms": 0,
            "fps_out": args.fps,
            "ensemble_size": len(bundles),
            "checkpoints": [f"{bundle.checkpoint_path.parent.name}/{bundle.checkpoint_path.name}" for bundle in bundles],
            "face_refiner": str(args.face_refiner) if args.face_refiner is not None else None,
            "face_refiner_strength": (
                args.face_refiner_strength
                if args.face_refiner_strength is not None
                else (face_refiner.default_strength if face_refiner is not None else None)
            ),
            "random_blinks": args.random_blinks,
            "blink_strength": args.blink_strength if args.random_blinks else None,
            "blink_seed": args.blink_seed if args.random_blinks else None,
        },
        "files": {},
    }
    max_duration_sec = 0.0

    with torch.no_grad():
        for wav_path in audio_paths:
            waveform = load_waveform(wav_path, feature_extractor.sample_rate)
            target_frames = max(1, int(round(waveform.shape[-1] / feature_extractor.sample_rate * args.fps)))
            raw_features = feature_extractor(waveform, target_frames=target_frames).float().to(device)
            lengths = torch.tensor([target_frames], dtype=torch.long, device=device)
            text = read_text_for_audio(wav_path, args.text_dir, args.default_text)

            start_time = time.perf_counter()
            raw_predictions = []
            for bundle in bundles:
                speaker_id = infer_speaker_id(wav_path.stem, bundle.speaker_to_id, args.default_speaker)
                speaker_ids = torch.tensor([speaker_id], dtype=torch.long, device=device)
                text_ids = text_to_char_ids(text, bundle.char_vocab).unsqueeze(0).to(device)
                text_lengths = torch.tensor([text_ids.shape[1]], dtype=torch.long, device=device)
                prediction = predict_raw_blendshapes(
                    bundle,
                    features=raw_features.unsqueeze(0),
                    speaker_ids=speaker_ids,
                    lengths=lengths,
                    text_ids=text_ids,
                    text_lengths=text_lengths,
                )
                raw_predictions.append(prediction.squeeze(0))
            elapsed = time.perf_counter() - start_time

            prediction = torch.stack(raw_predictions, dim=0).mean(dim=0).unsqueeze(0)
            if face_refiner is not None:
                prediction = apply_face_refiner(
                    prediction,
                    face_refiner,
                    strength=args.face_refiner_strength,
                    clamp=True,
                )
            prediction = prediction.squeeze(0).cpu().numpy()
            blink_info = None
            if args.random_blinks:
                prediction, blink_info = apply_random_blinks(
                    prediction,
                    fps=args.fps,
                    seed=args.blink_seed,
                    file_key=wav_path.stem,
                    strength=args.blink_strength,
                    return_info=True,
                )
            prediction = np.clip(prediction, 0.0, 1.0)
            output_csv = args.output_dir / f"{wav_path.stem}.csv"
            write_blendshape_csv(output_csv, prediction)

            duration_sec = waveform.shape[-1] / feature_extractor.sample_rate
            max_duration_sec = max(max_duration_sec, duration_sec)
            meta["files"][wav_path.name] = {
                "csv_path": output_csv.name,
                "text_used": text,
                "inference_time_sec": elapsed,
                "rtf": elapsed / max(duration_sec, 1e-6),
                "blink_postprocess": blink_info,
            }

    if any(bundle.config.get("temporal_encoder", "causal_tcn") == "bgru" for bundle in bundles):
        meta["system"]["lookahead_ms"] = int(math.ceil(max_duration_sec * 1000.0))

    save_json(args.output_dir / "meta.json", meta)
    print(f"Inference completed for {len(audio_paths)} files.")


if __name__ == "__main__":
    main()
