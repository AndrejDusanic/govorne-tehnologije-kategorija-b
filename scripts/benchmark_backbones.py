from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blendshape_project.data import AudioFeatureExtractor, load_waveform  # noqa: E402
from blendshape_project.io_utils import save_json  # noqa: E402


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def benchmark_mel(audio_paths: list[Path], fps: int) -> dict[str, float]:
    extractor = AudioFeatureExtractor(fps=fps)
    times = []
    durations = []
    dims = []
    with torch.no_grad():
        for path in audio_paths:
            waveform = load_waveform(path, extractor.sample_rate)
            duration_sec = waveform.shape[-1] / extractor.sample_rate
            target_frames = max(1, int(round(duration_sec * fps)))
            start = time.perf_counter()
            features = extractor(waveform, target_frames=target_frames)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            durations.append(duration_sec)
            dims.append(int(features.shape[-1]))
    avg_time = statistics.mean(times)
    avg_duration = statistics.mean(durations)
    return {
        "avg_time_sec": avg_time,
        "avg_duration_sec": avg_duration,
        "avg_rtf": avg_time / max(avg_duration, 1e-6),
        "feature_dim": int(statistics.mean(dims)),
        "n_files": len(audio_paths),
    }


def benchmark_pretrained(audio_paths: list[Path], bundle_name: str, device: torch.device) -> dict[str, float]:
    bundle = getattr(torchaudio.pipelines, bundle_name)
    model = bundle.get_model().to(device)
    model.eval()

    times = []
    durations = []
    dims = []
    with torch.no_grad():
        for path in audio_paths:
            waveform = load_waveform(path, bundle.sample_rate).to(device)
            duration_sec = waveform.shape[-1] / bundle.sample_rate
            start = time.perf_counter()
            features, _ = model.extract_features(waveform)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            last_layer = features[-1]
            dims.append(int(last_layer.shape[-1]))
            times.append(elapsed)
            durations.append(duration_sec)

    avg_time = statistics.mean(times)
    avg_duration = statistics.mean(durations)
    return {
        "bundle": bundle_name,
        "sample_rate": bundle.sample_rate,
        "avg_time_sec": avg_time,
        "avg_duration_sec": avg_duration,
        "avg_rtf": avg_time / max(avg_duration, 1e-6),
        "feature_dim": int(statistics.mean(dims)),
        "n_files": len(audio_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark mel and pretrained audio backbones on a few files.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "natural_samples.csv")
    parser.add_argument("--n-files", type=int, default=5)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=Path, default=ROOT / "reports" / "figures" / "backbone_benchmark.json")
    args = parser.parse_args()

    device = select_device(args.device)
    frame = pd.read_csv(args.manifest)
    audio_paths = [Path(path) for path in frame["audio_path"].tolist()[: args.n_files]]

    results = {
        "device": str(device),
        "mel": benchmark_mel(audio_paths, fps=args.fps),
        "hubert_base": benchmark_pretrained(audio_paths, "HUBERT_BASE", device=device),
        "wavlm_base": benchmark_pretrained(audio_paths, "WAVLM_BASE", device=device),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output_json, results)
    print(results)


if __name__ == "__main__":
    main()
