from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blendshape_project.constants import BLENDSHAPE_NAMES  # noqa: E402
from blendshape_project.io_utils import read_alignment, read_blendshape_csv  # noqa: E402


def plot_dataset_overview(frame: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    counts = frame.groupby("speaker")["sample_id"].count().reindex(sorted(frame["speaker"].unique()))
    plt.bar(counts.index.tolist(), counts.values.tolist(), color=["#2563eb", "#dc2626"])
    plt.title("Natural samples per speaker")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)

    plt.subplot(1, 2, 2)
    for speaker, color in [("spk08", "#2563eb"), ("spk14", "#dc2626")]:
        values = frame.loc[frame["speaker"] == speaker, "duration_sec"]
        plt.hist(values, bins=20, alpha=0.55, label=speaker, color=color)
    plt.title("Audio duration distribution")
    plt.xlabel("Seconds")
    plt.ylabel("Samples")
    plt.legend()
    plt.grid(alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_blendshape_activity(frame: pd.DataFrame, output_path: Path) -> None:
    sums = None
    sq_sums = None
    count = 0
    for path in tqdm(frame["blendshape_path"].tolist(), desc="Blendshape activity", leave=False):
        values = read_blendshape_csv(Path(path))
        if sums is None:
            sums = np.zeros(values.shape[1], dtype=np.float64)
            sq_sums = np.zeros(values.shape[1], dtype=np.float64)
        sums += values.sum(axis=0)
        sq_sums += (values**2).sum(axis=0)
        count += values.shape[0]
    means = sums / max(count, 1)
    stds = np.sqrt(np.maximum(sq_sums / max(count, 1) - means**2, 1e-8))
    order = np.argsort(stds)[::-1]

    plt.figure(figsize=(14, 7))
    plt.bar(range(len(order)), stds[order], color="#0f766e")
    plt.xticks(range(len(order)), [BLENDSHAPE_NAMES[idx] for idx in order], rotation=80, ha="right")
    plt.ylabel("Std. dev.")
    plt.title("Blendshape activity (standard deviation over all frames)")
    plt.grid(axis="y", alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_phoneme_distribution(frame: pd.DataFrame, output_path: Path) -> None:
    duration_counter: defaultdict[str, float] = defaultdict(float)
    for path in tqdm(frame["phoneme_path"].tolist(), desc="Phoneme distribution", leave=False):
        for start, end, label in read_alignment(Path(path)):
            duration_counter[label] += max(0.0, end - start)
    top_items = Counter(duration_counter).most_common(25)
    labels = [item[0] for item in top_items]
    values = [item[1] for item in top_items]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(values)), values, color="#7c3aed")
    plt.xticks(range(len(values)), labels, rotation=65, ha="right")
    plt.ylabel("Total aligned duration [s]")
    plt.title("Top phonemes by aligned duration")
    plt.grid(axis="y", alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create EDA figures for the competition dataset.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "manifests" / "natural_samples.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "figures",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    plot_dataset_overview(frame, args.output_dir / "dataset_overview.png")
    plot_blendshape_activity(frame, args.output_dir / "blendshape_activity.png")
    plot_phoneme_distribution(frame, args.output_dir / "phoneme_distribution.png")
    print(f"Saved analysis figures to {args.output_dir}")


if __name__ == "__main__":
    main()

