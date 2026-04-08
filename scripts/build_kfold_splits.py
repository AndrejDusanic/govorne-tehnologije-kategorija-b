from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blendshape_project.io_utils import save_json  # noqa: E402


def chunked(items: list[str], n_chunks: int) -> list[list[str]]:
    chunks = [[] for _ in range(n_chunks)]
    for index, item in enumerate(items):
        chunks[index % n_chunks].append(item)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build speaker-balanced k-fold split files for natural samples.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "natural_samples.csv")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "manifests" / "kfold_5_seed1337")
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    speaker_to_chunks: dict[str, list[list[str]]] = {}
    for speaker, speaker_frame in frame.groupby("speaker"):
        sample_ids = speaker_frame["sample_id"].astype(str).tolist()
        rng.shuffle(sample_ids)
        speaker_to_chunks[speaker] = chunked(sample_ids, args.n_folds)

    for fold_index in range(args.n_folds):
        val_ids: list[str] = []
        train_ids: list[str] = []
        for speaker_chunks in speaker_to_chunks.values():
            for chunk_index, chunk in enumerate(speaker_chunks):
                if chunk_index == fold_index:
                    val_ids.extend(chunk)
                else:
                    train_ids.extend(chunk)
        payload = {
            "fold_index": fold_index,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "train": sorted(train_ids),
            "val": sorted(val_ids),
        }
        save_json(args.output_dir / f"fold_{fold_index}.json", payload)

    save_json(
        args.output_dir / "meta.json",
        {
            "manifest": str(args.manifest),
            "n_folds": args.n_folds,
            "seed": args.seed,
            "folds": [f"fold_{index}.json" for index in range(args.n_folds)],
        },
    )
    print(f"Wrote {args.n_folds} fold files to {args.output_dir}")


if __name__ == "__main__":
    main()
