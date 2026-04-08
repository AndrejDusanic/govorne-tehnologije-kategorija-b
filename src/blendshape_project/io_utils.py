from __future__ import annotations

import csv
import json
import math
import random
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .constants import DEFAULT_FPS, PHONEME_SIL


@dataclass(frozen=True)
class AudioMetadata:
    sample_rate: int
    num_frames: int

    @property
    def duration_sec(self) -> float:
        return self.num_frames / max(self.sample_rate, 1)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_number_from_name(path_like: str | Path) -> int:
    match = re.search(r"_(\d+)\.(?:csv|wav|txt)$", str(path_like))
    if not match:
        raise ValueError(f"Could not parse sample number from {path_like}")
    return int(match.group(1))


def read_transcripts_xlsx(path: Path) -> dict[int, str]:
    dataframe = pd.read_excel(path, header=None, engine="openpyxl")
    transcripts: dict[int, str] = {}
    if dataframe.shape[1] >= 2:
        for _, row in dataframe.iterrows():
            sample_number = int(row.iloc[0])
            text_value = row.iloc[1]
            transcripts[sample_number] = "" if pd.isna(text_value) else str(text_value).strip()
        return transcripts

    for idx, text_value in enumerate(dataframe.iloc[:, 0].tolist(), start=1):
        transcripts[idx] = "" if pd.isna(text_value) else str(text_value).strip()
    return transcripts


def read_blendshape_csv(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    return values


def read_wav_metadata(path: str | Path) -> AudioMetadata:
    wav_path = Path(path)
    with wave.open(str(wav_path), "rb") as handle:
        return AudioMetadata(
            sample_rate=handle.getframerate(),
            num_frames=handle.getnframes(),
        )


def write_blendshape_csv(path: Path, values: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in values:
            writer.writerow([f"{float(item):.6f}" for item in row])


def read_alignment(path: Path) -> list[tuple[float, float, str]]:
    items: list[tuple[float, float, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = re.split(r"\s+", stripped)
            if len(parts) < 3:
                continue
            start, end, label = float(parts[0]), float(parts[1]), parts[2]
            items.append((start, end, label))
    return items


def framewise_phoneme_labels(
    alignment: list[tuple[float, float, str]],
    n_frames: int,
    fps: int = DEFAULT_FPS,
) -> list[str]:
    labels = [PHONEME_SIL] * n_frames
    if not alignment:
        return labels
    pointer = 0
    for frame_idx in range(n_frames):
        center_time = (frame_idx + 0.5) / fps
        while pointer + 1 < len(alignment) and center_time >= alignment[pointer][1]:
            pointer += 1
        start, end, label = alignment[pointer]
        if start <= center_time <= end:
            labels[frame_idx] = label
        elif center_time < start and pointer > 0:
            labels[frame_idx] = alignment[pointer - 1][2]
        else:
            labels[frame_idx] = label
    return labels


def round_duration_to_frames(duration_sec: float, fps: int = DEFAULT_FPS) -> int:
    return max(1, int(round(duration_sec * fps)))


def sorted_numeric_paths(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda item: sample_number_from_name(item))
