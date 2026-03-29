from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from .constants import (
    DEFAULT_FPS,
    DEFAULT_MELS,
    DEFAULT_SAMPLE_RATE,
    N_BLENDSHAPES,
    PHONEME_PAD,
    PHONEME_UNK,
    SPEAKER_ORDER,
    TEXT_PAD,
    TEXT_UNK,
)
from .io_utils import framewise_phoneme_labels, read_alignment, read_blendshape_csv


@dataclass
class DatasetStats:
    feature_mean: list[float]
    feature_std: list[float]
    target_mean: list[float]
    target_std: list[float]

    def to_json(self) -> dict[str, list[float]]:
        return {
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }

    @classmethod
    def from_json(cls, payload: dict[str, list[float]]) -> "DatasetStats":
        return cls(**payload)


class AudioFeatureExtractor:
    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        fps: int = DEFAULT_FPS,
        n_mels: int = DEFAULT_MELS,
        n_fft: int = 2048,
        win_length: int = 1470,
        hop_length: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.fps = fps
        self.n_mels = n_mels
        self.hop_length = hop_length or round(sample_rate / fps)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=self.hop_length,
            f_min=20.0,
            f_max=sample_rate / 2,
            n_mels=n_mels,
            power=2.0,
            center=True,
            pad_mode="reflect",
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    @property
    def feature_dim(self) -> int:
        return self.n_mels * 3

    def __call__(self, waveform: torch.Tensor, target_frames: int | None = None) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel = self.mel(waveform).clamp_min(1e-5)
        log_mel = self.db(mel)
        delta = torchaudio.functional.compute_deltas(log_mel)
        delta2 = torchaudio.functional.compute_deltas(delta)
        stacked = torch.cat([log_mel, delta, delta2], dim=1)
        features = stacked.squeeze(0).transpose(0, 1).contiguous()

        if target_frames is None:
            target_frames = max(1, round(waveform.shape[-1] * self.fps / self.sample_rate))
        if features.shape[0] != target_frames:
            features = (
                F.interpolate(
                    features.transpose(0, 1).unsqueeze(0),
                    size=target_frames,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .transpose(0, 1)
                .contiguous()
            )
        return features


def load_waveform(audio_path: str | Path, target_sr: int = DEFAULT_SAMPLE_RATE) -> torch.Tensor:
    audio_path = Path(audio_path)
    if audio_path.suffix.lower() != ".wav":
        waveform_np, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(waveform_np.T.copy())
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        return waveform

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WavFileWarning)
        sample_rate, waveform_np = wavfile.read(str(audio_path))
    if waveform_np.dtype == np.uint8:
        waveform_np = (waveform_np.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(waveform_np.dtype, np.integer):
        scale = float(max(abs(np.iinfo(waveform_np.dtype).min), np.iinfo(waveform_np.dtype).max))
        waveform_np = waveform_np.astype(np.float32) / max(scale, 1.0)
    else:
        waveform_np = waveform_np.astype(np.float32)

    if waveform_np.ndim == 1:
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)
    else:
        waveform = torch.from_numpy(waveform_np.T.copy())

    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    return waveform


def normalize_text(text: str) -> str:
    if text is None:
        value = ""
    elif isinstance(text, float) and np.isnan(text):
        value = ""
    else:
        value = str(text)
    return " ".join(value.strip().lower().split())


def build_char_vocab(texts: list[str]) -> dict[str, int]:
    symbols = {TEXT_PAD, TEXT_UNK}
    for text in texts:
        normalized = normalize_text(text)
        symbols.update(normalized)
    ordered = [TEXT_PAD, TEXT_UNK] + sorted(symbols - {TEXT_PAD, TEXT_UNK})
    return {symbol: index for index, symbol in enumerate(ordered)}


def text_to_char_ids(text: str, vocab: dict[str, int]) -> torch.Tensor:
    normalized = normalize_text(text)
    if not normalized:
        return torch.tensor([vocab.get(TEXT_UNK, 1)], dtype=torch.long)
    return torch.tensor([vocab.get(char, vocab.get(TEXT_UNK, 1)) for char in normalized], dtype=torch.long)


def compute_dataset_stats(
    records: list[dict[str, Any]],
    feature_extractor: AudioFeatureExtractor,
) -> DatasetStats:
    feature_sum = torch.zeros(feature_extractor.feature_dim)
    feature_sq_sum = torch.zeros(feature_extractor.feature_dim)
    feature_count = 0

    target_sum = torch.zeros(N_BLENDSHAPES)
    target_sq_sum = torch.zeros(N_BLENDSHAPES)
    target_count = 0

    for record in tqdm(records, desc="Computing stats", leave=False):
        target = torch.from_numpy(read_blendshape_csv(Path(record["blendshape_path"])))
        waveform = load_waveform(record["audio_path"], feature_extractor.sample_rate)
        features = feature_extractor(waveform, target_frames=target.shape[0]).cpu()

        feature_sum += features.sum(dim=0)
        feature_sq_sum += (features**2).sum(dim=0)
        feature_count += features.shape[0]

        target_sum += target.sum(dim=0)
        target_sq_sum += (target**2).sum(dim=0)
        target_count += target.shape[0]

    feature_mean = feature_sum / max(feature_count, 1)
    feature_var = feature_sq_sum / max(feature_count, 1) - feature_mean**2
    feature_std = torch.sqrt(feature_var.clamp_min(1e-6))

    target_mean = target_sum / max(target_count, 1)
    target_var = target_sq_sum / max(target_count, 1) - target_mean**2
    target_std = torch.sqrt(target_var.clamp_min(1e-6))

    return DatasetStats(
        feature_mean=feature_mean.tolist(),
        feature_std=feature_std.tolist(),
        target_mean=target_mean.tolist(),
        target_std=target_std.tolist(),
    )


class BlendshapeDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        feature_extractor: AudioFeatureExtractor,
        stats: DatasetStats | None = None,
        phoneme_vocab: dict[str, int] | None = None,
        char_vocab: dict[str, int] | None = None,
        speaker_to_id: dict[str, int] | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.stats = stats
        self.phoneme_vocab = phoneme_vocab or {PHONEME_PAD: 0, PHONEME_UNK: 1}
        self.char_vocab = char_vocab or {TEXT_PAD: 0, TEXT_UNK: 1}
        self.speaker_to_id = speaker_to_id or {speaker: idx for idx, speaker in enumerate(SPEAKER_ORDER)}
        self.feature_mean = None
        self.feature_std = None
        self.target_mean = None
        self.target_std = None
        if stats is not None:
            self.feature_mean = torch.tensor(stats.feature_mean, dtype=torch.float32)
            self.feature_std = torch.tensor(stats.feature_std, dtype=torch.float32)
            self.target_mean = torch.tensor(stats.target_mean, dtype=torch.float32)
            self.target_std = torch.tensor(stats.target_std, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.frame.iloc[index].to_dict()
        waveform = load_waveform(record["audio_path"], self.feature_extractor.sample_rate)
        text = record.get("text", "")
        text_ids = text_to_char_ids(text, self.char_vocab)

        target_tensor: torch.Tensor | None = None
        target_activity = None
        if pd.notna(record.get("blendshape_path", None)) and str(record.get("blendshape_path", "")).strip():
            target_tensor = torch.from_numpy(read_blendshape_csv(Path(record["blendshape_path"]))).float()
            target_activity = target_tensor.clamp_min(0.0)
            target_frames = target_tensor.shape[0]
        else:
            target_frames = max(1, int(round(float(record["duration_sec"]) * self.feature_extractor.fps)))

        features = self.feature_extractor(waveform, target_frames=target_frames).float()

        if self.feature_mean is not None and self.feature_std is not None:
            features = (features - self.feature_mean) / self.feature_std

        if target_tensor is not None and self.target_mean is not None and self.target_std is not None:
            target_tensor = (target_tensor - self.target_mean) / self.target_std

        phoneme_ids = torch.full((target_frames,), fill_value=-100, dtype=torch.long)
        phoneme_path = record.get("phoneme_path", "")
        if pd.notna(phoneme_path) and str(phoneme_path).strip():
            labels = framewise_phoneme_labels(
                read_alignment(Path(phoneme_path)),
                n_frames=target_frames,
                fps=int(record.get("fps", DEFAULT_FPS)),
            )
            phoneme_ids = torch.tensor(
                [self.phoneme_vocab.get(label, self.phoneme_vocab.get(PHONEME_UNK, 1)) for label in labels],
                dtype=torch.long,
            )

        return {
            "sample_id": record["sample_id"],
            "speaker": record["speaker"],
            "speaker_id": self.speaker_to_id[record["speaker"]],
            "text": text,
            "text_ids": text_ids,
            "features": features,
            "targets": target_tensor,
            "target_activity": target_activity,
            "phoneme_ids": phoneme_ids,
            "length": features.shape[0],
            "duration_sec": float(record["duration_sec"]),
            "fps": int(record.get("fps", DEFAULT_FPS)),
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    features = pad_sequence([item["features"] for item in batch], batch_first=True)
    max_length = features.shape[1]
    text_ids = pad_sequence([item["text_ids"] for item in batch], batch_first=True, padding_value=0)
    text_lengths = torch.tensor([item["text_ids"].shape[0] for item in batch], dtype=torch.long)

    targets = torch.zeros(features.shape[0], max_length, N_BLENDSHAPES, dtype=torch.float32)
    target_activity = torch.zeros(features.shape[0], max_length, N_BLENDSHAPES, dtype=torch.float32)
    target_mask = torch.zeros(features.shape[0], max_length, dtype=torch.bool)
    phoneme_ids = torch.full((features.shape[0], max_length), fill_value=-100, dtype=torch.long)

    for batch_index, item in enumerate(batch):
        length = item["length"]
        phoneme_ids[batch_index, :length] = item["phoneme_ids"]
        if item["targets"] is not None:
            targets[batch_index, :length] = item["targets"]
            target_activity[batch_index, :length] = item["target_activity"]
            target_mask[batch_index, :length] = True

    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "speakers": [item["speaker"] for item in batch],
        "speaker_ids": torch.tensor([item["speaker_id"] for item in batch], dtype=torch.long),
        "texts": [item["text"] for item in batch],
        "text_ids": text_ids,
        "text_lengths": text_lengths,
        "features": features,
        "targets": targets,
        "target_activity": target_activity,
        "target_mask": target_mask,
        "phoneme_ids": phoneme_ids,
        "lengths": lengths,
        "durations_sec": torch.tensor([item["duration_sec"] for item in batch], dtype=torch.float32),
        "fps": torch.tensor([item["fps"] for item in batch], dtype=torch.long),
    }


def unnormalize_targets(values: torch.Tensor, stats: DatasetStats) -> torch.Tensor:
    mean = torch.tensor(stats.target_mean, dtype=values.dtype, device=values.device)
    std = torch.tensor(stats.target_std, dtype=values.dtype, device=values.device)
    return values * std + mean
