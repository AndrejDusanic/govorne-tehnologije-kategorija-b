from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .constants import BLENDSHAPE_NAMES, DEFAULT_FPS


EYE_BLINK_LEFT = BLENDSHAPE_NAMES.index("eyeBlinkLeft")
EYE_BLINK_RIGHT = BLENDSHAPE_NAMES.index("eyeBlinkRight")
EYE_SQUINT_LEFT = BLENDSHAPE_NAMES.index("eyeSquintLeft")
EYE_SQUINT_RIGHT = BLENDSHAPE_NAMES.index("eyeSquintRight")
EYE_WIDE_LEFT = BLENDSHAPE_NAMES.index("eyeWideLeft")
EYE_WIDE_RIGHT = BLENDSHAPE_NAMES.index("eyeWideRight")
BROW_DOWN_LEFT = BLENDSHAPE_NAMES.index("browDownLeft")
BROW_DOWN_RIGHT = BLENDSHAPE_NAMES.index("browDownRight")


@dataclass(frozen=True)
class BlinkConfig:
    min_clip_sec: float = 1.2
    short_clip_single_blink_sec: float = 1.7
    short_clip_blink_probability: float = 0.45
    initial_offset_min_sec: float = 0.9
    initial_offset_max_sec: float = 2.2
    min_interval_sec: float = 2.4
    max_interval_sec: float = 4.8
    tail_guard_sec: float = 0.35
    min_duration_sec: float = 0.11
    max_duration_sec: float = 0.19
    min_peak: float = 0.45
    max_peak: float = 0.78
    asymmetry: float = 0.08
    squint_scale: float = 0.18
    brow_down_scale: float = 0.10
    eye_wide_suppress: float = 0.50


def _stable_seed(seed: int, file_key: str) -> int:
    payload = f"{seed}:{file_key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _pulse(frames: np.ndarray, center: int, half_width: float) -> np.ndarray:
    normalized = (frames - float(center)) / max(float(half_width), 1.0)
    pulse = np.zeros_like(normalized, dtype=np.float32)
    mask = np.abs(normalized) <= 1.0
    pulse[mask] = np.cos(normalized[mask] * np.pi * 0.5, dtype=np.float32) ** 2
    return pulse


def _sample_blink_centers(duration_sec: float, rng: np.random.Generator, config: BlinkConfig) -> list[float]:
    if duration_sec < config.min_clip_sec:
        if duration_sec < config.short_clip_single_blink_sec or rng.random() > config.short_clip_blink_probability:
            return []
        return [rng.uniform(duration_sec * 0.35, duration_sec * 0.75)]

    centers: list[float] = []
    cursor = rng.uniform(config.initial_offset_min_sec, config.initial_offset_max_sec)
    while cursor < duration_sec - config.tail_guard_sec:
        centers.append(cursor)
        cursor += rng.uniform(config.min_interval_sec, config.max_interval_sec)
    return centers


def apply_random_blinks(
    values: np.ndarray,
    fps: int = DEFAULT_FPS,
    seed: int = 1337,
    file_key: str = "",
    strength: float = 1.0,
    config: BlinkConfig | None = None,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    if config is None:
        config = BlinkConfig()

    base = np.asarray(values, dtype=np.float32)
    output = np.array(base, copy=True)
    if strength <= 0.0 or output.ndim != 2 or output.shape[0] < 3:
        info = {"count": 0, "centers_sec": [], "seed": seed, "strength": strength}
        return (output, info) if return_info else output

    rng = np.random.default_rng(_stable_seed(seed, file_key))
    n_frames = int(output.shape[0])
    duration_sec = n_frames / max(int(fps), 1)
    centers_sec = _sample_blink_centers(duration_sec, rng, config)

    left_track = np.zeros(n_frames, dtype=np.float32)
    right_track = np.zeros(n_frames, dtype=np.float32)
    for center_sec in centers_sec:
        center = int(round(center_sec * fps))
        duration_frames = max(6, int(round(rng.uniform(config.min_duration_sec, config.max_duration_sec) * fps)))
        half_width = max(duration_frames / 2.0, 2.5)
        start = max(0, int(np.floor(center - half_width - 1)))
        end = min(n_frames, int(np.ceil(center + half_width + 2)))
        frames = np.arange(start, end, dtype=np.float32)
        pulse = _pulse(frames, center=center, half_width=half_width)
        peak = float(np.clip(rng.uniform(config.min_peak, config.max_peak) * strength, 0.0, 1.0))
        left_scale = peak * rng.uniform(1.0 - config.asymmetry, 1.0 + config.asymmetry)
        right_scale = peak * rng.uniform(1.0 - config.asymmetry, 1.0 + config.asymmetry)
        left_track[start:end] = np.maximum(left_track[start:end], pulse * left_scale)
        right_track[start:end] = np.maximum(right_track[start:end], pulse * right_scale)

    output[:, EYE_BLINK_LEFT] = np.clip(output[:, EYE_BLINK_LEFT] + left_track, 0.0, 1.0)
    output[:, EYE_BLINK_RIGHT] = np.clip(output[:, EYE_BLINK_RIGHT] + right_track, 0.0, 1.0)
    output[:, EYE_SQUINT_LEFT] = np.clip(output[:, EYE_SQUINT_LEFT] + left_track * config.squint_scale, 0.0, 1.0)
    output[:, EYE_SQUINT_RIGHT] = np.clip(output[:, EYE_SQUINT_RIGHT] + right_track * config.squint_scale, 0.0, 1.0)
    output[:, BROW_DOWN_LEFT] = np.clip(output[:, BROW_DOWN_LEFT] + left_track * config.brow_down_scale, 0.0, 1.0)
    output[:, BROW_DOWN_RIGHT] = np.clip(output[:, BROW_DOWN_RIGHT] + right_track * config.brow_down_scale, 0.0, 1.0)
    output[:, EYE_WIDE_LEFT] = np.clip(output[:, EYE_WIDE_LEFT] * (1.0 - left_track * config.eye_wide_suppress), 0.0, 1.0)
    output[:, EYE_WIDE_RIGHT] = np.clip(output[:, EYE_WIDE_RIGHT] * (1.0 - right_track * config.eye_wide_suppress), 0.0, 1.0)

    info = {
        "count": len(centers_sec),
        "centers_sec": [round(center, 3) for center in centers_sec],
        "seed": seed,
        "strength": strength,
    }
    return (output, info) if return_info else output
