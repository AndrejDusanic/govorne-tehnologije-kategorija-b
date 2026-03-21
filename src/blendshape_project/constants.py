from __future__ import annotations

from pathlib import Path

BLENDSHAPE_NAMES = [
    "browInnerUp",
    "browDownLeft",
    "browDownRight",
    "browOuterUpLeft",
    "browOuterUpRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "noseSneerLeft",
    "noseSneerRight",
    "jawOpen",
    "jawForward",
    "jawLeft",
    "jawRight",
    "mouthFunnel",
    "mouthPucker",
    "mouthLeft",
    "mouthRight",
    "mouthRollUpper",
    "mouthRollLower",
    "mouthShrugUpper",
    "mouthShrugLower",
    "mouthClose",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "tongueOut",
]

EXPECTED_RAW_FILES = {
    "pdf": "kategorijaB_baza (2).pdf",
    "spk08_zip": "spk08_blendshapes (1).zip",
    "spk14_zip": "spk14_blendshapes (1).zip",
    "labels_zip": "labels_aligned (1).zip",
    "synth_zip": "audio_synth (1).zip",
    "avatar_zip": "avatar.zip",
}

SPEAKER_ORDER = ["spk08", "spk14"]
N_BLENDSHAPES = len(BLENDSHAPE_NAMES)
DEFAULT_SAMPLE_RATE = 44_100
DEFAULT_FPS = 60
DEFAULT_MELS = 80
DEFAULT_SPLIT_SEED = 1337
DEFAULT_VAL_FRACTION = 0.15
PHONEME_PAD = "<pad>"
PHONEME_UNK = "<unk>"
PHONEME_SIL = "SIL"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def blendshape_priority_weights() -> list[float]:
    weights: list[float] = []
    for name in BLENDSHAPE_NAMES:
        if name.startswith(("jaw", "mouth", "tongue")):
            weights.append(2.0)
        elif name.startswith(("cheek", "nose")):
            weights.append(1.2)
        else:
            weights.append(1.0)
    return weights

