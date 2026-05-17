from __future__ import annotations

from pathlib import Path

from .constants import PHONEME_PAD, PHONEME_SIL, PHONEME_UNK
from .io_utils import read_alignment


VISEME_VARIANTS: dict[str, dict[str, str]] = {
    "viseme_coarse_8": {
        "SIL": "silence",
        "A": "open_vowel",
        "E": "front_vowel",
        "I": "front_vowel",
        "O": "rounded_vowel",
        "U": "rounded_vowel",
        "Y": "neutral_vowel",
        "P": "closed_lips",
        "B": "closed_lips",
        "M": "closed_lips",
        "F": "teeth_lip",
        "V": "teeth_lip",
        "T": "tip_tongue",
        "D": "tip_tongue",
        "N": "tip_tongue",
        "R": "tip_tongue",
        "L": "tip_tongue",
        "LJ": "tip_tongue",
        "NJ": "tip_tongue",
        "J": "tip_tongue",
        "S": "sibilant",
        "Z": "sibilant",
        "Š": "sibilant",
        "Ž": "sibilant",
        "C": "sibilant",
        "Č": "sibilant",
        "Ć": "sibilant",
        "DŽ": "sibilant",
        "Đ": "sibilant",
        "K": "back_consonant",
        "G": "back_consonant",
        "H": "back_consonant",
    },
    "viseme_balanced_10": {
        "SIL": "silence",
        "A": "open_vowel",
        "E": "mid_front_vowel",
        "I": "tight_front_vowel",
        "O": "rounded_vowel",
        "U": "rounded_vowel",
        "Y": "neutral_vowel",
        "P": "bilabial_stop",
        "B": "bilabial_stop",
        "M": "bilabial_nasal",
        "F": "labiodental",
        "V": "labiodental",
        "T": "alveolar_stop",
        "D": "alveolar_stop",
        "N": "alveolar_stop",
        "R": "alveolar_liquid",
        "L": "alveolar_liquid",
        "LJ": "alveolar_liquid",
        "NJ": "alveolar_liquid",
        "J": "alveolar_liquid",
        "S": "sibilant_affricate",
        "Z": "sibilant_affricate",
        "Š": "sibilant_affricate",
        "Ž": "sibilant_affricate",
        "C": "sibilant_affricate",
        "Č": "sibilant_affricate",
        "Ć": "sibilant_affricate",
        "DŽ": "sibilant_affricate",
        "Đ": "sibilant_affricate",
        "K": "velar_glottal",
        "G": "velar_glottal",
        "H": "velar_glottal",
    },
    "viseme_fine_12": {
        "SIL": "silence",
        "A": "open_vowel",
        "E": "mid_front_vowel",
        "I": "tight_front_vowel",
        "O": "mid_rounded_vowel",
        "U": "tight_rounded_vowel",
        "Y": "neutral_vowel",
        "P": "bilabial_stop",
        "B": "bilabial_stop",
        "M": "bilabial_nasal",
        "F": "labiodental",
        "V": "labiodental",
        "T": "alveolar_stop_nasal",
        "D": "alveolar_stop_nasal",
        "N": "alveolar_stop_nasal",
        "R": "liquid_palatal",
        "L": "liquid_palatal",
        "LJ": "liquid_palatal",
        "NJ": "liquid_palatal",
        "J": "liquid_palatal",
        "S": "sibilant",
        "Z": "sibilant",
        "Š": "postalveolar",
        "Ž": "postalveolar",
        "C": "affricate",
        "Č": "affricate",
        "Ć": "affricate",
        "DŽ": "affricate",
        "Đ": "affricate",
        "K": "velar",
        "G": "velar",
        "H": "glottal",
    },
}


def available_aux_target_types() -> list[str]:
    return ["phoneme", "viseme"]


def available_viseme_variants() -> list[str]:
    return sorted(VISEME_VARIANTS)


def project_aux_label(
    label: str,
    aux_target_type: str = "phoneme",
    viseme_variant: str = "viseme_balanced_10",
) -> str:
    normalized = str(label).strip().upper()
    if aux_target_type == "phoneme":
        return normalized
    if aux_target_type != "viseme":
        raise ValueError(f"Unsupported aux target type: {aux_target_type}")
    if viseme_variant not in VISEME_VARIANTS:
        raise ValueError(f"Unsupported viseme variant: {viseme_variant}")
    return VISEME_VARIANTS[viseme_variant].get(normalized, normalized)


def build_aux_vocab(
    phoneme_paths: list[str | Path],
    aux_target_type: str = "phoneme",
    viseme_variant: str = "viseme_balanced_10",
) -> dict[str, int]:
    labels = {PHONEME_PAD, PHONEME_UNK}
    for raw_path in phoneme_paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        for _, _, label in read_alignment(path):
            labels.add(project_aux_label(label, aux_target_type=aux_target_type, viseme_variant=viseme_variant))

    ordered = [PHONEME_PAD, PHONEME_UNK]
    projected_sil = project_aux_label(PHONEME_SIL, aux_target_type=aux_target_type, viseme_variant=viseme_variant)
    if projected_sil in labels:
        ordered.append(projected_sil)
    ordered.extend(sorted(label for label in labels if label not in set(ordered)))
    return {label: index for index, label in enumerate(ordered)}
