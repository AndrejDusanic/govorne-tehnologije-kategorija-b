from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blendshape_project.constants import (  # noqa: E402
    BLENDSHAPE_NAMES,
    DEFAULT_FPS,
    DEFAULT_SPLIT_SEED,
    DEFAULT_VAL_FRACTION,
    EXPECTED_RAW_FILES,
    PHONEME_PAD,
    PHONEME_SIL,
    PHONEME_UNK,
    canonical_speaker_name,
    sort_speakers,
)
from blendshape_project.io_utils import (  # noqa: E402
    ensure_dir,
    read_alignment,
    read_wav_metadata,
    read_blendshape_csv,
    read_transcripts_xlsx,
    sample_number_from_name,
    save_json,
    sorted_numeric_paths,
)


def ensure_extracted(archive_path: Path, destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        return
    ensure_dir(destination)
    shutil.unpack_archive(str(archive_path), str(destination))


def sanitize_path_fragment(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def normalize_media_name(name: str, speaker: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix not in {".csv", ".wav"}:
        return name
    lowered = name.lower()
    if lowered.startswith(f"{speaker}_"):
        return name
    match = re.search(r"(\d+)", Path(name).stem)
    if not match:
        return name
    return f"{speaker}_{int(match.group(1)):03d}{suffix}"


def discover_speaker_archives(root: Path) -> dict[str, Path]:
    archives: dict[str, Path] = {}
    for archive_path in root.glob("*.zip"):
        stem_lower = archive_path.stem.lower()
        if "blendshape" not in stem_lower and "speaker_" not in stem_lower and not stem_lower.startswith("spk"):
            continue
        try:
            speaker = canonical_speaker_name(stem_lower)
        except ValueError:
            continue

        previous = archives.get(speaker)
        if previous is None:
            archives[speaker] = archive_path
            continue

        previous_stat = previous.stat()
        current_stat = archive_path.stat()
        if (current_stat.st_size, current_stat.st_mtime) > (previous_stat.st_size, previous_stat.st_mtime):
            archives[speaker] = archive_path

    return {speaker: archives[speaker] for speaker in sort_speakers(list(archives.keys()))}


def discover_labels_archive(root: Path) -> Path:
    candidates = [path for path in root.glob("labels_aligned*.zip") if path.is_file()]
    if not candidates:
        legacy = root / EXPECTED_RAW_FILES["labels_zip"]
        if legacy.exists():
            return legacy
        raise FileNotFoundError("Could not find any labels_aligned*.zip archive in the project root.")
    return max(candidates, key=lambda path: (path.stat().st_size, path.stat().st_mtime))


def resolve_alignment_root(extracted_labels_root: Path) -> Path:
    for per_phoneme_dir in extracted_labels_root.rglob("per_phoneme"):
        candidate_root = per_phoneme_dir.parent
        if (candidate_root / "per_word").exists():
            return candidate_root
    raise FileNotFoundError(f"Could not locate per_phoneme/per_word alignment folders under {extracted_labels_root}")


def normalize_speaker_extract(speaker_root: Path, speaker: str) -> None:
    renamed_dir = ensure_dir(speaker_root / f"renamed_{speaker}")
    transcript_out = speaker_root / f"{speaker}_transcript.xlsx"

    source_dirs = [path for path in speaker_root.rglob("*") if path.is_dir() and path.name.lower().endswith("_blendshapes_and_audio")]
    if not source_dirs and renamed_dir.exists():
        source_dirs = [renamed_dir]

    for source_dir in source_dirs:
        for item in source_dir.iterdir():
            if item.suffix.lower() not in {".csv", ".wav"}:
                continue
            target = renamed_dir / normalize_media_name(item.name, speaker)
            if not target.exists():
                shutil.copy2(item, target)

    transcript_candidates = [path for path in speaker_root.rglob("*_transcript.xlsx")]
    if not transcript_out.exists() and transcript_candidates:
        shutil.copy2(transcript_candidates[0], transcript_out)

    csv_count = len(list(renamed_dir.glob("*.csv")))
    wav_count = len(list(renamed_dir.glob("*.wav")))
    if csv_count == 0 or wav_count == 0:
        raise FileNotFoundError(f"Normalized folder {renamed_dir} does not contain paired CSV/WAV files for {speaker}.")
    if not transcript_out.exists():
        raise FileNotFoundError(f"Could not locate transcript XLSX for speaker {speaker} under {speaker_root}.")


def discover_speaker_roots(extracted_root: Path) -> dict[str, Path]:
    speaker_roots: dict[str, Path] = {}
    for path in extracted_root.glob("*_blendshapes"):
        if not path.is_dir():
            continue
        try:
            speaker = canonical_speaker_name(path.name)
        except ValueError:
            continue
        speaker_roots[speaker] = path
    return {speaker: speaker_roots[speaker] for speaker in sort_speakers(list(speaker_roots.keys()))}


def extract_avatar_metadata(archive_path: Path, output_path: Path) -> None:
    if output_path.exists():
        return
    with zipfile.ZipFile(archive_path) as archive:
        member = "FTNFacialRig 0.19/FTNFacialRig/Blendshape_Names.txt"
        output_path.write_text(archive.read(member).decode("utf-8"), encoding="utf-8")


def build_phoneme_vocab(phoneme_dir: Path) -> dict[str, int]:
    labels = {PHONEME_PAD, PHONEME_UNK, PHONEME_SIL}
    for path in sorted(phoneme_dir.glob("*.txt")):
        for _, _, label in read_alignment(path):
            labels.add(label)
    ordered = [PHONEME_PAD, PHONEME_UNK] + sorted(labels - {PHONEME_PAD, PHONEME_UNK})
    return {label: idx for idx, label in enumerate(ordered)}


def split_sample_ids(sample_ids: list[str], seed: int, val_fraction: float) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    ids = sample_ids[:]
    rng.shuffle(ids)
    val_count = max(1, int(round(len(ids) * val_fraction)))
    val_ids = sorted(ids[:val_count])
    train_ids = sorted(ids[val_count:])
    return train_ids, val_ids


def build_manifests(seed: int, val_fraction: float) -> dict[str, int]:
    data_root = ROOT / "data"
    extracted_root = data_root / "extracted"
    ensure_dir(extracted_root)
    ensure_dir(data_root / "manifests")

    speaker_archives = discover_speaker_archives(ROOT)
    for speaker, archive_path in speaker_archives.items():
        speaker_root = extracted_root / f"{speaker}_blendshapes"
        ensure_extracted(archive_path, speaker_root)
        normalize_speaker_extract(speaker_root, speaker)

    labels_archive = discover_labels_archive(ROOT)
    labels_extract_root = extracted_root / f"labels_{sanitize_path_fragment(labels_archive.stem)}"
    ensure_extracted(labels_archive, labels_extract_root)

    ensure_extracted(ROOT / EXPECTED_RAW_FILES["synth_zip"], extracted_root / "audio_synth")
    extract_avatar_metadata(ROOT / EXPECTED_RAW_FILES["avatar_zip"], data_root / "manifests" / "avatar_blendshape_names.txt")

    natural_records: list[dict[str, object]] = []
    synth_records: list[dict[str, object]] = []

    label_root = resolve_alignment_root(labels_extract_root)
    phoneme_dir = label_root / "per_phoneme"
    word_dir = label_root / "per_word"
    speaker_roots = discover_speaker_roots(extracted_root)
    speakers = sort_speakers(list(speaker_roots.keys()))

    for speaker in speakers:
        speaker_root = speaker_roots[speaker]
        transcript_path = speaker_root / f"{speaker}_transcript.xlsx"
        transcripts = read_transcripts_xlsx(transcript_path)
        renamed_dir = speaker_root / f"renamed_{speaker}"
        csv_paths = sorted_numeric_paths(list(renamed_dir.glob("*.csv")))

        for csv_path in csv_paths:
            number = sample_number_from_name(csv_path)
            wav_path = csv_path.with_suffix(".wav")
            target = read_blendshape_csv(csv_path)
            audio_info = read_wav_metadata(wav_path)
            sample_id = f"{speaker}_{number:03d}"
            natural_records.append(
                {
                    "sample_id": sample_id,
                    "speaker": speaker,
                    "sample_number": number,
                    "audio_path": str(wav_path.resolve()),
                    "blendshape_path": str(csv_path.resolve()),
                    "phoneme_path": str((phoneme_dir / f"{sample_id}.txt").resolve()),
                    "word_alignment_path": str((word_dir / f"{sample_id}.txt").resolve()),
                    "text": transcripts.get(number, ""),
                    "duration_sec": audio_info.duration_sec,
                    "n_frames": int(target.shape[0]),
                    "fps": DEFAULT_FPS,
                    "split_type": "natural",
                }
            )

    synth_dir = extracted_root / "audio_synth" / "synth"
    for speaker in speakers:
        transcript_path = speaker_roots[speaker] / f"{speaker}_transcript.xlsx"
        transcripts = read_transcripts_xlsx(transcript_path)
        synth_paths = sorted_numeric_paths(list(synth_dir.glob(f"{speaker}_*.wav")))
        if not synth_paths:
            continue
        numbers = [sample_number_from_name(path) for path in synth_paths]
        transcript_offset = 1 if numbers and min(numbers) == 0 and max(numbers) == len(transcripts) - 1 else 0
        for wav_path in synth_paths:
            raw_number = sample_number_from_name(wav_path)
            mapped_number = raw_number + transcript_offset
            audio_info = read_wav_metadata(wav_path)
            synth_records.append(
                {
                    "sample_id": wav_path.stem,
                    "speaker": speaker,
                    "sample_number": raw_number,
                    "mapped_transcript_number": mapped_number,
                    "audio_path": str(wav_path.resolve()),
                    "blendshape_path": "",
                    "phoneme_path": "",
                    "word_alignment_path": "",
                    "text": transcripts.get(mapped_number, ""),
                    "duration_sec": audio_info.duration_sec,
                    "n_frames": int(round(audio_info.duration_sec * DEFAULT_FPS)),
                    "fps": DEFAULT_FPS,
                    "split_type": "synth",
                }
            )

    natural_df = pd.DataFrame(natural_records).sort_values(["speaker", "sample_number"]).reset_index(drop=True)
    synth_df = pd.DataFrame(synth_records).sort_values(["speaker", "sample_number"]).reset_index(drop=True)
    all_df = pd.concat([natural_df, synth_df], ignore_index=True)

    split_payload: dict[str, list[str]] = {"train": [], "val": []}
    for speaker in speakers:
        speaker_ids = natural_df[natural_df["speaker"] == speaker]["sample_id"].tolist()
        train_ids, val_ids = split_sample_ids(speaker_ids, seed=seed, val_fraction=val_fraction)
        split_payload["train"].extend(train_ids)
        split_payload["val"].extend(val_ids)
    split_payload["train"] = sorted(split_payload["train"])
    split_payload["val"] = sorted(split_payload["val"])

    natural_df["split"] = natural_df["sample_id"].apply(lambda item: "val" if item in split_payload["val"] else "train")
    all_df = pd.concat([natural_df, synth_df], ignore_index=True)

    phoneme_vocab = build_phoneme_vocab(phoneme_dir)

    natural_df.to_csv(data_root / "manifests" / "natural_samples.csv", index=False)
    synth_df.to_csv(data_root / "manifests" / "synth_samples.csv", index=False)
    all_df.to_csv(data_root / "manifests" / "all_samples.csv", index=False)
    save_json(data_root / "manifests" / "split.json", split_payload)
    save_json(data_root / "manifests" / "phoneme_vocab.json", phoneme_vocab)
    save_json(
        data_root / "manifests" / "dataset_summary.json",
        {
            "n_natural": int(len(natural_df)),
            "n_synth": int(len(synth_df)),
            "n_train": int((natural_df["split"] == "train").sum()),
            "n_val": int((natural_df["split"] == "val").sum()),
            "blendshape_count": len(BLENDSHAPE_NAMES),
            "phoneme_count": len(phoneme_vocab),
            "fps": DEFAULT_FPS,
            "seed": seed,
            "val_fraction": val_fraction,
        },
    )
    return {
        "n_natural": int(len(natural_df)),
        "n_synth": int(len(synth_df)),
        "n_train": int((natural_df["split"] == "train").sum()),
        "n_val": int((natural_df["split"] == "val").sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract archives and build manifests for the competition dataset.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    args = parser.parse_args()

    summary = build_manifests(seed=args.seed, val_fraction=args.val_fraction)
    print("Prepared dataset manifests:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
