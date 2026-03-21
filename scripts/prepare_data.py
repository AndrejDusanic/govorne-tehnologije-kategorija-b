from __future__ import annotations

import argparse
import random
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
    SPEAKER_ORDER,
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

    ensure_extracted(ROOT / EXPECTED_RAW_FILES["spk08_zip"], extracted_root / "spk08_blendshapes")
    ensure_extracted(ROOT / EXPECTED_RAW_FILES["spk14_zip"], extracted_root / "spk14_blendshapes")
    ensure_extracted(ROOT / EXPECTED_RAW_FILES["labels_zip"], extracted_root / "labels_aligned")
    ensure_extracted(ROOT / EXPECTED_RAW_FILES["synth_zip"], extracted_root / "audio_synth")
    extract_avatar_metadata(ROOT / EXPECTED_RAW_FILES["avatar_zip"], data_root / "manifests" / "avatar_blendshape_names.txt")

    natural_records: list[dict[str, object]] = []
    synth_records: list[dict[str, object]] = []

    label_root = extracted_root / "labels_aligned" / "labels_aligned"
    phoneme_dir = label_root / "per_phoneme"
    word_dir = label_root / "per_word"

    for speaker in SPEAKER_ORDER:
        speaker_root = extracted_root / f"{speaker}_blendshapes"
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
    for speaker in SPEAKER_ORDER:
        transcripts = read_transcripts_xlsx(extracted_root / f"{speaker}_blendshapes" / f"{speaker}_transcript.xlsx")
        synth_paths = sorted_numeric_paths(list(synth_dir.glob(f"{speaker}_*.wav")))
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
    for speaker in SPEAKER_ORDER:
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
