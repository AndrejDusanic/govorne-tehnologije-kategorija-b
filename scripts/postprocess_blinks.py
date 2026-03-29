from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blendshape_project.blink_postprocess import apply_random_blinks  # noqa: E402
from blendshape_project.constants import DEFAULT_FPS  # noqa: E402
from blendshape_project.io_utils import read_blendshape_csv, save_json, write_blendshape_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Add reproducible random blink events to existing blendshape CSV files.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--blink-seed", type=int, default=1337)
    parser.add_argument("--blink-strength", type=float, default=1.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(args.input_dir.glob("*.csv"))
    meta = {
        "system": {
            "source_dir": str(args.input_dir),
            "fps_out": args.fps,
            "random_blinks": True,
            "blink_seed": args.blink_seed,
            "blink_strength": args.blink_strength,
        },
        "files": {},
    }

    for csv_path in csv_paths:
        values = read_blendshape_csv(csv_path)
        blinked, blink_info = apply_random_blinks(
            values,
            fps=args.fps,
            seed=args.blink_seed,
            file_key=csv_path.stem,
            strength=args.blink_strength,
            return_info=True,
        )
        output_csv = args.output_dir / csv_path.name
        write_blendshape_csv(output_csv, blinked)

        copied_wav = None
        if args.audio_dir is not None:
            wav_path = args.audio_dir / f"{csv_path.stem}.wav"
            if wav_path.exists():
                copied_wav = args.output_dir / wav_path.name
                shutil.copy2(wav_path, copied_wav)

        meta["files"][csv_path.name] = {
            "csv_path": output_csv.name,
            "wav_path": copied_wav.name if copied_wav is not None else None,
            "blink_postprocess": blink_info,
        }

    save_json(args.output_dir / "meta.json", meta)
    print(f"Blink post-processing completed for {len(csv_paths)} files.")


if __name__ == "__main__":
    main()
