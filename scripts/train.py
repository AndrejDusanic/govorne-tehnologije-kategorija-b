from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blendshape_project.constants import BLENDSHAPE_NAMES, SPEAKER_ORDER, blendshape_priority_weights  # noqa: E402
from blendshape_project.data import (  # noqa: E402
    AudioFeatureExtractor,
    BlendshapeDataset,
    DatasetStats,
    collate_batch,
    compute_dataset_stats,
)
from blendshape_project.io_utils import load_json, save_json, set_seed  # noqa: E402
from blendshape_project.model import BlendshapeRegressor  # noqa: E402
from blendshape_project.train_utils import (  # noqa: E402
    compute_losses,
    evaluate_model,
    save_history_plot,
    save_overlay_plot,
    save_per_blendshape_plot,
)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the competition baseline model.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "natural_samples.csv")
    parser.add_argument("--split-json", type=Path, default=ROOT / "data" / "manifests" / "split.json")
    parser.add_argument("--phoneme-vocab", type=Path, default=ROOT / "data" / "manifests" / "phoneme_vocab.json")
    parser.add_argument("--run-name", type=str, default="baseline_causal")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--phoneme-loss-weight", type=float, default=0.2)
    parser.add_argument("--temporal-loss-weight", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit-train", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    run_dir = ROOT / "artifacts" / "checkpoints" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.manifest)
    split = load_json(args.split_json)
    phoneme_vocab = load_json(args.phoneme_vocab)
    speaker_to_id = {speaker: idx for idx, speaker in enumerate(SPEAKER_ORDER)}

    train_df = frame[frame["sample_id"].isin(split["train"])].copy()
    val_df = frame[frame["sample_id"].isin(split["val"])].copy()
    if args.limit_train > 0:
        train_df = train_df.head(args.limit_train).copy()
        val_df = val_df.head(max(1, min(len(val_df), max(1, args.limit_train // 4)))).copy()

    feature_extractor = AudioFeatureExtractor()
    stats = compute_dataset_stats(train_df.to_dict("records"), feature_extractor)
    save_json(run_dir / "stats.json", stats.to_json())

    train_dataset = BlendshapeDataset(train_df, feature_extractor, stats=stats, phoneme_vocab=phoneme_vocab, speaker_to_id=speaker_to_id)
    val_dataset = BlendshapeDataset(val_df, feature_extractor, stats=stats, phoneme_vocab=phoneme_vocab, speaker_to_id=speaker_to_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    model = BlendshapeRegressor(
        input_dim=feature_extractor.feature_dim,
        num_blendshapes=len(BLENDSHAPE_NAMES),
        num_speakers=len(speaker_to_id),
        num_phonemes=len(phoneme_vocab),
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    coefficient_weights = torch.tensor(blendshape_priority_weights(), device=device, dtype=torch.float32)

    history: list[dict[str, float]] = []
    best_val_mae = float("inf")
    best_metrics = None
    checkpoint_config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_reg = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            features = batch["features"].to(device)
            speaker_ids = batch["speaker_ids"].to(device)
            targets = batch["targets"].to(device)
            target_mask = batch["target_mask"].to(device)
            phoneme_ids = batch["phoneme_ids"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(features, speaker_ids)
            losses = compute_losses(
                outputs,
                {"targets": targets, "target_mask": target_mask, "phoneme_ids": phoneme_ids},
                phoneme_loss_weight=args.phoneme_loss_weight,
                temporal_loss_weight=args.temporal_loss_weight,
                coefficient_weights=coefficient_weights,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(losses["loss"].item())
            running_reg += float(losses["regression_loss"].item())
            progress.set_postfix(loss=f"{losses['loss'].item():.4f}", reg=f"{losses['regression_loss'].item():.4f}")

        scheduler.step()
        metrics = evaluate_model(model, val_loader, device, stats)
        epoch_log = {
            "epoch": epoch,
            "train_loss": running_loss / max(len(train_loader), 1),
            "train_regression_loss": running_reg / max(len(train_loader), 1),
            "val_mae": float(metrics["mae"]),
            "val_rmse": float(metrics["rmse"]),
            "val_mouth_mae": float(metrics["mouth_mae"]),
            "val_jaw_open_mae": float(metrics["jaw_open_mae"]),
        }
        history.append(epoch_log)

        checkpoint = {
            "model_state": model.state_dict(),
            "config": checkpoint_config,
            "stats": stats.to_json(),
            "speaker_to_id": speaker_to_id,
            "phoneme_vocab": phoneme_vocab,
            "blendshape_names": BLENDSHAPE_NAMES,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if metrics["mae"] < best_val_mae:
            best_val_mae = float(metrics["mae"])
            best_metrics = metrics
            torch.save(checkpoint, run_dir / "best.pt")

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": round(epoch_log["train_loss"], 5),
                    "val_mae": round(epoch_log["val_mae"], 5),
                    "val_rmse": round(epoch_log["val_rmse"], 5),
                    "val_mouth_mae": round(epoch_log["val_mouth_mae"], 5),
                },
                ensure_ascii=False,
            )
        )

    save_json(run_dir / "history.json", history)
    save_history_plot(history, run_dir / "training_curves.png")

    best_checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    best_metrics = evaluate_model(model, val_loader, device, stats)
    save_json(
        run_dir / "best_metrics.json",
        {
            "mae": float(best_metrics["mae"]),
            "rmse": float(best_metrics["rmse"]),
            "mouth_mae": float(best_metrics["mouth_mae"]),
            "jaw_open_mae": float(best_metrics["jaw_open_mae"]),
            "per_dim_mae": best_metrics["per_dim_mae"],
        },
    )
    save_per_blendshape_plot(best_metrics["per_dim_mae"], run_dir / "val_per_blendshape_mae.png")
    if best_metrics["samples"]:
        sample_payload = best_metrics["samples"][0]
        save_overlay_plot(
            prediction=sample_payload["prediction"],
            target=sample_payload["target"],
            output_path=run_dir / f"{sample_payload['sample_id']}_overlay.png",
            title=f"Prediction overlay for {sample_payload['sample_id']}",
        )


if __name__ == "__main__":
    main()
