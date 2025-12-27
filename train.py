from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import csv

import torch
import pandas as pd
import matplotlib.pyplot as plt

from config import IMAGE_SIZE, NUM_CLASSES, OUTPUT_DIR
from eunis_labels import eunis_id_to_lab
from src.data.dataloaders import build_dataloaders
from src.models.fusion import ImageOnlyModel, EarlyFusionModel
from src.train.engine import train_one_epoch, evaluate
from src.utils.seed import set_seed
from src.utils.visualize import save_confusion_matrix, save_normalized_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["image", "fusion"], default="image")
    parser.add_argument(
        "--group",
        default=None,
        help="SWECO group name, required for fusion.",
    )
    parser.add_argument("--backbone", choices=["resnet18", "vit"], default="resnet18")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    # the default training epoch is set to 10 for quick testing
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE[0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.mode == "fusion" and not args.group:
        raise ValueError("Fusion mode requires --group (e.g., --group bioclim).")

    image_size = (args.image_size, args.image_size)
    train_loader, val_loader, test_loader, group_cols = build_dataloaders(
        mode=args.mode,
        group=args.group,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
    )

    if args.mode == "image":
        model = ImageOnlyModel(args.backbone, args.pretrained, image_size=image_size)
    else:
        tabular_dim = len(group_cols)
        model = EarlyFusionModel(
            args.backbone,
            args.pretrained,
            tabular_dim=tabular_dim,
            image_size=image_size,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cuda":
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                print(result.stdout.strip())
            if result.stderr:
                print(result.stderr.strip())
        except FileNotFoundError:
            print(
                f"[info] CUDA available but nvidia-smi not found. "
                f"Using GPU: {torch.cuda.get_device_name(0)}"
            )
    else:
        print("[info] CUDA not available; using CPU.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_name = args.run_name
    elif args.mode == "image":
        run_name = f"{args.backbone}_image_{timestamp}"
    else:
        run_name = f"{args.backbone}_fusion_{args.group}_{timestamp}"

    run_dir = Path(OUTPUT_DIR) / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"

    best_val = -1.0
    train_losses = []
    val_accs = []
    val_macro_f1s = []
    val_micro_f1s = []

    if not metrics_path.exists():
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_accuracy",
                    "val_macro_f1",
                    "val_micro_f1",
                    "best_checkpoint",
                    "last_checkpoint",
                ]
            )

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, mode=args.mode)
        val_metrics, _, _, _ = evaluate(
            model, val_loader, device, mode=args.mode, num_classes=NUM_CLASSES
        )

        val_score = val_metrics["macro_f1"]
        best_ckpt_path = ""
        if val_score > best_val:
            best_val = val_score
            best_ckpt_path = str(ckpt_dir / "best.pt")
            torch.save(model.state_dict(), best_ckpt_path)

        last_ckpt_path = str(ckpt_dir / "last.pt")
        torch.save(model.state_dict(), last_ckpt_path)
        train_losses.append(loss)
        val_accs.append(val_metrics["accuracy"])
        val_macro_f1s.append(val_metrics["macro_f1"])
        val_micro_f1s.append(val_metrics["micro_f1"])

        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{loss:.6f}",
                    f"{val_metrics['accuracy']:.6f}",
                    f"{val_metrics['macro_f1']:.6f}",
                    f"{val_metrics['micro_f1']:.6f}",
                    best_ckpt_path,
                    last_ckpt_path,
                ]
            )

        print(
            f"Epoch {epoch}/{args.epochs} | loss={loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
    test_metrics, y_true, y_pred, per_class = evaluate(
        model, test_loader, device, mode=args.mode, num_classes=NUM_CLASSES
    )

    print(
        f"Test | acc={test_metrics['accuracy']:.4f} | "
        f"macro_f1={test_metrics['macro_f1']:.4f} | micro_f1={test_metrics['micro_f1']:.4f}"
    )

    labels = [eunis_id_to_lab[i] for i in range(NUM_CLASSES)]
    save_confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        output_dir=str(run_dir),
        filename_prefix="confusion_matrix",
    )
    save_normalized_confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        output_dir=str(run_dir),
        filename_prefix="confusion_matrix_normalized",
    )

    per_class_df = pd.DataFrame(
        {"class_id": list(range(NUM_CLASSES)), "class_name": labels, "f1": per_class.numpy()}
    )
    per_class_df.to_csv(run_dir / "per_class_f1.csv", index=False)

    epochs = list(range(1, args.epochs + 1))
    if train_losses:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train Loss vs Epoch")
        plt.tight_layout()
        plt.savefig(run_dir / "loss_vs_epoch.png", dpi=300)
        plt.close()

    if val_accs:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, val_accs, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Val Accuracy vs Epoch")
        plt.tight_layout()
        plt.savefig(run_dir / "accuracy_vs_epoch.png", dpi=300)
        plt.close()

    if val_macro_f1s or val_micro_f1s:
        plt.figure(figsize=(6, 4))
        if val_macro_f1s:
            plt.plot(epochs, val_macro_f1s, marker="o", label="macro_f1")
        if val_micro_f1s:
            plt.plot(epochs, val_micro_f1s, marker="o", label="micro_f1")
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.title("Val F1 vs Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "val_f1_vs_epoch.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
