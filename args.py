# args.py
from __future__ import annotations
import argparse
from config import IMAGE_SIZE

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # ---------- basic ----------
    parser.add_argument("--mode", choices=["image", "fusion"], default="image")
    parser.add_argument("--group", default=None)
    parser.add_argument("--backbone", choices=[
        "resnet18", "vit", "convnext_tiny", "efficientnet_b0"
    ], default="resnet18")

    # ---------- training ----------
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)

    #---------- freeze-unfreeze training ----------
    parser.add_argument("--encoder_lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--freeze_epochs", type=int, default=10)
    parser.add_argument("--encoder_lr_drop_epoch", type=int, default=20)
    parser.add_argument("--encoder_lr_after", type=float, default=1e-5)

    # ---------- system ----------
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--pretrained", action="store_true")

    # ---------- resume / run ----------
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--run_name", default=None)

    return parser

def parse_args():
    return build_parser().parse_args()
