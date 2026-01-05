from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import csv

import torch
import pandas as pd

from config import IMAGE_SIZE, NUM_CLASSES, OUTPUT_DIR
from eunis_labels import eunis_id_to_lab
from eunis_labels_mapping import level1_id_to_level2_ids, level1_id_to_name, num_level1_classes
from src.data.dataloaders import build_dataloaders
from src.models.fusion import (
    ImageOnlyModel,
    EarlyFusionModel,
    TabularOnlyModel,
    TwoStageImageModel,
    TwoStageEarlyFusionModel,
    TwoStageTabularModel,
)
from src.train.engine import train_one_epoch, evaluate
from src.utils.seed import set_seed
from src.utils.visualize import save_confusion_matrix, save_normalized_confusion_matrix
from src.utils.plot import plot_all

# ----------------------------- Argument parser -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["image", "fusion", "tabular"], default="image")
    parser.add_argument(
        "--group",
        default=None,
        help="SWECO group name, required for fusion.",
    )
    parser.add_argument("--backbone", choices=["resnet18", "vit"], default="resnet18")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    # the default training epoch is set to 10 for quick testing
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights.",
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume.")
    parser.add_argument("--two_stage", action="store_true", help="Enable Level 1 + Level 2 two-stage heads.")
    parser.add_argument("--level1_weight", type=float, default=0.3, help="Loss weight for Level 1 head.")
    parser.add_argument(
        "--freeze_backbone_ratio",
        type=float,
        default=0.0,
        help="Freeze the first ratio of backbone parameters (image/fusion only).",
    )
    parser.add_argument(
        "--label_level",
        choices=[1, 2],
        type=int,
        default=2,
        help="Train on Level 2 (default) or Level 1 labels.",
    )
    
    return parser.parse_args()

# ----------------------------- Main -----------------------------
def main() -> None:
    args = parse_args()
    args.pretrained = not args.no_pretrained
    set_seed(args.seed)

    if args.mode == "fusion" and not args.group:
        raise ValueError("Fusion mode requires --group (e.g., --group bioclim).")
    if args.mode == "tabular" and not args.group:
        raise ValueError("Tabular mode requires --group (e.g., --group bioclim).")
    if not 0.0 <= args.level1_weight <= 1.0:
        raise ValueError("--level1_weight must be between 0 and 1.")
    if not 0.0 <= args.freeze_backbone_ratio <= 1.0:
        raise ValueError("--freeze_backbone_ratio must be between 0 and 1.")
    if args.label_level == 1 and args.two_stage:
        raise ValueError("--label_level 1 is not compatible with --two_stage.")
        
    # ----------------------------- Load Data -----------------------------
    image_size = (args.image_size, args.image_size)
    train_loader, val_loader, test_loader, group_cols, weights = build_dataloaders(
        mode=args.mode,
        group=args.group,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
        include_level1=args.two_stage,
        label_level=args.label_level,
    )
    
    # ------------------------- Model -------------------------
    if args.two_stage:
        if args.mode == "image":
            model = TwoStageImageModel(
                args.backbone,
                args.pretrained,
                image_size=image_size,
                num_level1_classes=num_level1_classes,
            )
        elif args.mode == "tabular":
            tabular_dim = len(group_cols)
            model = TwoStageTabularModel(
                tabular_dim=tabular_dim,
                num_level1_classes=num_level1_classes,
            )
        else:
            tabular_dim = len(group_cols)
            model = TwoStageEarlyFusionModel(
                args.backbone,
                args.pretrained,
                tabular_dim=tabular_dim,
                image_size=image_size,
                num_level1_classes=num_level1_classes,
            )
    else:
        if args.mode == "image":
            model = ImageOnlyModel(args.backbone, args.pretrained, image_size=image_size)
        elif args.mode == "tabular":
            tabular_dim = len(group_cols)
            model = TabularOnlyModel(tabular_dim=tabular_dim)
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

    if args.freeze_backbone_ratio > 0.0:
        if args.mode == "tabular":
            print("[info] freeze_backbone_ratio ignored for tabular mode.")
        else:
            if not args.pretrained:
                print("[warn] Freezing backbone without pretrained weights may reduce performance.")
            backbone = model.encoder
            params = list(backbone.parameters())
            if params:
                freeze_count = int(len(params) * args.freeze_backbone_ratio)
                for param in params[:freeze_count]:
                    param.requires_grad = False
                total = len(params)
                print(
                    f"[info] Frozen {freeze_count}/{total} backbone parameter tensors "
                    f"({args.freeze_backbone_ratio:.2f} ratio)."
                )

    # ------------------------- Loss / Optimizer / Scheduler -------------------------
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion_l1 = torch.nn.CrossEntropyLoss() if args.two_stage else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',       
        factor=0.1,       
        patience=5
    )
    
    # ------------------------- Training state -------------------------
    # params for best model and early stopping
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    patience = 10
    min_delta = 1e-4
    no_improve_epochs = 0
    min_epochs = 20 
    start_epoch = 1
    
    # ------------------------- Resume -------------------------
    if args.resume:
        resume_path = Path(args.resume)
        ckpt_dir = resume_path.parent
        run_dir = ckpt_dir.parent
        
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
    
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        best_val_f1 = ckpt["best_val_f1"]
        no_improve_epochs = ckpt["no_improve_epochs"]

        print(
            f"[info] Resumed from epoch {ckpt['epoch']} | "
            f"best_val_loss={best_val_loss:.4f}, best_val_f1={best_val_f1:.4f}"
        )
        print(f"[info] Continue in run: {run_dir}")

    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.run_name:
            run_name = args.run_name
        elif args.two_stage:
            if args.mode == "image":
                run_name = f"{args.backbone}_image_two_stage_{timestamp}"
            elif args.mode == "tabular":
                run_name = f"tabular_{args.group}_two_stage_{timestamp}"
            else:
                run_name = f"{args.backbone}_fusion_{args.group}_two_stage_{timestamp}"
        elif args.mode == "image":
            run_name = f"{args.backbone}_image_{timestamp}"
        elif args.mode == "tabular":
            run_name = f"tabular_{args.group}_{timestamp}"
        else:
            run_name = f"{args.backbone}_fusion_{args.group}_{timestamp}"
    
        run_dir = Path(OUTPUT_DIR) / run_name
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
    metrics_path = run_dir / "metrics.csv"
    
    # ------------------------- Metrics -------------------------
    if not metrics_path.exists():
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            headers = [
                "epoch",
                "lr",
                "weight_decay",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "val_macro_f1",
                "val_micro_f1",
            ]
            if args.two_stage:
                headers += ["val_l1_accuracy", "val_l1_macro_f1", "val_l1_micro_f1"]
            headers += ["best_checkpoint", "last_checkpoint"]
            writer.writerow(headers)
    # ------------------------- History -------------------------
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []
    val_macro_f1s = []
    val_micro_f1s = []

    # ------------------------- Training loop -------------------------
    if args.label_level == 1:
        num_classes = num_level1_classes
    else:
        num_classes = NUM_CLASSES
    for epoch in range(start_epoch, args.epochs + 1):
        best_ckpt_path = ""
        last_ckpt_path = ""
        
        # 1. train
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            criterion_l1,
            device,
            mode=args.mode,
            two_stage=args.two_stage,
            level1_weight=args.level1_weight,
        )
        # 2. val
        if args.two_stage:
            (
                val_metrics,
                val_metrics_l1,
                val_loss,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = evaluate(
                model,
                val_loader,
                criterion,
                criterion_l1,
                device,
                mode=args.mode,
                num_classes=num_classes,
                two_stage=True,
                num_level1_classes=num_level1_classes,
                level1_id_to_level2_ids=level1_id_to_level2_ids,
                level1_weight=args.level1_weight,
            )
        else:
            val_metrics, val_loss, _, _, _ = evaluate(
                model, val_loader, criterion, criterion_l1, device, mode=args.mode, num_classes=num_classes
            )

        scheduler.step(val_metrics["macro_f1"])

        # print result of current epoch
        if args.two_stage:
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
                f"val_l1_macro_f1={val_metrics_l1['macro_f1']:.4f}"
            )
        else:
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )
        
        # 3. best model (F1)
        val_macro_f1 = val_metrics["macro_f1"]

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_ckpt_path = str(ckpt_dir / "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_val_f1": best_val_f1,
                    "no_improve_epochs": no_improve_epochs,
                },
                best_ckpt_path,
            )
            print(f"[Best Model] Saved successfully: {best_ckpt_path}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # 4. last checkpoint
        # save checkpoint every 5 epochs 
        ckpt_interval = 5 
        if epoch % ckpt_interval == 0:
            last_ckpt_path = ckpt_dir / f"last_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_val_f1": best_val_f1,
                    "no_improve_epochs": no_improve_epochs,
                },
                last_ckpt_path,
            )
            print(f"[Checkpoint] Saved successfully: {last_ckpt_path}")

        # 5. save into CSV 
        
        # add into history   
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_metrics["accuracy"])
        val_macro_f1s.append(val_metrics["macro_f1"])
        val_micro_f1s.append(val_metrics["micro_f1"])

        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            row = [
                epoch,
                optimizer.param_groups[0]["lr"],
                args.weight_decay,
                f"{train_loss:.6f}",
                f"{train_acc:.6f}",
                f"{val_loss:.6f}",
                f"{val_metrics['accuracy']:.6f}",
                f"{val_metrics['macro_f1']:.6f}",
                f"{val_metrics['micro_f1']:.6f}",
            ]
            if args.two_stage:
                row += [
                    f"{val_metrics_l1['accuracy']:.6f}",
                    f"{val_metrics_l1['macro_f1']:.6f}",
                    f"{val_metrics_l1['micro_f1']:.6f}",
                ]
            row += [best_ckpt_path, last_ckpt_path]
            writer.writerow(row)

        # 6. execute early stopping
        if no_improve_epochs >= patience and epoch >= min_epochs:
            # final safety save
            last_ckpt_path = ckpt_dir / f"last_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_val_f1": best_val_f1,
                    "no_improve_epochs": no_improve_epochs,
                },
                last_ckpt_path,
            )
            print(f"[Checkpoint] Saved successfully: {last_ckpt_path}")
            print(
                f"[info] Early stopping triggered at epoch {epoch}. "
                f"Best val loss: {best_val_loss:.4f}, "
                f"Best val macro-F1: {best_val_f1:.4f}"
            )
            break
                  
            
    # load best model to test
    best_path = ckpt_dir / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"best.pt not found in {ckpt_dir}")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if args.two_stage:
        (
            test_metrics,
            test_metrics_l1,
            _,
            y_true,
            y_pred,
            y_true_l1,
            y_pred_l1,
            per_class,
            per_class_l1,
        ) = evaluate(
            model,
            test_loader,
            criterion,
            criterion_l1,
            device,
            mode=args.mode,
            num_classes=num_classes,
            two_stage=True,
            num_level1_classes=num_level1_classes,
            level1_id_to_level2_ids=level1_id_to_level2_ids,
            level1_weight=args.level1_weight,
        )
        print(
            f"Test L2 | acc={test_metrics['accuracy']:.4f} | "
            f"macro_f1={test_metrics['macro_f1']:.4f} | micro_f1={test_metrics['micro_f1']:.4f}"
        )
        print(
            f"Test L1 | acc={test_metrics_l1['accuracy']:.4f} | "
            f"macro_f1={test_metrics_l1['macro_f1']:.4f} | micro_f1={test_metrics_l1['micro_f1']:.4f}"
        )
    else:
        test_metrics, _, y_true, y_pred, per_class = evaluate(
            model, test_loader, criterion, criterion_l1, device, mode=args.mode, num_classes=num_classes
        )
        print(
            f"Test | acc={test_metrics['accuracy']:.4f} | "
            f"macro_f1={test_metrics['macro_f1']:.4f} | micro_f1={test_metrics['micro_f1']:.4f}"
        )

    if args.label_level == 1:
        labels = [level1_id_to_name[i] for i in range(num_level1_classes)]
        cm_prefix = "confusion_matrix_level1"
        f1_path = run_dir / "per_class_f1_level1.csv"
    else:
        labels = [eunis_id_to_lab[i] for i in range(NUM_CLASSES)]
        cm_prefix = "confusion_matrix"
        f1_path = run_dir / "per_class_f1.csv"

    save_confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        output_dir=str(run_dir),
        filename_prefix=cm_prefix,
    )
    save_normalized_confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        output_dir=str(run_dir),
        filename_prefix=f"{cm_prefix}_normalized",
    )

    per_class_df = pd.DataFrame(
        {"class_id": list(range(len(labels))), "class_name": labels, "f1": per_class.numpy()}
    )
    per_class_df.to_csv(f1_path, index=False)

    if args.two_stage:
        level1_labels = [level1_id_to_name[i] for i in range(num_level1_classes)]
        save_confusion_matrix(
            y_true_l1,
            y_pred_l1,
            labels=level1_labels,
            output_dir=str(run_dir),
            filename_prefix="confusion_matrix_level1",
        )
        save_normalized_confusion_matrix(
            y_true_l1,
            y_pred_l1,
            labels=level1_labels,
            output_dir=str(run_dir),
            filename_prefix="confusion_matrix_level1_normalized",
        )
        per_class_l1_df = pd.DataFrame(
            {
                "class_id": list(range(num_level1_classes)),
                "class_name": level1_labels,
                "f1": per_class_l1.numpy(),
            }
        )
        per_class_l1_df.to_csv(run_dir / "per_class_f1_level1.csv", index=False)

    # plot accuracy/loss/f1 score in one figure
    epochs = list(range(1, len(train_losses) + 1))  
    plot_all(
        epochs,
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        val_macro_f1s,
        run_dir,
    )

if __name__ == "__main__":
    main()
