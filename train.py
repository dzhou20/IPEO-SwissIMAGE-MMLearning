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
from src.data.dataloaders import build_dataloaders
from src.models.fusion import TabularOnlyModel, ImageOnlyModel, EarlyFusionModel, GatedFusionModel, LateFusionModel
from src.train.engine import train_one_epoch, evaluate, run_one_epoch
from src.train.state_control import TrainingState
from src.utils.seed import set_seed
from src.utils.visualize import save_confusion_matrix, save_normalized_confusion_matrix
from src.utils.plot import plot_all
from args import parse_args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.mode == "fusion" and not args.group:
        raise ValueError("Fusion mode requires --group (e.g., --group bioclim).")
        
    # ----------------------------- Data -----------------------------
    image_size = (args.image_size, args.image_size)
    
    train_loader, val_loader, test_loader, group_cols, weights = build_dataloaders(
        mode=args.mode,
        group=args.group,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
    )

    
    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    
    # ------------------------- Model -------------------------
    if args.mode == "tabular": 
        model = TabularOnlyModel(tabular_dim=len(group_cols))
    elif args.mode == "image":
        model = ImageOnlyModel(args.backbone, args.pretrained, image_size=image_size)
    else:
        model = EarlyFusionModel(
            args.backbone,
            args.pretrained,
            tabular_dim=len(group_cols),
            image_size=image_size,
        )
        # model = GatedFusionModel(
        #     backbone=args.backbone,
        #     pretrained=args.pretrained,
        #     tabular_dim=len(group_cols), 
        #     image_size=image_size
        # )
        # model = LateFusionModel(
        #     args.backbone,
        #     args.pretrained,
        #     tabular_dim=len(group_cols),
        #     image_size=image_size,
        # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if device.type == "cuda":
        try:
            subprocess.run(["nvidia-smi"], check=False)
        except FileNotFoundError:
            print(f"[info] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[info] Using CPU")
    '''
    # --------------------- if Freeze encoder  ---------------------
    for p in model.encoder.parameters():
        p.requires_grad = False
    print("[info] Stage 1: encoder frozen, training head only")
    '''

    # --------------- Loss / Optimizer / Scheduler --------------
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if "encoder" in name or "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW([
    #     {'params': backbone_params, 'lr': 1e-5}, 
    #     {'params': other_params, 'lr': args.lr}     
    # ],  weight_decay=args.weight_decay)
    
    ''' 
    # optimizer for freeze-unfreeze training
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    '''

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',       
        factor=0.1,       
        patience=5
    )


    # ---------------------- State ----------------------
    state = TrainingState(args.patience, args.min_delta)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }
    # ---------------------- Resume ----------------------
    start_epoch = 1

    if args.resume is not None:
        resume_path = Path(args.resume)
        ckpt = torch.load(resume_path, map_location=device)
    
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
    
        # restore training state
        state.best_val_f1 = ckpt["training_state"]["best_val_f1"]
        state.best_val_loss = ckpt["training_state"]["best_val_loss"]
        state.no_improve_epochs = ckpt["training_state"]["no_improve_epochs"]
    
        start_epoch = ckpt["epoch"] + 1

        ckpt_dir = resume_path.parent
        run_dir = ckpt_dir.parent
    
        print(
            f"[Resume] Loaded from {resume_path}\n"
            f"         start_epoch={start_epoch}, "
            f"best_f1={state.best_val_f1:.4f}\n"
            f"         run_dir={run_dir}"
        )

    else:
        # start from 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = args.run_name or f"{args.backbone}_{args.mode}_{timestamp}"
        run_dir = Path(OUTPUT_DIR) / run_name
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
    # ---------------------- metrics ----------------------
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "epoch",
                "lr",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "val_macro_f1",
                "val_micro_f1",
                "best_ckpt",
                "last_ckpt",
            ])

    # ===================================================
    # Training loop
    # ===================================================
    
    for epoch in range(start_epoch, args.epochs + 1):
        
        '''
        # ---- staged unfreeze ----
        if epoch >= args.freeze_epochs:
            for name, p in model.encoder.named_parameters():
                if name.startswith(args.unfreeze_layer):
                    p.requires_grad = True

        if epoch >= args.encoder_lr_drop_epoch:
            optimizer.param_groups[0]["lr"] = args.encoder_lr_after
        ''' 
        # run one epoch (train and val)
        metrics = run_one_epoch(
            model,
            loaders,
            NUM_CLASSES,
            optimizer,
            criterion,
            device,
            args.mode,
            epoch,
            scheduler,
        )
        
        print(
            f"Epoch {epoch:03d} | "
            f"train_acc={metrics['train_acc']:.4f} | "
            f"train_loss={metrics['train_loss']:.4f} | "
            f"val_acc={metrics['val_acc']:.4f} | "
            f"val_loss={metrics['val_loss']:.4f} | "
            f"val_macro_f1={metrics['val_macro_f1']:.4f}"
        )
        
        # record state
        improved = state.update(
            metrics["val_loss"], metrics["val_macro_f1"]
        )

        best_ckpt_path = ""
        last_ckpt_path = ""
        
        # update best model
        if improved:
            best_ckpt_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "training_state": {
                        "best_val_f1": state.best_val_f1,
                        "best_val_loss": state.best_val_loss,
                        "no_improve_epochs": state.no_improve_epochs,
                    },
                },
                best_ckpt_path,
            )
            print("[Best] checkpoint saved")
            
        # save checkpoint every 5 epochs    
        if epoch % 5 == 0:
            last_ckpt_path = ckpt_dir / f"last_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "training_state": {
                        "best_val_f1": state.best_val_f1,
                        "best_val_loss": state.best_val_loss,
                        "no_improve_epochs": state.no_improve_epochs,
                    },
                },
                last_ckpt_path,
            )

        # record history
        for k in history:
            history[k].append(metrics[k])

        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch,
                metrics["lr"],
                metrics["train_loss"],
                metrics["train_acc"],
                metrics["val_loss"],
                metrics["val_acc"],
                metrics["val_macro_f1"],
                metrics["val_micro_f1"],
                best_ckpt_path,
                last_ckpt_path,
            ])

        if (
            state.no_improve_epochs >= args.patience
            and epoch >= args.min_epochs
        ):
            print("[info] Early stopping triggered")
            break

    # ===================================================
    # Test best model
    # ===================================================
    best_path = ckpt_dir / "best.pt"
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    test_metrics, _, y_true, y_pred, per_class = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        mode=args.mode,
        num_classes=NUM_CLASSES,
    )

    print(
        f"Test | acc={test_metrics['accuracy']:.4f} | "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )

    labels = [eunis_id_to_lab[i] for i in range(NUM_CLASSES)]
    save_confusion_matrix(y_true, y_pred, labels, run_dir)
    save_normalized_confusion_matrix(y_true, y_pred, labels, run_dir)

    pd.DataFrame({
        "class_id": range(NUM_CLASSES),
        "class_name": labels,
        "f1": per_class.numpy(),
    }).to_csv(run_dir / "per_class_f1.csv", index=False)

    plot_all(
        list(range(1, len(history["train_loss"]) + 1)),
        history["train_loss"],
        history["val_loss"],
        history["train_acc"],
        history["val_acc"],
        history["val_macro_f1"],
        run_dir,
    )

if __name__ == "__main__":
    main()
