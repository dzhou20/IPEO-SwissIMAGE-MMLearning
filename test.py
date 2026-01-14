from __future__ import annotations

from pathlib import Path
import torch
import pandas as pd
import csv

from config import NUM_CLASSES
from eunis_labels import eunis_id_to_lab
from src.data.dataloaders import build_dataloaders
from src.models.fusion import (
    TabularOnlyModel,
    ImageOnlyModel,
    EarlyFusionModel,
    GatedFusionModel,
    LateFusionModel,
)
from src.train.engine import evaluate
from src.utils.seed import set_seed
from src.utils.visualize import (
    save_confusion_matrix,
    save_normalized_confusion_matrix,
)
from args import parse_args


def main() -> None:
    test_ckpt = "ablation_study_results/D1/checkpoints/best.pt"
    
    # ===================================================
    # Args & seed
    # ===================================================
    args = parse_args()
    args.backbone = 'convnext_tiny'
    args.mode = "fusion"
    args.group = "all"
    
    set_seed(args.seed)

    if test_ckpt is None:
        raise ValueError("Testing requires --ckpt pointing to best.pt")

    ckpt_path = Path(test_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # run_dir: outputs/run_xxx/checkpoints/best.pt → outputs/run_xxx
    run_dir = ckpt_path.parent.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    # ===================================================
    # Data
    # ===================================================
    image_size = (args.image_size, args.image_size)

    _, _, test_loader, group_cols, weights = build_dataloaders(
        mode=args.mode,
        group=args.group,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
    )

    # ===================================================
    # Model
    # ===================================================
    if args.mode == "tabular":
        model = TabularOnlyModel(tabular_dim=len(group_cols))
    elif args.mode == "image":
        model = ImageOnlyModel(
            args.backbone,
            args.pretrained,
            image_size=image_size,
        )
    else:
        model = EarlyFusionModel(
            args.backbone,
            args.pretrained,
            tabular_dim=len(group_cols),
            image_size=image_size,
        )
        # 如果你 test gated / late，只需要在这里换

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # ===================================================
    # Load checkpoint
    # ===================================================
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    # ===================================================
    # Loss (only for evaluate)
    # ===================================================
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ===================================================
    # Test
    # ===================================================
    with torch.no_grad():
        test_metrics, _, y_true, y_pred, per_class = evaluate(
            model,
            test_loader,
            criterion,
            device,
            mode=args.mode,
            num_classes=NUM_CLASSES,
        )

    print(
        f"[Test] acc={test_metrics['accuracy']:.4f} | "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )

    # ===================================================
    # Save test metrics (single run)
    # ===================================================

    metrics_df = pd.DataFrame([{
        "mode": args.mode,
        "group": args.group,
        "backbone": args.backbone,
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
    }])

    metrics_df.to_csv(
        run_dir / "test_metrics.csv",
        index=False,
    )

    # ===================================================
    # Save results
    # ===================================================
    labels = [eunis_id_to_lab[i] for i in range(NUM_CLASSES)]

    save_confusion_matrix(
        y_true,
        y_pred,
        labels,
        run_dir,
    )

    save_normalized_confusion_matrix(
        y_true,
        y_pred,
        labels,
        run_dir,
    )

    pd.DataFrame({
        "class_id": range(NUM_CLASSES),
        "class_name": labels,
        "f1": per_class.cpu().numpy(),
    }).to_csv(
        run_dir / "per_class_f1_test.csv",
        index=False,
    )

    # ===================================================
    # Append to global summary (for ablation comparison)
    # ===================================================

    summary_path = run_dir.parent / "summary_metrics.csv"

    write_header = not summary_path.exists()

    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run_name",
                "mode",
                "group",
                "backbone",
                "accuracy",
                "macro_f1",
            ])
        writer.writerow([
            run_dir.name,
            args.mode,
            args.group,
            args.backbone,
            f"{test_metrics['accuracy']:.6f}",
            f"{test_metrics['macro_f1']:.6f}",
        ])



if __name__ == "__main__":
    main()
