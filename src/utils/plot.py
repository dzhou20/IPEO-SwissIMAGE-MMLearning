from pathlib import Path
import matplotlib.pyplot as plt


def plot_loss(
    epochs,
    train_losses,
    val_losses,
    output_dir,
    filename="loss_vs_epoch.png",
):
    if not train_losses and not val_losses:
        return

    plt.figure(figsize=(6, 4))
    if train_losses:
        plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    if val_losses:
        plt.plot(epochs, val_losses, marker="s", label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / filename, dpi=300)
    plt.close()


def plot_accuracy(
    epochs,
    train_accs,
    val_accs,
    output_dir,
    filename="accuracy_vs_epoch.png",
):
    if not train_accs and not val_accs:
        return

    plt.figure(figsize=(6, 4))
    if train_accs:
        plt.plot(epochs, train_accs, marker="o", label="Train Accuracy")
    if val_accs:
        plt.plot(epochs, val_accs, marker="s", label="Val Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / filename, dpi=300)
    plt.close()


def plot_f1(
    epochs,
    macro_f1s=None,
    micro_f1s=None,
    output_dir=None,
    filename="val_f1_vs_epoch.png",
):
    if not macro_f1s and not micro_f1s:
        return

    plt.figure(figsize=(6, 4))
    if macro_f1s:
        plt.plot(epochs, macro_f1s, marker="o", label="Macro-F1")
    if micro_f1s:
        plt.plot(epochs, micro_f1s, marker="s", label="Micro-F1")

    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Validation F1 vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / filename, dpi=300)
    plt.close()

def plot_all(
    epochs,
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    val_macro_f1s,
    output_dir,
    filename="all_indices_vs_epoch.png",
):

    if not epochs:
        return

    plt.figure(figsize=(8, 5))

    # ---- Loss ----
    if train_losses:
        plt.plot(
            epochs,
            train_losses,
            linestyle="-",
            marker="o",
            label="Train Loss",
        )
    if val_losses:
        plt.plot(
            epochs,
            val_losses,
            linestyle="--",
            marker="s",
            label="Val Loss",
        )

    # ---- Accuracy ----
    if train_accs:
        plt.plot(
            epochs,
            train_accs,
            linestyle="-",
            marker="^",
            label="Train Accuracy",
        )
    if val_accs:
        plt.plot(
            epochs,
            val_accs,
            linestyle="--",
            marker="v",
            label="Val Accuracy",
        )

    # ---- F1 ----
    if val_macro_f1s:
        plt.plot(
            epochs,
            val_macro_f1s,
            linestyle="-.",
            marker="d",
            label="Val Macro-F1",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Training and Validation Metrics")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    plt.savefig(Path(output_dir) / filename, dpi=300)
    plt.close()

