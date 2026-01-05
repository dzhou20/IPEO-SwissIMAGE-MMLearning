# ============================================================
# Training state (early stopping + best tracking)
# ============================================================
class TrainingState:
    def __init__(self, patience: int, min_delta: float):
        self.best_val_f1 = -1.0
        self.best_val_loss = float("inf")
        self.no_improve_epochs = 0
        self.patience = patience
        self.min_delta = min_delta

    def update(self, val_loss: float, val_f1: float) -> bool:
        improved = False

        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss

        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            self.no_improve_epochs = 0
            improved = True
        else:
            self.no_improve_epochs += 1

        return improved
