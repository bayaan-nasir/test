import torch
import torch.nn as nn
import torch_directml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

from models.pneumonia.model import build_model, CLASS_NAMES, NUM_CLASSES
from training.pneumonia.dataset import build_dataloaders
from core.config import settings
from core.logger import logger


# Hyperparameters 
HPARAMS = {
    "batch_size": 32,
    "num_epochs": 30,              # Back to 30 for full GPU training
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "dropout": 0.5,
    "patience": 7,                 # early stopping patience
    "num_workers": 4,              # Note: Lower to 2 or 0 if lab server hits memory limits
    "image_size": 224,
    "freeze_backbone_epochs": 3,   # freeze DenseNet features for first N epochs
}


class EarlyStopping:
    """Stops training if val_loss doesn't improve for `patience` epochs."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # don't stop
        self.counter += 1
        return self.counter >= self.patience


def freeze_backbone(model: nn.Module):
    """Freeze DenseNet feature extractor — only train the classifier head."""
    for param in model.features.parameters():
        param.requires_grad = False
    logger.info("Backbone frozen — training classifier head only")


def unfreeze_backbone(model: nn.Module):
    """Unfreeze all layers for full fine-tuning."""
    for param in model.features.parameters():
        param.requires_grad = True
    logger.info("Backbone unfrozen — full fine-tuning")


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    scaler: GradScaler,
    device: str,
) -> tuple[float, float]:
    """Returns (avg_loss, accuracy) for one training epoch."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(enabled=(device == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: str,
) -> tuple[float, float, dict]:
    """Returns (avg_loss, accuracy, metrics_dict) on validation/test set."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total

    # Dynamic class labels so sklearn doesn't crash on partial datasets
    unique_labels = np.unique(all_labels)
    current_target_names = [CLASS_NAMES[i] for i in unique_labels]

    report = classification_report(
        all_labels, 
        all_preds, 
        labels=unique_labels,
        target_names=current_target_names, 
        output_dict=True
    )

    # AUC-ROC (one-vs-rest for multi-class)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except Exception:
        auc = None

    metrics = {
        "accuracy": accuracy,
        "loss": avg_loss,
        "classification_report": report,
        "auc_roc": auc,
    }
    return avg_loss, accuracy, metrics


def save_checkpoint(model: nn.Module, epoch: int, val_loss: float, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "class_names": CLASS_NAMES,
    }, path)
    logger.info(f"Checkpoint saved → {path} (epoch {epoch}, val_loss={val_loss:.4f})")


def train():
    if torch_directml.is_available():
        device = torch_directml.device()
        logger.info(f"Training on AMD Radeon via DirectML: {device}")
    else:
        device = "cpu"
        logger.info(f"Training on device: {device}")

    # Data 
    data_dir = Path("data/raw/pneumonia")
    loaders = build_dataloaders(
        data_dir,
        batch_size=HPARAMS["batch_size"],
        num_workers=HPARAMS["num_workers"],
    )

    # Model 
    model = build_model().to(device)
    freeze_backbone(model)   # start with frozen backbone

    # Loss — weighted cross-entropy 
    class_weights = loaders["train"].dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    #  Optimiser and scheduler 
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=HPARAMS["learning_rate"],
        weight_decay=HPARAMS["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=HPARAMS["num_epochs"], eta_min=1e-6
    )
    scaler = GradScaler(enabled=(device == "cuda"))
    early_stop = EarlyStopping(patience=HPARAMS["patience"])

    best_model_path = settings.model_registry_dir / "pneumonia" / "best_model.pth"

    # MLflow 
    
    # Force MLflow to use a local 'mlruns' directory on the current drive
    # This prevents the "Device is not ready: D:\" crash.
    local_tracking_uri = f"file:///{Path('./mlruns').resolve().as_posix()}"
    mlflow.set_tracking_uri(local_tracking_uri)
    
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="pneumonia_densenet121"):
        mlflow.log_params(HPARAMS)
        mlflow.log_param("model_arch", "DenseNet-121")
        mlflow.log_param("classes", CLASS_NAMES)

        best_val_loss = np.inf

        for epoch in range(1, HPARAMS["num_epochs"] + 1):

            # Unfreeze backbone after warmup
            if epoch == HPARAMS["freeze_backbone_epochs"] + 1:
                unfreeze_backbone(model)
                
                # Rebuild parameter groups on existing optimizer to maintain scheduler sync
                optimizer.param_groups.clear()
                optimizer.add_param_group({
                    "params": model.parameters(), 
                    "lr": HPARAMS["learning_rate"] / 10,
                    "weight_decay": HPARAMS["weight_decay"]
                })
                scheduler.base_lrs = [HPARAMS["learning_rate"] / 10]

            train_loss, train_acc = train_one_epoch(
                model, loaders["train"], optimizer, criterion, scaler, device
            )
            val_loss, val_acc, val_metrics = evaluate(
                model, loaders["val"], criterion, device
            )
            scheduler.step()

            logger.info(
                f"Epoch {epoch:02d}/{HPARAMS['num_epochs']} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            # MLflow logging
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                **({"val_auc": val_metrics["auc_roc"]} if val_metrics["auc_roc"] else {}),
            }, step=epoch)

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, epoch, val_loss, best_model_path)
                mlflow.pytorch.log_model(model, "best_model")

            # Early stopping
            if early_stop(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Final test evaluation 
        logger.info("Running final evaluation on test set...")
        
        # Clear optimizer graphs and empty GPU cache before test evaluation 
        del optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        _, test_acc, test_metrics = evaluate(model, loaders["test"], criterion, device)

        logger.info(f"Test accuracy: {test_acc:.4f}")
        if test_metrics["auc_roc"]:
            logger.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")

        mlflow.log_metrics({
            "test_acc": test_acc,
            **({"test_auc": test_metrics["auc_roc"]} if test_metrics["auc_roc"] else {}),
        })

        logger.info("Training complete.")


if __name__ == "__main__":
    train()