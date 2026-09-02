"""
training/skin_cancer/train.py
Training loop for the 7-class skin lesion classifier (HAM10000).

Key features:
  - Supports seamless checkpoint saving and resuming across training sessions
  - 7-class multi-class classification — macro-averaged metrics matter most
  - Severe class imbalance (58x between largest and smallest class) —
    weighted CrossEntropyLoss + WeightedRandomSampler used together
  - Per-class recall reported separately — critical for clinical evaluation

Run from ml_service root (Windows CMD):
  python -m ml_service.training.skin_cancer.train
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from models.skin_cancer.model import build_model, CLASS_NAMES
from training.skin_cancer.dataset import build_dataloaders
from core.config import settings
from core.logger import logger


HPARAMS = {
    "batch_size": 16,
    "num_epochs": 35,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "dropout": 0.4,
    "patience": 8,
    "num_workers": 4,
    "freeze_backbone_epochs": 4,
}


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.counter, self.best_score = 0, -np.inf

    def __call__(self, val_macro_f1: float) -> bool:
        """Stops on macro F1 plateau, not loss — better signal for imbalanced classes."""
        if val_macro_f1 > self.best_score + self.min_delta:
            self.best_score, self.counter = val_macro_f1, 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
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
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss, accuracy = total_loss / total, correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    # Melanoma recall specifically — the single most clinically important metric
    melanoma_recall = report["MELANOMA"]["recall"] if "MELANOMA" in report else None

    return avg_loss, accuracy, {
        "macro_f1": macro_f1,
        "melanoma_recall": melanoma_recall,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, macro_f1, path: Path):
    """Saves the complete state dictionary to allow seamless resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "macro_f1": macro_f1,
        "class_names": CLASS_NAMES,
    }
    torch.save(state, path)
    logger.info(f"Checkpoint saved → {path} (epoch {epoch}, macro_f1={macro_f1:.4f})")


def train():
    device = settings.device
    logger.info(f"Training skin cancer classifier on: {device}")

    data_dir = Path("data/raw/skin_cancer")
    loaders = build_dataloaders(data_dir, batch_size=HPARAMS["batch_size"], num_workers=HPARAMS["num_workers"])

    model = build_model().to(device)
    for param in model.features.parameters():
        param.requires_grad = False
    logger.info("Backbone frozen for warmup")

    class_weights = loaders["train"].dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=HPARAMS["learning_rate"], weight_decay=HPARAMS["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=HPARAMS["num_epochs"], eta_min=1e-6)
    scaler = GradScaler(enabled=(device == "cuda"))
    early_stop = EarlyStopping(patience=HPARAMS["patience"])

    best_model_path = settings.model_registry_dir / "skin_cancer" / "best_model.pth"

    start_epoch = 0
    best_macro_f1 = -np.inf

    # ── RESUME CHECKPOINT LOGIC
    if best_model_path.exists():
        logger.info(f"Found existing checkpoint at {best_model_path}. Loading state...")
        checkpoint = torch.load(best_model_path, map_location=device)

        saved_epoch = checkpoint.get("epoch", 0)
        best_macro_f1 = checkpoint.get("macro_f1", -np.inf)

        # Unfreeze backbone if saved state was already past freeze_backbone_epochs
        if saved_epoch >= HPARAMS["freeze_backbone_epochs"]:
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = AdamW(model.parameters(), lr=HPARAMS["learning_rate"] / 10,
                              weight_decay=HPARAMS["weight_decay"])
            logger.info("Backbone unfrozen to match saved state")

        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and scaler is not None and checkpoint["scaler_state_dict"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = saved_epoch
        early_stop.best_score = best_macro_f1
        logger.info(f"✓ Resumed successfully from Epoch {start_epoch} (Best macro_f1: {best_macro_f1:.4f})")
    # --

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="skin_cancer_efficientnet_b3"):
        mlflow.log_params(HPARAMS)
        mlflow.log_param("model_arch", "EfficientNet-B3")
        mlflow.log_param("classes", CLASS_NAMES)

        for epoch in range(start_epoch + 1, HPARAMS["num_epochs"] + 1):
            if epoch == HPARAMS["freeze_backbone_epochs"] + 1:
                for param in model.features.parameters():
                    param.requires_grad = True
                optimizer = AdamW(model.parameters(), lr=HPARAMS["learning_rate"] / 10,
                                   weight_decay=HPARAMS["weight_decay"])
                logger.info("Backbone unfrozen — full fine-tuning")

            train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, scaler, device)
            val_loss, val_acc, val_metrics = evaluate(model, loaders["val"], criterion, device)
            scheduler.step()

            logger.info(
                f"Epoch {epoch:02d}/{HPARAMS['num_epochs']} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_acc={val_acc:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} | "
                f"melanoma_recall={val_metrics['melanoma_recall']}"
            )

            log_dict = {
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "val_macro_f1": val_metrics["macro_f1"],
            }
            if val_metrics["melanoma_recall"] is not None:
                log_dict["val_melanoma_recall"] = val_metrics["melanoma_recall"]
            mlflow.log_metrics(log_dict, step=epoch)

            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_macro_f1, best_model_path)
                mlflow.pytorch.log_model(model, "best_model")

            if early_stop(val_metrics["macro_f1"]):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        logger.info("Running final test evaluation...")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        _, test_acc, test_metrics = evaluate(model, loaders["test"], criterion, device)

        logger.info(f"Test accuracy : {test_acc:.4f}")
        logger.info(f"Test macro F1 : {test_metrics['macro_f1']:.4f}")
        logger.info(f"Melanoma recall: {test_metrics['melanoma_recall']}")
        logger.info(f"Confusion matrix:\n{test_metrics['confusion_matrix']}")

        mlflow.log_metrics({
            "test_acc": test_acc,
            "test_macro_f1": test_metrics["macro_f1"],
            **({"test_melanoma_recall": test_metrics["melanoma_recall"]}
               if test_metrics["melanoma_recall"] is not None else {}),
        })
        logger.info("Skin cancer training complete.")


if __name__ == "__main__":
    train()