###  Training loop for the malaria cell classifier.
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
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, f1_score
)

from models.malaria.model import build_model, CLASS_NAMES
from training.malaria.dataset import build_dataloaders
from core.config import settings
from core.logger import logger


HPARAMS = {
    "batch_size": 64,
    "num_epochs": 20,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "dropout": 0.4,
    "patience": 6,
    "num_workers": 4,
    "image_size": 128,
    "freeze_backbone_epochs": 2,
}


class EarlyStopping:
    def __init__(self, patience: int = 6, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = np.inf

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
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
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits  = model(images)
        loss    = criterion(logits, labels)
        probs   = torch.softmax(logits, dim=1)
        preds   = probs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # prob of PARASITIZED

    avg_loss = total_loss / total
    accuracy = correct / total

    # AUC-ROC — binary, use probability of positive class
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = None

    f1 = f1_score(all_labels, all_preds, average="binary")
    cm = confusion_matrix(all_labels, all_preds)

    report = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True
    )

    metrics = {
        "accuracy": accuracy,
        "loss": avg_loss,
        "auc_roc": auc,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    return avg_loss, accuracy, metrics


def save_checkpoint(model, epoch, val_loss, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "class_names": CLASS_NAMES,
    }, path)
    logger.info(f"Checkpoint saved → {path} (epoch {epoch}, val_loss={val_loss:.4f})")


def train():
    device = settings.device
    logger.info(f"Training malaria classifier on: {device}")

    data_dir = Path("data/raw/malaria")
    loaders  = build_dataloaders(
        data_dir,
        batch_size=HPARAMS["batch_size"],
        num_workers=HPARAMS["num_workers"],
    )

    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss()  # balanced dataset — no weighting needed
    optimizer = AdamW(
        model.parameters(),
        lr=HPARAMS["learning_rate"],
        weight_decay=HPARAMS["weight_decay"],
    )
    scheduler  = CosineAnnealingLR(optimizer, T_max=HPARAMS["num_epochs"], eta_min=1e-6)
    scaler     = GradScaler(enabled=(device == "cuda"))
    early_stop = EarlyStopping(patience=HPARAMS["patience"])

    best_model_path = settings.model_registry_dir / "malaria" / "best_model.pth"

    # Freeze backbone for warmup epochs
    for param in model.features.parameters():
        param.requires_grad = False
    logger.info("Backbone frozen for warmup")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="malaria_efficientnet_b0"):
        mlflow.log_params(HPARAMS)
        mlflow.log_param("model_arch", "EfficientNet-B0")
        mlflow.log_param("classes", CLASS_NAMES)

        best_val_loss = np.inf

        for epoch in range(1, HPARAMS["num_epochs"] + 1):

            # Unfreeze after warmup
            if epoch == HPARAMS["freeze_backbone_epochs"] + 1:
                for param in model.features.parameters():
                    param.requires_grad = True
                optimizer = AdamW(
                    model.parameters(),
                    lr=HPARAMS["learning_rate"] / 10,
                    weight_decay=HPARAMS["weight_decay"],
                )
                logger.info("Backbone unfrozen — full fine-tuning")

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
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"val_auc={val_metrics['auc_roc']:.4f} | "
                f"val_f1={val_metrics['f1_score']:.4f}"
            )

            log_dict = {
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "val_f1": val_metrics["f1_score"],
            }
            if val_metrics["auc_roc"]:
                log_dict["val_auc"] = val_metrics["auc_roc"]
            mlflow.log_metrics(log_dict, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, epoch, val_loss, best_model_path)
                mlflow.pytorch.log_model(model, "best_model")

            if early_stop(val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Final test evaluation
        logger.info("Evaluating on test set...")
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        _, test_acc, test_metrics = evaluate(model, loaders["test"], criterion, device)

        logger.info(f"Test accuracy : {test_acc:.4f}")
        logger.info(f"Test AUC-ROC  : {test_metrics['auc_roc']:.4f}")
        logger.info(f"Test F1 Score : {test_metrics['f1_score']:.4f}")
        logger.info(f"Confusion matrix:\n{test_metrics['confusion_matrix']}")

        mlflow.log_metrics({
            "test_acc": test_acc,
            "test_f1":  test_metrics["f1_score"],
            **({"test_auc": test_metrics["auc_roc"]} if test_metrics["auc_roc"] else {}),
        })

        logger.info("Malaria training complete.")


if __name__ == "__main__":
    train()
