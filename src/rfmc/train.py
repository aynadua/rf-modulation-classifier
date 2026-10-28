"""CNN training loop.

Model selection is on validation accuracy, and the validation split is a
separate set of channel realisations rather than a slice of the training
array. The test split is touched exactly once, at the end.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rfmc.config import Config
from rfmc.dataset import IQDataset
from rfmc.models import IQCNN, count_parameters
from rfmc.modulators import SCHEMES


@dataclass
class TrainResult:
    model: IQCNN
    history: list[dict[str, float]]
    best_val_accuracy: float
    epochs_run: int
    train_seconds: float


def _loader(data: IQDataset, batch_size: int, shuffle: bool, workers: int = 0) -> DataLoader:
    tensors = TensorDataset(torch.from_numpy(data.x), torch.from_numpy(data.y))
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle, num_workers=workers)


@torch.no_grad()
def evaluate_logits(model: nn.Module, data: IQDataset, batch_size: int = 512) -> np.ndarray:
    """Logits for every example, in dataset order."""
    model.eval()
    chunks = []
    for start in range(0, len(data), batch_size):
        batch = torch.from_numpy(data.x[start : start + batch_size])
        chunks.append(model(batch).numpy())
    return np.concatenate(chunks)


def predict(model: nn.Module, data: IQDataset, batch_size: int = 512) -> np.ndarray:
    """Predicted class indices."""
    return evaluate_logits(model, data, batch_size).argmax(axis=1)


def train(
    train_data: IQDataset, val_data: IQDataset, cfg: Config, verbose: bool = True
) -> TrainResult:
    """Fit the CNN, keeping the weights with the best validation accuracy."""
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    model = IQCNN(n_classes=len(SCHEMES))
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.train.epochs)
    criterion = nn.CrossEntropyLoss()
    loader = _loader(train_data, cfg.train.batch_size, shuffle=True, workers=cfg.train.num_workers)

    history: list[dict[str, float]] = []
    best_accuracy = -1.0
    best_state: dict[str, Any] | None = None
    epochs_without_gain = 0
    start_time = time.perf_counter()

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for xb, yb in loader:
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * len(yb)
            seen += len(yb)
        scheduler.step()

        val_accuracy = float(np.mean(predict(model, val_data) == val_data.y))
        entry = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "val_accuracy": val_accuracy,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(entry)
        if verbose:
            print(
                f"epoch {epoch:3d}  loss {entry['train_loss']:.4f}  "
                f"val_acc {val_accuracy:.4f}",
                flush=True,
            )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= cfg.train.patience:
                if verbose:
                    print(f"early stop at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(
        model=model,
        history=history,
        best_val_accuracy=best_accuracy,
        epochs_run=len(history),
        train_seconds=time.perf_counter() - start_time,
    )


def save_checkpoint(result: TrainResult, cfg: Config, path: str | Path) -> Path:
    """Persist weights alongside the config that produced them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "config": cfg.to_dict(),
            "classes": list(SCHEMES),
            "n_parameters": count_parameters(result.model),
            "best_val_accuracy": result.best_val_accuracy,
            "history": result.history,
        },
        path,
    )
    path.with_suffix(".history.json").write_text(json.dumps(result.history, indent=2))
    return path


def load_checkpoint(path: str | Path) -> tuple[IQCNN, dict[str, Any]]:
    """Restore a trained model and its metadata."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = IQCNN(n_classes=len(blob["classes"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob
