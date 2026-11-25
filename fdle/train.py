"""Training and evaluation loops for PyTorch classifiers."""

import random
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Use a CUDA GPU when available, otherwise use the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters, optionally including frozen ones."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def _as_loader(data, shuffle=False):
    if isinstance(data, tuple):
        data = TensorDataset(*data)
    if isinstance(data, TensorDataset):
        return DataLoader(data, batch_size=64, shuffle=shuffle)
    return data


def _batch_step(model, batch, loss_fn, device, batch_transform, step_fn):
    if step_fn is not None:
        # The custom step handles device placement and its own preprocessing.
        return step_fn(model, batch, loss_fn)

    inputs, targets = (tensor.to(device) for tensor in batch)
    if batch_transform is not None:
        inputs = batch_transform(inputs)
    logits = model(inputs)
    if isinstance(loss_fn, nn.BCEWithLogitsLoss):
        targets = targets.to(logits.dtype).reshape_as(logits)
    else:
        targets = targets.long().reshape(-1)
    return loss_fn(logits, targets), logits, targets


@torch.no_grad()
def evaluate(model, loader, *, loss_fn=None, device=None, batch_transform=None,
             step_fn=None) -> dict:
    """Return mean loss, accuracy, labels, predictions and probabilities.

    The transform takes the input tensor on the model's device. A custom
    `step_fn(model, batch, loss_fn)` replaces the default step and returns
    `(loss, logits, targets)`. Loss is a scalar batch mean. With BCE, or
    single-logit output from a custom step, probabilities are for class 1.
    Otherwise, they form a matrix with one column per class.
    """
    device = torch.device(device) if device is not None else get_device()
    loss_fn = nn.CrossEntropyLoss() if loss_fn is None else loss_fn
    if isinstance(loss_fn, nn.Module):
        loss_fn.to(device)
    model.to(device).eval()
    loss_sum, labels, probabilities = 0.0, [], []
    for batch in _as_loader(loader):
        loss, logits, targets = _batch_step(
            model, batch, loss_fn, device, batch_transform, step_fn
        )
        targets = targets.reshape(-1)
        loss_sum += loss.item() * len(targets)
        labels.append(targets.cpu())
        probabilities.append(logits.float().cpu())

    if not labels:
        raise ValueError("Evaluation requires at least one example.")
    y_true = torch.cat(labels).long().numpy()
    logits = torch.cat(probabilities)
    if logits.ndim == 1 or logits.shape[1] == 1:
        y_prob = logits.reshape(-1).sigmoid().numpy()
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_prob = logits.softmax(dim=1).numpy()
        y_pred = y_prob.argmax(axis=1)
    return dict(loss=loss_sum / len(y_true), acc=float((y_true == y_pred).mean()),
                y_true=y_true, y_pred=y_pred, y_prob=y_prob)


def fit(model, train_loader, val_loader=None, *, epochs=5, lr=1e-3,
        optimizer=None, loss_fn=None, grad_clip=None, amp=False,
        batch_transform=None, step_fn=None, device=None, name=None,
        table=None, verbose=True) -> dict:
    """Train a classifier and return per-epoch losses and validation accuracy.

    Inputs can be loaders, TensorDatasets or pairs of tensors. Adam and cross
    entropy are the defaults. See evaluate for the custom step and transform.
    AMP uses mixed precision on CUDA. The model keeps its final epoch weights,
    and the results table records the best validation accuracy. Without
    validation, the validation lists are empty and best accuracy is NaN.
    """
    start = perf_counter()
    device = torch.device(device) if device is not None else get_device()
    model.to(device)
    train_loader = _as_loader(train_loader, shuffle=True)
    val_loader = _as_loader(val_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) if optimizer is None else optimizer
    loss_fn = nn.CrossEntropyLoss() if loss_fn is None else loss_fn
    if isinstance(loss_fn, nn.Module):
        loss_fn.to(device)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    train_loss, val_loss, val_acc = [], [], []

    for epoch in range(epochs):
        model.train()
        loss_sum, seen = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss, _, targets = _batch_step(
                    model, batch, loss_fn, device, batch_transform, step_fn
                )
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            batch_size = targets.numel()
            seen += batch_size
            loss_sum += loss.item() * batch_size

        if not seen:
            raise ValueError("Training requires at least one example.")
        train_loss.append(loss_sum / seen)
        message = f"Epoch {epoch + 1}/{epochs} | train loss {train_loss[-1]:.4f}"
        if val_loader is not None:
            scores = evaluate(model, val_loader, loss_fn=loss_fn, device=device,
                              batch_transform=batch_transform, step_fn=step_fn)
            val_loss.append(scores["loss"])
            val_acc.append(scores["acc"])
            message += f" | val loss {val_loss[-1]:.4f} | val acc {val_acc[-1]:.1%}"
        if verbose:
            print(message)

    params, time_s = count_params(model), perf_counter() - start
    best_val_acc = max(val_acc, default=float("nan"))
    if table is not None and name is not None:
        table.add(name, params, best_val_acc, time_s)
    return dict(train_loss=train_loss, val_loss=val_loss, val_acc=val_acc,
                time_s=time_s, params=params, best_val_acc=best_val_acc)


class ResultsTable:
    """Collect and print experiment results in the order they were added."""

    def __init__(self):
        self.rows = []

    def add(self, name, params, val_acc, time_s, **extra):
        """Append one result with optional configuration fields."""
        self.rows.append(dict(name=name, params=params, val_acc=val_acc, time_s=time_s, **extra))

    def show(self):
        """Print all rows, including any extra configuration fields."""
        if not self.rows:
            print("No results yet.")
            return
        columns = dict.fromkeys(key for row in self.rows for key in row)
        print(" | ".join(columns))
        for row in self.rows:
            print(" | ".join(str(row.get(key, "")) for key in columns))
