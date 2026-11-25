"""Training and evaluation loops for PyTorch classifiers."""

import random
from html import escape
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
             step_fn=None, keep_outputs=True) -> dict:
    """Return mean loss, accuracy, labels, predictions and probabilities.

    The transform takes the input tensor on the model's device. A custom
    `step_fn(model, batch, loss_fn)` replaces the default step and returns
    `(loss, logits, targets)`. Loss is a scalar batch mean. With BCE, or
    single-logit output from a custom step, probabilities are for class 1.
    Otherwise, they form a matrix with one column per class. Set
    `keep_outputs=False` to return labels and predictions with `y_prob=None`.
    """
    device = torch.device(device) if device is not None else get_device()
    loss_fn = nn.CrossEntropyLoss() if loss_fn is None else loss_fn
    if isinstance(loss_fn, nn.Module):
        loss_fn.to(device)
    model.to(device).eval()
    loss_sum, labels, predictions, probabilities = 0.0, [], [], []
    for batch in _as_loader(loader):
        loss, logits, targets = _batch_step(
            model, batch, loss_fn, device, batch_transform, step_fn
        )
        targets = targets.reshape(-1)
        loss_sum += loss.item() * len(targets)
        labels.append(targets.long().cpu())
        if logits.ndim == 1 or logits.shape[1] == 1:
            logits = logits.reshape(-1)
            pred = (logits >= 0).long()
            if keep_outputs:
                probabilities.append(logits.float().sigmoid().cpu())
        else:
            pred = logits.argmax(dim=1)
            if keep_outputs:
                probabilities.append(logits.float().softmax(dim=1).cpu())
        predictions.append(pred.cpu())

    if not labels:
        raise ValueError("Evaluation requires at least one example.")
    y_true, y_pred = torch.cat(labels).numpy(), torch.cat(predictions).numpy()
    y_prob = torch.cat(probabilities).numpy() if keep_outputs else None
    return dict(loss=loss_sum / len(y_true), acc=float((y_true == y_pred).mean()),
                y_true=y_true, y_pred=y_pred, y_prob=y_prob)


def fit(model, train_loader, val_loader=None, *, epochs=5, lr=1e-3,
        optimizer=None, scheduler=None, loss_fn=None, grad_clip=None, amp=False,
        batch_transform=None, step_fn=None, device=None, name=None,
        table=None, verbose=True) -> dict:
    """Train a classifier and return per-epoch losses and validation accuracy.

    Inputs can be loaders, TensorDatasets or pairs of tensors. Adam and cross
    entropy are the defaults. See evaluate for the custom step and transform.
    AMP uses mixed precision on CUDA. A scheduler with a no-argument
    `step()` advances once per epoch. The table records final epoch validation
    accuracy. Without validation, the validation lists are empty and accuracy is NaN.
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
                              batch_transform=batch_transform, step_fn=step_fn,
                              keep_outputs=False)
            val_loss.append(scores["loss"])
            val_acc.append(scores["acc"])
            message += f" | val loss {val_loss[-1]:.4f} | val acc {val_acc[-1]:.1%}"
        if verbose:
            print(message)
        if scheduler is not None:
            scheduler.step()

    params, time_s = count_params(model), perf_counter() - start
    best_val_acc = max(val_acc, default=float("nan"))
    if table is not None and name is not None:
        table.add(name, params, val_acc[-1] if val_acc else float("nan"), time_s)
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
        """Print a compact text table."""
        if not self.rows:
            print("No results yet.")
            return
        columns, rows = self._display_data()
        print(" | ".join(columns))
        for row in rows:
            print(" | ".join(row))

    def _repr_html_(self):
        """Render a compact HTML table in notebooks."""
        if not self.rows:
            return "<p>No results yet.</p>"
        columns, rows = self._display_data()
        header = "".join(f'<th scope="col">{escape(key)}</th>' for key in columns)
        body = []
        for row in rows:
            cells = "".join(f"<td>{escape(value)}</td>" for value in row)
            body.append(f"<tr>{cells}</tr>")
        return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    def _display_data(self):
        columns = dict.fromkeys(
            key for row in self.rows for key, value in row.items()
            if key != "config" or not isinstance(value, dict)
        )
        rows = [[self._format_cell(key, row.get(key, "")) for key in columns]
                for row in self.rows]
        return columns, rows

    @staticmethod
    def _format_cell(key, value):
        if key == "val_acc":
            return f"{value:.1%}"
        if key == "time_s":
            return f"{value:.1f} s"
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        if isinstance(value, dict):
            return f"[dict, size={len(value)}]"
        return str(value)
