"""Cross-validation, bootstrap confidence intervals and classification metrics."""

import inspect
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from .train import evaluate, fit, set_seed


def _feature_array(rows):
    try:
        array = np.asarray(rows)
    except ValueError:  # Ragged rows need preprocessing before tensor conversion.
        return np.asarray(rows, dtype=object)
    if array.dtype.kind in "US":
        return np.asarray(rows, dtype=object)
    if array.dtype.kind in "biuf":
        return array.astype(float, copy=False)
    return array


def _fold_loader(X, y, batch_size, shuffle):
    features = torch.as_tensor(X)
    if features.is_floating_point():
        features = features.float()
    targets = torch.as_tensor(y, dtype=torch.long, device=features.device)
    return DataLoader(TensorDataset(features, targets), batch_size=batch_size, shuffle=shuffle)


def cross_validate(model_fn, X, y, *, k=5, stratified=True, preprocess_fn=None,
                   epochs=5, lr=1e-3, batch_size=64, grad_clip=None, amp=False,
                   batch_transform=None, seed=0, device=None, verbose=True,
                   loss_fn=None, step_fn=None) -> dict:
    """Train a fresh model per fold and return predictions in original row order.

    Preprocessing receives `(X_train, y_train, X_val)` or `(X_train, X_val)`.
    Fit preprocessing on the training rows and apply it to both feature sets.
    Use preprocessing to turn strings or variable-length rows into tensors or
    numeric arrays of fixed shape. Numeric feature lists and floating-point
    arrays become float32 tensors. Integer arrays and tensors can hold token IDs.
    A custom loss or step is used for both training and evaluation.
    """
    start = perf_counter()
    set_seed(seed)
    if isinstance(X, list):
        X = _feature_array(X)
    y = (y if torch.is_tensor(y) else np.asarray(y)).reshape(-1)
    labels = y.detach().cpu().numpy() if torch.is_tensor(y) else y
    splitter = StratifiedKFold if stratified else KFold
    splits = splitter(n_splits=k, shuffle=True, random_state=seed)
    fold_accs = []
    oof_pred = np.empty(len(labels), dtype=int)
    oof_prob = None
    if preprocess_fn is not None:
        try:
            inspect.signature(preprocess_fn).bind(X, y, X)
            use_train_labels = True
        except TypeError:
            use_train_labels = False

    for train_idx, val_idx in splits.split(X, labels):
        X_train, X_val = X[train_idx], X[val_idx]
        if preprocess_fn is not None:
            args = (X_train, y[train_idx], X_val) if use_train_labels else (X_train, X_val)
            X_train, X_val = preprocess_fn(*args)
        train_loader = _fold_loader(X_train, labels[train_idx], batch_size, shuffle=True)
        val_loader = _fold_loader(X_val, labels[val_idx], batch_size, shuffle=False)
        model = model_fn()
        fit(model, train_loader, epochs=epochs, lr=lr, grad_clip=grad_clip,
            amp=amp, batch_transform=batch_transform, device=device, verbose=False,
            loss_fn=loss_fn, step_fn=step_fn)
        scores = evaluate(model, val_loader, batch_transform=batch_transform,
                          device=device, loss_fn=loss_fn, step_fn=step_fn)
        fold_accs.append(scores["acc"])
        oof_pred[val_idx] = scores["y_pred"]
        if oof_prob is None:
            prob = scores["y_prob"]
            oof_prob = np.empty((len(labels), *prob.shape[1:]), dtype=prob.dtype)
        oof_prob[val_idx] = scores["y_prob"]
        if verbose:
            print(f"Fold {len(fold_accs)}/{k} | val acc {scores['acc']:.1%}")

    return dict(fold_accs=fold_accs, mean_acc=float(np.mean(fold_accs)),
                std_acc=float(np.std(fold_accs)), oof_true=labels.copy(),
                oof_pred=oof_pred, oof_prob=oof_prob, time_s=perf_counter() - start)


def _accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def bootstrap_ci(y_true, y_pred, *, metric_fn=None, n_boot=500, ci=0.95, seed=0) -> dict:
    """Return the original metric, bootstrap mean and paired percentile interval."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) != len(y_pred) or not len(y_true):
        raise ValueError("Bootstrap inputs must have the same positive length.")
    if n_boot < 1 or not 0 < ci < 1:
        raise ValueError("Use at least one bootstrap sample and a confidence level between 0 and 1.")

    metric_fn = _accuracy if metric_fn is None else metric_fn
    point = float(metric_fn(y_true, y_pred))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        samples[i] = metric_fn(y_true[idx], y_pred[idx])

    tail = (1 - ci) / 2
    low, high = np.quantile(samples, [tail, 1 - tail])
    return dict(point=point, mean=float(samples.mean()), low=float(low),
                high=float(high), samples=samples)


def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Report accuracy, precision, recall, F1 and ROC AUC when defined.

    Precision, recall and F1 use class 1 for binary 0/1 labels and macro
    averaging otherwise. Multiclass AUC uses macro one-vs-rest averaging.
    """
    y_true, y_pred = np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)
    labels = np.union1d(y_true, y_pred)
    average = "binary" if np.isin(labels, [0, 1]).all() else "macro"
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    auc = None
    if y_prob is not None and len(np.unique(y_true)) >= 2:
        prob = np.asarray(y_prob)
        if prob.ndim == 2 and prob.shape[1] <= 2:
            prob = prob[:, -1]
        try:
            auc = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
        except ValueError:
            pass  # AUC is undefined when the required classes are missing.
    return dict(acc=_accuracy(y_true, y_pred), precision=float(precision),
                recall=float(recall), f1=float(f1), auc=auc)
