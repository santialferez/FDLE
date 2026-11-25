"""Notebook plots and readable source listings."""

import inspect
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def plot_history(history, title=None):
    """Plot losses and validation accuracy from fit, and return the figure."""
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(9, 3), layout="constrained")
    train_loss, val_loss, val_acc = (history[key] for key in ("train_loss", "val_loss", "val_acc"))
    loss_ax.plot(range(1, len(train_loss) + 1), train_loss, label="Training")
    if val_loss:
        loss_ax.plot(range(1, len(val_loss) + 1), val_loss, label="Validation")
    loss_ax.set(xlabel="Epoch", ylabel="Loss")
    loss_ax.legend()
    acc_ax.plot(range(1, len(val_acc) + 1), val_acc)
    acc_ax.set(xlabel="Epoch", ylabel="Validation accuracy", ylim=(0, 1.05))
    if title:
        fig.suptitle(title)
    return fig


def plot_cv_scores(fold_accs, single_split_acc=None, title=None):
    """Plot accuracy per fold and an optional single-split reference."""
    fig, ax = plt.subplots(layout="constrained")
    folds = range(1, len(fold_accs) + 1)
    ax.bar(folds, fold_accs)
    ax.axhline(np.mean(fold_accs), color="black", linestyle="--", label="CV mean")
    if single_split_acc is not None:
        ax.axhline(single_split_acc, color="tab:orange", linestyle=":", label="Single split")
    ax.set(xlabel="Fold", ylabel="Accuracy", xticks=list(folds), ylim=(0, 1.05))
    ax.set_title(title or "Cross-validation accuracy")
    ax.legend()
    return fig


def plot_confusion(y_true, y_pred, class_names=None, title=None):
    """Plot a confusion matrix and return the figure."""
    fig, ax = plt.subplots(layout="constrained")
    labels = np.union1d(y_true, y_pred)
    if class_names is not None and np.isin(labels, np.arange(len(class_names))).all():
        labels = np.arange(len(class_names))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, display_labels=class_names, ax=ax, colorbar=False
    )
    ax.set_title(title or "Confusion matrix")
    return fig


def show_source(obj):
    """Print a header and the source of a function or class."""
    obj = inspect.unwrap(obj)
    print(f"# {obj.__module__}.{obj.__qualname__}")
    print(dedent(inspect.getsource(obj)).rstrip())
