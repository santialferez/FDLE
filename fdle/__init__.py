"""Course helpers for Foundations of Deep Learning for Engineering."""

from .eval import bootstrap_ci, classification_metrics, cross_validate
from .text import SimpleVocab, make_text_loaders, tokenize
from .train import ResultsTable, count_params, evaluate, fit, get_device, set_seed
from .viz import plot_confusion, plot_cv_scores, plot_history, show_source

__all__ = [
    "set_seed", "get_device", "count_params", "evaluate", "fit", "ResultsTable",
    "cross_validate", "bootstrap_ci", "classification_metrics",
    "plot_history", "plot_cv_scores", "plot_confusion", "show_source",
    "tokenize", "SimpleVocab", "make_text_loaders",
]
