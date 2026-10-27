"""Metrics at realistic prevalence, kernel diagnostics and statistical tests."""

from qfd.evaluation.diagnostics import (
    geometric_difference,
    kernel_offdiag_stats,
    target_alignment,
)
from qfd.evaluation.metrics import evaluate, precision_at_k, recall_at_fpr
from qfd.evaluation.stats import bootstrap_ci, paired_wilcoxon

__all__ = [
    "bootstrap_ci",
    "evaluate",
    "geometric_difference",
    "kernel_offdiag_stats",
    "paired_wilcoxon",
    "precision_at_k",
    "recall_at_fpr",
    "target_alignment",
]
