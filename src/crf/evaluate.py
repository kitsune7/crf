"""
Evaluation helpers: token accuracy, per-class P/R/F1, macro/weighted F1,
confusion matrix, and a switch-point-token accuracy breakdown.

Why a bespoke module instead of sklearn?
----------------------------------------
We still use sklearn for the per-class PRF numbers, but we want two things
that don't come out of the box:

* A **flat** accuracy that aligns all predicted/gold tokens across sentences.
* A **switch-point accuracy** — accuracy only on tokens where the gold label
  differs from the previous token's gold label. Those are the hard cases
  where a CRF should beat the per-token baseline. Having this metric surfaced
  makes the "does the CRF actually help?" question easy to answer.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)


@dataclass
class Evaluation:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    switch_point_accuracy: float | None
    confusion: np.ndarray
    labels: list[str]
    classification_report: str


def _flatten(
    sentences: Sequence[dict],
    predictions: Sequence[Sequence[str]],
) -> tuple[list[str], list[str]]:
    gold: list[str] = []
    pred: list[str] = []
    for sentence, prediction in zip(sentences, predictions):
        gold.extend(sentence["lid"])
        pred.extend(prediction)
    assert len(gold) == len(pred), "gold/pred length mismatch"
    return gold, pred


def _switch_point_mask(sentences: Sequence[dict]) -> list[bool]:
    """True at tokens whose gold label differs from the previous gold label."""
    mask: list[bool] = []
    for sentence in sentences:
        labels = sentence["lid"]
        for i, label in enumerate(labels):
            mask.append(i > 0 and label != labels[i - 1])
    return mask


def evaluate(
    sentences: Sequence[dict],
    predictions: Sequence[Sequence[str]],
) -> Evaluation:
    gold, pred = _flatten(sentences, predictions)
    labels = sorted(set(gold) | set(pred))

    accuracy = float(np.mean([g == p for g, p in zip(gold, pred)]))
    macro = float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0))
    weighted = float(f1_score(gold, pred, labels=labels, average="weighted", zero_division=0))
    cm = confusion_matrix(gold, pred, labels=labels)
    report = classification_report(gold, pred, labels=labels, zero_division=0)

    mask = _switch_point_mask(sentences)
    n_switch = sum(mask)
    if n_switch == 0:
        switch_acc = None
    else:
        correct = sum(1 for flag, g, p in zip(mask, gold, pred) if flag and g == p)
        switch_acc = correct / n_switch

    return Evaluation(
        accuracy=accuracy,
        macro_f1=macro,
        weighted_f1=weighted,
        switch_point_accuracy=switch_acc,
        confusion=cm,
        labels=labels,
        classification_report=report,
    )


def format_evaluation(name: str, evaluation: Evaluation) -> str:
    lines = [f"=== {name} ==="]
    lines.append(f"token accuracy : {evaluation.accuracy:.4f}")
    lines.append(f"macro F1       : {evaluation.macro_f1:.4f}")
    lines.append(f"weighted F1    : {evaluation.weighted_f1:.4f}")
    if evaluation.switch_point_accuracy is not None:
        lines.append(f"switch-point acc: {evaluation.switch_point_accuracy:.4f}")
    lines.append("")
    lines.append("per-class:")
    lines.append(evaluation.classification_report)
    lines.append("confusion matrix (rows=gold, cols=pred):")
    header = "        " + "  ".join(f"{label:>8}" for label in evaluation.labels)
    lines.append(header)
    for label, row in zip(evaluation.labels, evaluation.confusion):
        row_str = "  ".join(f"{count:>8}" for count in row)
        lines.append(f"{label:>8}  {row_str}")
    return "\n".join(lines)
