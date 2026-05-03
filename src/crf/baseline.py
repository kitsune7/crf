"""
Per-token logistic regression baseline.

Why we need this
----------------
The whole point of a CRF over a per-token classifier is that it can learn
"lang2 words tend to follow lang2 words" — a pattern a logistic regression that
sees one token at a time cannot capture. To *measure* that advantage we need a
per-token classifier that uses the same features as the CRF; anything else
conflates feature quality with model structure.

Implementation
--------------
We reuse the same `feature_strings` extractor as the CRF. For each training
token we produce one sparse row (1.0 at every active feature ID, 0 elsewhere)
and hand the whole matrix to `sklearn.linear_model.LogisticRegression`.
sklearn is responsible for the actual optimisation; we just wire inputs/outputs.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

from crf.feature_extractor import feature_strings


@dataclass
class BaselineModel:
    """Trained LR model plus the feature/label index it uses."""

    classifier: LogisticRegression
    feature_to_id: dict[str, int]
    label_to_id: dict[str, int]
    id_to_label: list[str]

    @property
    def n_features(self) -> int:
        return len(self.feature_to_id)


def _build_feature_vocab(sentences: Sequence[dict], min_count: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sentence in sentences:
        tokens = sentence["words"]
        for i in range(len(tokens)):
            for feature in feature_strings(tokens, i):
                counts[feature] = counts.get(feature, 0) + 1
    return {
        feature: idx
        for idx, feature in enumerate(
            sorted(feature for feature, count in counts.items() if count >= min_count)
        )
    }


def _build_label_vocab(sentences: Sequence[dict]) -> tuple[dict[str, int], list[str]]:
    labels: set[str] = set()
    for sentence in sentences:
        labels.update(sentence["lid"])
    ordered = sorted(labels)
    return {label: idx for idx, label in enumerate(ordered)}, ordered


def _encode_tokens(
    sentences: Sequence[dict],
    feature_to_id: dict[str, int],
    label_to_id: dict[str, int] | None = None,
) -> tuple[csr_matrix, np.ndarray | None]:
    """
    Build the ``(n_tokens, n_features)`` sparse matrix and (optionally) the
    per-token label array. If ``label_to_id`` is None we're encoding at test
    time and only return the feature matrix.
    """
    rows: list[int] = []
    cols: list[int] = []
    labels: list[int] = []
    row = 0
    for sentence in sentences:
        tokens = sentence["words"]
        for i in range(len(tokens)):
            for feature in feature_strings(tokens, i):
                idx = feature_to_id.get(feature)
                if idx is not None:
                    rows.append(row)
                    cols.append(idx)
            if label_to_id is not None:
                labels.append(label_to_id[sentence["lid"][i]])
            row += 1
    data = np.ones(len(rows), dtype=np.float32)
    X = csr_matrix((data, (rows, cols)), shape=(row, len(feature_to_id)))
    y = np.asarray(labels) if label_to_id is not None else None
    return X, y


def train_baseline(
    sentences: Sequence[dict],
    min_count: int = 2,
    max_iter: int = 200,
    random_state: int = 0,
) -> BaselineModel:
    feature_to_id = _build_feature_vocab(sentences, min_count=min_count)
    label_to_id, id_to_label = _build_label_vocab(sentences)

    X, y = _encode_tokens(sentences, feature_to_id, label_to_id)

    classifier = LogisticRegression(
        max_iter=max_iter,
        solver="liblinear",  # handles sparse inputs well for this dataset size
        random_state=random_state,
    )
    classifier.fit(X, y)

    return BaselineModel(
        classifier=classifier,
        feature_to_id=feature_to_id,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
    )


def predict_baseline(
    model: BaselineModel,
    sentences: Sequence[dict],
) -> list[list[str]]:
    X, _ = _encode_tokens(sentences, model.feature_to_id, label_to_id=None)
    if X.shape[0] == 0:
        return [[] for _ in sentences]

    predicted = model.classifier.predict(X)

    # Unflatten back to per-sentence lists of label strings.
    predictions: list[list[str]] = []
    cursor = 0
    for sentence in sentences:
        n = len(sentence["words"])
        predictions.append([model.id_to_label[int(label)] for label in predicted[cursor : cursor + n]])
        cursor += n
    return predictions
