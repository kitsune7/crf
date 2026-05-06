"""
MEMM baseline (Maximum Entropy Markov Model).

Why an MEMM rather than a memoryless classifier
------------------------------------------------
The CRF has two structural advantages over a per-token logistic regression:

1. It models label-to-label transitions.
2. It normalises *globally* — the partition function sums over every possible
   labeling of the whole sentence, not just over the labels at a single
   position.

A memoryless baseline conflates those two effects. A locally-normalised
sequence model — an MEMM — controls for (1), so the remaining CRF advantage
isolates (2): the label-bias effect that global normalisation fixes.

Model
-----
For each position `i`,

    P(y_i | y_{i-1}, x_i) = softmax( w · f(x, i, y_{i-1}, y_i) )

The features are the CRF's observation features at `i` (via
`feature_strings`) plus a single `prev_label=X` indicator. Training is
one multinomial logistic regression over all training tokens using *gold*
previous labels — a standard per-token classification problem.

Decoding is Viterbi. At each position and for each candidate previous label
`u`, we score `log P(y_i = v | y_{i-1} = u, x_i)` and pick the highest-
scoring path. This is the "locally normalised" part: each position's `Z`
depends only on `(x_i, y_{i-1})`, so probability mass cannot flow between
positions. That's precisely the label-bias limitation the CRF's global
`log Z` removes.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

from crf.feature_extractor import feature_strings

START = "<START>"


@dataclass
class BaselineModel:
    """Trained MEMM plus the feature/label index it uses."""

    classifier: LogisticRegression
    feature_to_id: dict[str, int]
    label_to_id: dict[str, int]
    id_to_label: list[str]

    @property
    def n_features(self) -> int:
        return len(self.feature_to_id)


def _prev_label_feature(label: str) -> str:
    return f"prev_label={label}"


def _build_label_vocab(sentences: Sequence[dict]) -> tuple[dict[str, int], list[str]]:
    labels: set[str] = set()
    for sentence in sentences:
        labels.update(sentence["lid"])
    ordered = sorted(labels)
    return {label: idx for idx, label in enumerate(ordered)}, ordered


def _build_feature_vocab(
    sentences: Sequence[dict],
    min_count: int,
    id_to_label: Sequence[str],
) -> dict[str, int]:
    """
    Build the combined observation + prev-label feature index.

    Observation features (word=tacos, suffix3=cos, ...) are pruned by
    `min_count` as usual. Prev-label features are always kept — each one
    fires on roughly (n_tokens / n_labels) training tokens, so pruning them
    would just break the model.
    """
    counts: dict[str, int] = {}
    for sentence in sentences:
        tokens = sentence["words"]
        for i in range(len(tokens)):
            for feature in feature_strings(tokens, i):
                counts[feature] = counts.get(feature, 0) + 1

    feature_to_id: dict[str, int] = {
        feature: idx
        for idx, feature in enumerate(
            sorted(feature for feature, count in counts.items() if count >= min_count)
        )
    }
    for label in [*id_to_label, START]:
        feature_to_id[_prev_label_feature(label)] = len(feature_to_id)
    return feature_to_id


def _encode_training_tokens(
    sentences: Sequence[dict],
    feature_to_id: dict[str, int],
    label_to_id: dict[str, int],
) -> tuple[csr_matrix, np.ndarray]:
    """
    Build the `(n_tokens, n_features)` sparse matrix and the per-token label
    array used to fit the underlying logistic regression. The prev-label
    feature uses the *gold* previous label.
    """
    rows: list[int] = []
    cols: list[int] = []
    labels: list[int] = []
    row = 0
    for sentence in sentences:
        tokens = sentence["words"]
        gold = sentence["lid"]
        for i in range(len(tokens)):
            for feature in feature_strings(tokens, i):
                idx = feature_to_id.get(feature)
                if idx is not None:
                    rows.append(row)
                    cols.append(idx)
            prev = gold[i - 1] if i > 0 else START
            rows.append(row)
            cols.append(feature_to_id[_prev_label_feature(prev)])
            labels.append(label_to_id[gold[i]])
            row += 1
    data = np.ones(len(rows), dtype=np.float32)
    X = csr_matrix((data, (rows, cols)), shape=(row, len(feature_to_id)))
    y = np.asarray(labels)
    return X, y


def train_baseline(
    sentences: Sequence[dict],
    min_count: int = 2,
    max_iter: int = 200,
    random_state: int = 0,
) -> BaselineModel:
    label_to_id, id_to_label = _build_label_vocab(sentences)
    feature_to_id = _build_feature_vocab(
        sentences, min_count=min_count, id_to_label=id_to_label
    )

    X, y = _encode_training_tokens(sentences, feature_to_id, label_to_id)

    classifier = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",  # sparse-capable and supports multinomial multiclass
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
    """Viterbi decoding with locally-normalised transition probabilities."""
    # The classifier columns are in classifier.classes_ order, which may not
    # equal our label-id order if some label never appeared in training. Map
    # once so Viterbi indexes everything in label-id space.
    class_ids = model.classifier.classes_
    col_to_label_id = {col: int(class_id) for col, class_id in enumerate(class_ids)}

    predictions: list[list[str]] = []
    for sentence in sentences:
        tokens = sentence["words"]
        if not tokens:
            predictions.append([])
            continue
        label_ids = _viterbi(model, tokens, col_to_label_id)
        predictions.append([model.id_to_label[i] for i in label_ids])
    return predictions


def _viterbi(
    model: BaselineModel,
    tokens: Sequence[str],
    col_to_label_id: dict[int, int],
) -> list[int]:
    """
    Standard Viterbi over log P(y_i | y_{i-1}, x_i).

    At position 0 the only candidate `y_{i-1}` is the START sentinel, so we
    score a single row. At positions 1..n-1 we enumerate every real label as
    a candidate previous label and score one row per (position, prev-label)
    pair.
    """
    n = len(tokens)
    L = len(model.id_to_label)

    # Build one sparse row per (position, prev-label-candidate).
    prev_labels_per_step: list[list[str]] = [[START]] + [list(model.id_to_label)] * (n - 1)

    rows: list[int] = []
    cols: list[int] = []
    row = 0
    for i in range(n):
        obs_cols: list[int] = []
        for feature in feature_strings(tokens, i):
            idx = model.feature_to_id.get(feature)
            if idx is not None:
                obs_cols.append(idx)
        for prev_label in prev_labels_per_step[i]:
            for c in obs_cols:
                rows.append(row)
                cols.append(c)
            rows.append(row)
            cols.append(model.feature_to_id[_prev_label_feature(prev_label)])
            row += 1

    data = np.ones(len(rows), dtype=np.float32)
    X = csr_matrix((data, (rows, cols)), shape=(row, len(model.feature_to_id)))
    log_proba = model.classifier.predict_log_proba(X)  # (row, n_classes)

    # Translate column-space (sklearn classes_ order) to label-id space.
    logp_in_label_space = np.full((log_proba.shape[0], L), -np.inf)
    for col, label_id in col_to_label_id.items():
        logp_in_label_space[:, label_id] = log_proba[:, col]

    # Slice per position so Viterbi can index V[i-1] directly by prev-label id.
    logp_per_position: list[np.ndarray] = []
    cursor = 0
    for i in range(n):
        k = len(prev_labels_per_step[i])
        logp_per_position.append(logp_in_label_space[cursor : cursor + k])
        cursor += k

    V = np.full((n, L), -np.inf)
    back = np.zeros((n, L), dtype=np.int64)
    V[0] = logp_per_position[0][0]  # only one candidate prev (START)
    for i in range(1, n):
        # scores[u, v] = V[i-1, u] + log P(y_i = v | y_{i-1} = u, x_i)
        scores = V[i - 1][:, None] + logp_per_position[i]
        back[i] = scores.argmax(axis=0)
        V[i] = scores.max(axis=0)

    last = int(V[n - 1].argmax())
    labels = [0] * n
    labels[n - 1] = last
    for i in range(n - 1, 0, -1):
        labels[i - 1] = int(back[i, labels[i]])
    return labels
