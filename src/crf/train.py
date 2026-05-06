"""
Algorithm S: iterative-scaling training of a linear-chain CRF.

The intuition in one paragraph
------------------------------
For every feature we want to learn a weight for, we compute two numbers:

* `observed[k]` — how often the feature fired on the *gold* labels.
* `expected[k]` — how often the feature fires *in expectation* under the
  current model (weighting each possible labeling by its probability).

If observed > expected, the model under-uses the feature → push its weight up.
If observed < expected, it over-uses it → push down. When the two match, the
weight is right. That's Algorithm S.

Update rule::

    weight[k] += (1 / S) * log(observed[k] / expected[k])

`S` is a scaling constant chosen so the algorithm is guaranteed to move in
the right direction; any value that upper-bounds the total active-feature
count per sentence works.

What we accumulate separately
-----------------------------
We keep four parallel "weight vectors" and update each of them with their own
observed/expected counts:

* `emission_weights[k, v]` — observation feature k firing with label v.
* `transition_weights[u, v]` — jumping from u to v.
* `start_weights[v]` — starting a sentence with v.
* `stop_weights[u]` — ending a sentence with u.

Each of the four has its own observed- and expected-count arrays of matching
shape. The bookkeeping is mostly "sum probabilities into the right bucket."
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from crf.inference import (
    CRFParameters,
    compute_emission_scores,
    forward_backward,
    pairwise_marginals,
    score_labeling,
    start_marginals,
    stop_marginals,
    unary_marginals,
)
from crf.vocabulary import Vocabulary


@dataclass
class ObservedCounts:
    """How often each feature fires on the gold labels, summed over training data."""

    emission: np.ndarray   # (n_observations, R)
    transition: np.ndarray  # (R, R)
    start: np.ndarray       # (R,)
    stop: np.ndarray        # (R,)


@dataclass
class EncodedSentence:
    """A training sentence after vocabulary encoding — ready for training loops."""

    obs_ids: list[list[int]]  # active observation IDs per position
    label_ids: list[int]       # gold real-label IDs


def encode_sentences(
    vocab: Vocabulary,
    sentences: Sequence[dict],
) -> list[EncodedSentence]:
    """Convert list-of-dict sentences into pre-encoded form for faster training."""
    encoded = []
    for sentence in sentences:
        tokens = sentence["words"]
        labels = sentence["lid"]
        obs_ids = [vocab.encode_features(tokens, i) for i in range(len(tokens))]
        label_ids = vocab.encode_labels(labels)
        encoded.append(EncodedSentence(obs_ids=obs_ids, label_ids=label_ids))
    return encoded


# ------------------------------------------------------------ observed counts

def compute_observed_counts(
    vocab: Vocabulary,
    encoded: Sequence[EncodedSentence],
) -> ObservedCounts:
    """
    Count feature firings on gold labels. Only depends on the data — call once.
    """
    n_real = len(vocab.real_label_ids())

    emission = np.zeros((vocab.n_observations, n_real))
    transition = np.zeros((n_real, n_real))
    start = np.zeros(n_real)
    stop = np.zeros(n_real)

    for sentence in encoded:
        labels = sentence.label_ids
        n = len(labels)
        start[labels[0]] += 1
        stop[labels[-1]] += 1
        for i in range(n):
            for obs_id in sentence.obs_ids[i]:
                emission[obs_id, labels[i]] += 1
            if i > 0:
                transition[labels[i - 1], labels[i]] += 1

    return ObservedCounts(emission=emission, transition=transition, start=start, stop=stop)


# ------------------------------------------------------------ expected counts

def compute_expected_counts_and_loglik(
    params: CRFParameters,
    encoded: Sequence[EncodedSentence],
) -> tuple[ObservedCounts, float]:
    """
    Expected feature firings under the current model, plus total log-likelihood.

    We reuse the `ObservedCounts` dataclass as a container — it has the
    right shape — even though these are expected rather than observed counts.

    Log-likelihood is returned too because we need it to track convergence:
    it must increase monotonically, or there is a bug somewhere.
    """
    vocab = params.vocab
    n_real = len(vocab.real_label_ids())

    expected_emission = np.zeros((vocab.n_observations, n_real))
    expected_transition = np.zeros((n_real, n_real))
    expected_start = np.zeros(n_real)
    expected_stop = np.zeros(n_real)

    total_loglik = 0.0

    for sentence in encoded:
        emissions = compute_emission_scores(params, sentence.obs_ids)
        fb = forward_backward(params, emissions)

        gold_score = score_labeling(params, emissions, sentence.label_ids)
        total_loglik += gold_score - fb.log_Z

        uni = unary_marginals(fb)              # (n, R)
        pair = pairwise_marginals(params, fb)  # (n-1, R, R)

        # Emission expected counts: for each position i and label v,
        # add p(y_i=v) to every active observation feature's (k, v) entry.
        for i, obs_ids in enumerate(sentence.obs_ids):
            if obs_ids:
                # Broadcasting: expected_emission[obs_ids] has shape (|obs|, R);
                # we add the same length-R row uni[i] to each.
                expected_emission[np.asarray(obs_ids)] += uni[i]

        if pair.shape[0] > 0:
            expected_transition += pair.sum(axis=0)

        expected_start += start_marginals(params, fb)
        expected_stop += stop_marginals(params, fb)

    expected = ObservedCounts(
        emission=expected_emission,
        transition=expected_transition,
        start=expected_start,
        stop=expected_stop,
    )
    return expected, total_loglik


# ------------------------------------------------------------ update rule

def _algorithm_s_update(
    weights: np.ndarray,
    observed: np.ndarray,
    expected: np.ndarray,
    S: float,
    epsilon: float = 1e-12,
) -> None:
    """
    Apply the Algorithm S update in-place.

    Features that never fire on gold labels (`observed == 0`) can't learn
    anything useful — the log would blow up — so we skip them.
    """
    active = observed > 0
    safe_expected = np.maximum(expected, epsilon)
    delta = np.zeros_like(weights)
    delta[active] = (1.0 / S) * np.log(observed[active] / safe_expected[active])
    weights += delta


# ------------------------------------------------------------ top-level trainer

@dataclass
class TrainingHistory:
    loglik: list[float]


def choose_scaling_constant(encoded: Sequence[EncodedSentence]) -> float:
    """
    Pick `S` big enough to guarantee Algorithm S converges.

    The safe choice is an upper bound on the total number of active features
    any single sentence contributes. Sentence contributes:
      - 1 start + 1 stop
      - (n-1) transitions
      - sum over positions of (number of active observation features)
    Taking the max over training sentences gives a loose but valid bound.
    """
    max_count = 1
    for sentence in encoded:
        n = len(sentence.label_ids)
        obs_total = sum(len(active) for active in sentence.obs_ids)
        total = 1 + 1 + max(n - 1, 0) + obs_total
        if total > max_count:
            max_count = total
    return float(max_count)


def train(
    vocab: Vocabulary,
    encoded: Sequence[EncodedSentence],
    max_iters: int = 50,
    scaling_constant: float | None = None,
    tol: float = 1e-4,
    verbose: bool = False,
) -> tuple[CRFParameters, TrainingHistory]:
    """
    Train a CRF with Algorithm S.

    Stops when the relative improvement in log-likelihood drops below `tol`
    for two iterations in a row, or after `max_iters` iterations.
    """
    params = CRFParameters.zeros(vocab)
    observed = compute_observed_counts(vocab, encoded)

    S = scaling_constant if scaling_constant is not None else choose_scaling_constant(encoded)

    history = TrainingHistory(loglik=[])
    small_improvement_streak = 0
    prev_loglik = -np.inf

    for iteration in range(max_iters):
        expected, loglik = compute_expected_counts_and_loglik(params, encoded)
        history.loglik.append(loglik)

        if verbose:
            print(f"iter {iteration:3d}  log-likelihood={loglik:.4f}  S={S}")

        _algorithm_s_update(params.emission_weights, observed.emission, expected.emission, S)
        _algorithm_s_update(params.transition_weights, observed.transition, expected.transition, S)
        _algorithm_s_update(params.start_weights, observed.start, expected.start, S)
        _algorithm_s_update(params.stop_weights, observed.stop, expected.stop, S)

        if iteration > 0:
            denom = max(abs(prev_loglik), 1.0)
            rel = (loglik - prev_loglik) / denom
            if rel < tol:
                small_improvement_streak += 1
                if small_improvement_streak >= 2:
                    break
            else:
                small_improvement_streak = 0
        prev_loglik = loglik

    return params, history


# ------------------------------------------------------------ prediction

def predict(
    params: CRFParameters,
    sentences: Sequence[dict],
) -> list[list[str]]:
    """Return the Viterbi-best label sequence (as strings) for each sentence."""
    from crf.inference import viterbi  # local import to avoid circularity issues at module load
    id_to_label = params.vocab.id_to_label()
    predictions: list[list[str]] = []
    for sentence in sentences:
        tokens = sentence["words"]
        obs_ids = [params.vocab.encode_features(tokens, i) for i in range(len(tokens))]
        emissions = compute_emission_scores(params, obs_ids)
        label_ids, _ = viterbi(params, emissions)
        predictions.append([id_to_label[i] for i in label_ids])
    return predictions
