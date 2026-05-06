"""
CRF inference: scoring, Viterbi, and forward-backward.

Model parameters
----------------
A CRF has three kinds of weights:

* `emission_weights` — shape `(n_observations, n_real_labels)`. For every
  active observation feature `k` at position `i`, we add `emission_weights[k, v]`
  to the score of assigning label `v` at position `i`.
* `transition_weights` — shape `(R, R)` where R = number of real labels.
  `transition_weights[u, v]` scores the jump from label `u` at position
  `i-1` to label `v` at position `i`. This is what lets the CRF learn
  "lang2 tends to follow lang2" — something a per-token classifier can't see.
* `start_weights` / `stop_weights` — shape `(R,)` each. Boundary
  "transition" scores: `start_weights[v]` is the cost of starting a sentence
  with label `v`; `stop_weights[u]` is the cost of ending with label `u`.
  Keeping them separate (instead of folding START/STOP into the transition
  matrix) lets every `log_M` matrix have a uniform (R, R) shape.

Positions and indexing
----------------------
Positions are 0-indexed over the sentence. For a length-`n` sentence:

* position 0 … n-1 emit real labels
* the START sentinel lives conceptually "before" position 0
* the STOP  sentinel lives conceptually "after" position n-1

Total score of a labeling `y = (y_0, ..., y_{n-1})`:

::

    score(y) = start_weights[y_0]
             + emission_scores[0][y_0]
             + sum_{i=1..n-1} ( transition_weights[y_{i-1}, y_i] + emission_scores[i][y_i] )
             + stop_weights[y_{n-1}]
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from crf.vocabulary import Vocabulary


@dataclass
class CRFParameters:
    """All learnable parameters of the CRF, plus the vocabulary they index."""

    vocab: Vocabulary
    emission_weights: np.ndarray   # (n_observations, n_real_labels)
    transition_weights: np.ndarray  # (n_real_labels, n_real_labels)  u -> v
    start_weights: np.ndarray       # (n_real_labels,)  START -> v
    stop_weights: np.ndarray        # (n_real_labels,)  u -> STOP

    @classmethod
    def zeros(cls, vocab: Vocabulary) -> "CRFParameters":
        """All-zero initialisation — the standard starting point for Algorithm S."""
        n_real = len(vocab.real_label_ids())
        return cls(
            vocab=vocab,
            emission_weights=np.zeros((vocab.n_observations, n_real)),
            transition_weights=np.zeros((n_real, n_real)),
            start_weights=np.zeros(n_real),
            stop_weights=np.zeros(n_real),
        )


# ---------------------------------------------------------------- score pieces

def compute_emission_scores(
    params: CRFParameters,
    observation_ids_per_position: Sequence[Sequence[int]],
) -> np.ndarray:
    """
    Emission score for every (position, real-label) pair.

    Returns an array of shape `(n, R)` where entry `[i, v]` is the sum of
    `emission_weights[k, v]` across every active observation feature `k` at
    position `i`. This is the "how well does this word match label v" term.
    """
    n = len(observation_ids_per_position)
    n_real = params.emission_weights.shape[1]
    emission_scores = np.zeros((n, n_real))
    for i, obs_ids in enumerate(observation_ids_per_position):
        if obs_ids:
            # Sum rows of emission_weights for every active feature; the result
            # is a length-R vector that we drop into emission_scores[i].
            emission_scores[i] = params.emission_weights[list(obs_ids)].sum(axis=0)
    return emission_scores


def score_labeling(
    params: CRFParameters,
    emission_scores: np.ndarray,
    label_ids: Sequence[int],
) -> float:
    """Total (unnormalised) score of one specific labeling for a sentence."""
    labels = np.asarray(label_ids)
    total = float(params.start_weights[labels[0]] + emission_scores[0, labels[0]])
    for i in range(1, len(labels)):
        total += float(
            params.transition_weights[labels[i - 1], labels[i]] + emission_scores[i, labels[i]]
        )
    total += float(params.stop_weights[labels[-1]])
    return total


# ----------------------------------------------------------------------- Viterbi

def viterbi(
    params: CRFParameters,
    emission_scores: np.ndarray,
) -> tuple[list[int], float]:
    """
    Find the highest-scoring labeling for a sentence.

    Returns `(best_labels, best_score)` where `best_labels` is a list of
    real-label IDs of length `n`.

    The algorithm is classical dynamic programming: at each position, remember
    the best score of any path ending in each possible label, plus a backpointer
    to the label at the previous position that achieved that best score. At the
    end, we follow the backpointers from the best final state back to position 0.
    """
    n, n_real = emission_scores.shape
    # V[i, v] = best score of any path ending at label v at position i.
    V = np.full((n, n_real), -np.inf)
    back = np.zeros((n, n_real), dtype=np.int64)

    # Base case: first position. Previous "label" is START, so add start_weights.
    V[0] = params.start_weights + emission_scores[0]

    for i in range(1, n):
        # scores[u, v] = V[i-1, u] + transition[u, v]   (broadcasting u as column)
        scores = V[i - 1][:, None] + params.transition_weights
        # Best previous label for each v.
        back[i] = scores.argmax(axis=0)
        V[i] = scores.max(axis=0) + emission_scores[i]

    # Final step: pick the last label that maximises V[n-1, u] + stop_weights[u].
    final_scores = V[n - 1] + params.stop_weights
    last_label = int(final_scores.argmax())
    best_score = float(final_scores[last_label])

    # Backtrack.
    labels = [0] * n
    labels[n - 1] = last_label
    for i in range(n - 1, 0, -1):
        labels[i - 1] = int(back[i, labels[i]])

    return labels, best_score


# -------------------------------------------------------------- forward-backward

@dataclass
class ForwardBackward:
    """Cached forward-backward outputs for a single sentence."""

    log_alpha: np.ndarray   # (n, R)
    log_beta: np.ndarray    # (n, R)
    log_Z: float
    emission_scores: np.ndarray  # (n, R) — kept alongside for pairwise marginal math


def forward_backward(
    params: CRFParameters,
    emission_scores: np.ndarray,
) -> ForwardBackward:
    """
    Log-space forward-backward.

    `log_alpha[i, v]` = log of the total score of all paths from START to
    position `i` ending at label `v`.

    `log_beta[i, u]`  = log of the total score of all paths from position
    `i` (starting at label `u`) to STOP.

    `log_Z` is the log partition function — the normaliser that turns
    path-scores into probabilities.
    """
    n, n_real = emission_scores.shape

    log_alpha = np.full((n, n_real), -np.inf)
    log_alpha[0] = params.start_weights + emission_scores[0]

    for i in range(1, n):
        # scores[u, v] = log_alpha[i-1, u] + transition[u, v]
        scores = log_alpha[i - 1][:, None] + params.transition_weights
        # logsumexp over the previous label u gives log_alpha[i, v] (minus emission).
        log_alpha[i] = logsumexp(scores, axis=0) + emission_scores[i]

    log_Z_forward = logsumexp(log_alpha[n - 1] + params.stop_weights)

    log_beta = np.full((n, n_real), -np.inf)
    log_beta[n - 1] = params.stop_weights

    for i in range(n - 2, -1, -1):
        # scores[u, v] = transition[u, v] + emission_scores[i+1, v] + log_beta[i+1, v]
        scores = (
            params.transition_weights
            + emission_scores[i + 1][None, :]
            + log_beta[i + 1][None, :]
        )
        log_beta[i] = logsumexp(scores, axis=1)

    log_Z_backward = logsumexp(params.start_weights + emission_scores[0] + log_beta[0])

    # Use the average; the two should agree to ~1e-10 when implemented correctly.
    log_Z = float((log_Z_forward + log_Z_backward) / 2.0)

    return ForwardBackward(
        log_alpha=log_alpha,
        log_beta=log_beta,
        log_Z=log_Z,
        emission_scores=emission_scores,
    )


def pairwise_marginals(
    params: CRFParameters,
    fb: ForwardBackward,
) -> np.ndarray:
    """
    `p(y_{i-1}=u, y_i=v | x)` for every transition i=1..n-1.

    Returns an array of shape `(n-1, R, R)` of probabilities (not logs).
    Used during Algorithm S training to compute expected transition counts.
    """
    n = fb.log_alpha.shape[0]
    if n < 2:
        # No transitions in a length-1 sentence.
        R = params.transition_weights.shape[0]
        return np.zeros((0, R, R))

    log_alpha_prev = fb.log_alpha[:-1]       # (n-1, R)
    emission_curr = fb.emission_scores[1:]   # (n-1, R)
    log_beta_curr = fb.log_beta[1:]          # (n-1, R)

    # Broadcast: (n-1, R, 1) + (R, R) + (n-1, 1, R) + (n-1, 1, R) = (n-1, R, R)
    log_p = (
        log_alpha_prev[:, :, None]
        + params.transition_weights[None, :, :]
        + emission_curr[:, None, :]
        + log_beta_curr[:, None, :]
        - fb.log_Z
    )
    return np.exp(log_p)


def unary_marginals(fb: ForwardBackward) -> np.ndarray:
    """
    `p(y_i = v | x)` for every position i. Shape `(n, R)`.
    """
    return np.exp(fb.log_alpha + fb.log_beta - fb.log_Z)


def start_marginals(params: CRFParameters, fb: ForwardBackward) -> np.ndarray:
    """`p(y_0 = v | x)` — probability the first label is v. Shape `(R,)`."""
    return np.exp(
        params.start_weights + fb.emission_scores[0] + fb.log_beta[0] - fb.log_Z
    )


def stop_marginals(params: CRFParameters, fb: ForwardBackward) -> np.ndarray:
    """`p(y_{n-1} = u | x)` — probability the last label is u. Shape `(R,)`."""
    return np.exp(fb.log_alpha[-1] + params.stop_weights - fb.log_Z)
