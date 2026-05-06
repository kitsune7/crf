"""
Tests for the CRF inference engine.

The single most important test here is `test_log_Z_consistency` — the plan
calls it out as the thing that catches 90% of forward-backward bugs. If that
passes, the rest of the machinery (Viterbi, marginals, training) has solid
ground to stand on.
"""

import numpy as np
import pytest
from scipy.special import logsumexp

from crf.inference import (
    CRFParameters,
    compute_emission_scores,
    forward_backward,
    pairwise_marginals,
    score_labeling,
    start_marginals,
    stop_marginals,
    unary_marginals,
    viterbi,
)
from crf.vocabulary import Vocabulary


def _toy_params():
    """Build a tiny vocab + random (but fixed) parameters to exercise the math."""
    sentences = [
        {"words": ["hola", "mundo"], "lid": ["lang2", "lang2"]},
        {"words": ["hello", "world"], "lid": ["lang1", "lang1"]},
        {"words": ["hello", "mundo"], "lid": ["lang1", "lang2"]},
    ]
    vocab = Vocabulary.build(sentences, min_count=1)
    rng = np.random.default_rng(0)
    n_real = len(vocab.real_label_ids())
    params = CRFParameters(
        vocab=vocab,
        emission_weights=rng.normal(size=(vocab.n_observations, n_real)),
        transition_weights=rng.normal(size=(n_real, n_real)),
        start_weights=rng.normal(size=n_real),
        stop_weights=rng.normal(size=n_real),
    )
    return vocab, params


def _encode(vocab, tokens):
    return [vocab.encode_features(tokens, i) for i in range(len(tokens))]


def test_log_Z_consistency():
    """
    Forward and backward must agree on log_Z, and at any position i,
    logsumexp(log_alpha[i] + log_beta[i]) must also equal log_Z.
    """
    vocab, params = _toy_params()
    tokens = ["hello", "mundo"]
    obs = _encode(vocab, tokens)
    emissions = compute_emission_scores(params, obs)
    fb = forward_backward(params, emissions)

    # Forward and backward computed log_Z must agree to ~1e-10.
    log_Z_forward = logsumexp(fb.log_alpha[-1] + params.stop_weights)
    log_Z_backward = logsumexp(params.start_weights + emissions[0] + fb.log_beta[0])
    assert np.isclose(log_Z_forward, log_Z_backward, atol=1e-10)
    assert np.isclose(fb.log_Z, log_Z_forward, atol=1e-10)

    # At any position i, logsumexp(alpha[i] + beta[i]) = log_Z.
    for i in range(emissions.shape[0]):
        lse = logsumexp(fb.log_alpha[i] + fb.log_beta[i])
        assert np.isclose(lse, fb.log_Z, atol=1e-10)


def test_log_Z_equals_logsumexp_over_all_labelings():
    """
    With a tiny label set we can enumerate every possible labeling and check
    that log_Z matches the brute-force log-sum-exp of all labeling scores.
    """
    vocab, params = _toy_params()
    tokens = ["hello", "mundo"]
    obs = _encode(vocab, tokens)
    emissions = compute_emission_scores(params, obs)
    fb = forward_backward(params, emissions)

    n_real = params.transition_weights.shape[0]
    scores = []
    for y0 in range(n_real):
        for y1 in range(n_real):
            scores.append(score_labeling(params, emissions, [y0, y1]))
    brute = logsumexp(scores)
    assert np.isclose(fb.log_Z, brute, atol=1e-10)


def test_marginals_sum_to_one():
    vocab, params = _toy_params()
    tokens = ["hello", "world", "amigo"]
    # Not every word is in vocab; encode_features will just skip unknowns.
    obs = _encode(vocab, tokens)
    emissions = compute_emission_scores(params, obs)
    fb = forward_backward(params, emissions)

    uni = unary_marginals(fb)
    assert np.allclose(uni.sum(axis=1), 1.0, atol=1e-9)

    pair = pairwise_marginals(params, fb)
    assert pair.shape == (len(tokens) - 1, uni.shape[1], uni.shape[1])
    assert np.allclose(pair.sum(axis=(1, 2)), 1.0, atol=1e-9)

    assert np.isclose(start_marginals(params, fb).sum(), 1.0, atol=1e-9)
    assert np.isclose(stop_marginals(params, fb).sum(), 1.0, atol=1e-9)


def test_pairwise_marginals_marginalize_to_unary():
    """Summing pairwise p(y_{i-1}, y_i) over y_{i-1} should give unary p(y_i)."""
    vocab, params = _toy_params()
    tokens = ["hello", "world", "mundo"]
    obs = _encode(vocab, tokens)
    emissions = compute_emission_scores(params, obs)
    fb = forward_backward(params, emissions)

    uni = unary_marginals(fb)
    pair = pairwise_marginals(params, fb)

    # pair[i-1].sum(axis=0) = p(y_i) for i = 1..n-1
    for i in range(1, len(tokens)):
        summed = pair[i - 1].sum(axis=0)
        assert np.allclose(summed, uni[i], atol=1e-9)

    # And summing the first pair over y_i gives p(y_0).
    assert np.allclose(pair[0].sum(axis=1), uni[0], atol=1e-9)


def test_viterbi_finds_best_labeling_vs_brute_force():
    vocab, params = _toy_params()
    tokens = ["hello", "mundo"]
    obs = _encode(vocab, tokens)
    emissions = compute_emission_scores(params, obs)

    labels, score = viterbi(params, emissions)

    n_real = params.transition_weights.shape[0]
    best_score = -np.inf
    best_labels = None
    for y0 in range(n_real):
        for y1 in range(n_real):
            s = score_labeling(params, emissions, [y0, y1])
            if s > best_score:
                best_score = s
                best_labels = [y0, y1]

    assert np.isclose(score, best_score, atol=1e-10)
    assert labels == best_labels


def test_viterbi_on_length_one_sentence():
    vocab, params = _toy_params()
    tokens = ["hello"]
    obs = _encode(vocab, tokens)
    emissions = compute_emission_scores(params, obs)
    labels, score = viterbi(params, emissions)

    # Manual check: score should be start[v] + emission[0,v] + stop[v], maximised over v.
    manual = params.start_weights + emissions[0] + params.stop_weights
    assert np.isclose(score, manual.max(), atol=1e-10)
    assert labels == [int(manual.argmax())]


def test_score_labeling_matches_manual_expansion():
    vocab, params = _toy_params()
    tokens = ["hello", "mundo"]
    obs = _encode(vocab, tokens)
    emissions = compute_emission_scores(params, obs)

    label_ids = [0, 1]  # arbitrary real labels
    expected = (
        params.start_weights[label_ids[0]]
        + emissions[0, label_ids[0]]
        + params.transition_weights[label_ids[0], label_ids[1]]
        + emissions[1, label_ids[1]]
        + params.stop_weights[label_ids[1]]
    )
    assert np.isclose(score_labeling(params, emissions, label_ids), expected, atol=1e-12)


def test_zero_weights_give_uniform_distribution():
    """With all-zero weights, every labeling has score 0 → uniform marginals."""
    vocab, _ = _toy_params()
    params = CRFParameters.zeros(vocab)
    tokens = ["hello", "mundo"]
    obs = _encode(vocab, tokens)
    emissions = compute_emission_scores(params, obs)
    fb = forward_backward(params, emissions)

    n_real = params.transition_weights.shape[0]
    uni = unary_marginals(fb)
    assert np.allclose(uni, 1.0 / n_real, atol=1e-9)

    # log_Z should equal log(R^n) when everything is zero.
    n = emissions.shape[0]
    assert np.isclose(fb.log_Z, n * np.log(n_real), atol=1e-10)


@pytest.mark.parametrize("length", [1, 2, 3, 5])
def test_forward_backward_handles_various_lengths(length):
    vocab, params = _toy_params()
    tokens = ["hello"] * length
    obs = _encode(vocab, tokens)
    emissions = compute_emission_scores(params, obs)
    fb = forward_backward(params, emissions)

    # At every position, alpha+beta - log_Z should give a valid log-probability
    # distribution (sums to 1 after exponentiation).
    uni = unary_marginals(fb)
    assert np.allclose(uni.sum(axis=1), 1.0, atol=1e-9)
