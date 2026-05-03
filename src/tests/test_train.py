"""
Tests for Algorithm S training.

The key invariants we verify:

1. Log-likelihood is monotonically non-decreasing iteration to iteration.
   This is the canary — if it ever drops, there's a bug upstream.
2. The model can overfit a tiny training set to ~100% accuracy. The plan
   calls this the "machinery works" smoke test.
3. Observed and expected counts match after training converges — the defining
   condition of Algorithm S.
"""

import numpy as np
import pytest

from crf.train import (
    compute_expected_counts_and_loglik,
    compute_observed_counts,
    encode_sentences,
    predict,
    train,
)
from crf.vocabulary import Vocabulary


def _tiny_dataset():
    return [
        {"words": ["I", "love", "tacos"], "lid": ["lang1", "lang1", "lang2"]},
        {"words": ["hola", "amigo"], "lid": ["lang2", "lang2"]},
        {"words": ["hello", "world"], "lid": ["lang1", "lang1"]},
        {"words": ["comer", "tacos"], "lid": ["lang2", "lang2"]},
        {"words": ["I", "eat", "mucho"], "lid": ["lang1", "lang1", "lang2"]},
    ]


def test_loglikelihood_is_monotone():
    data = _tiny_dataset()
    vocab = Vocabulary.build(data, min_count=1)
    encoded = encode_sentences(vocab, data)

    _, history = train(vocab, encoded, max_iters=15)
    lls = history.loglik
    # Allow a tiny numerical slop, but not a real drop.
    for earlier, later in zip(lls, lls[1:]):
        assert later >= earlier - 1e-8, f"log-likelihood decreased: {earlier} -> {later}"


def test_can_overfit_tiny_dataset():
    """With enough iterations on tiny data, training accuracy should hit ~100%."""
    data = _tiny_dataset()
    vocab = Vocabulary.build(data, min_count=1)
    encoded = encode_sentences(vocab, data)

    params, _ = train(vocab, encoded, max_iters=200, tol=0.0)

    predictions = predict(params, data)
    total = sum(len(sentence["lid"]) for sentence in data)
    correct = sum(
        1
        for sentence, prediction in zip(data, predictions)
        for gold, pred in zip(sentence["lid"], prediction)
        if gold == pred
    )
    assert correct / total >= 0.99


def test_converged_model_matches_observed_and_expected_counts():
    """
    After (close to) convergence, observed[k] and expected[k] should agree for
    every feature that ever fires. This is Algorithm S's fixed-point condition.
    """
    data = _tiny_dataset()
    vocab = Vocabulary.build(data, min_count=1)
    encoded = encode_sentences(vocab, data)

    params, _ = train(vocab, encoded, max_iters=300, tol=0.0)

    observed = compute_observed_counts(vocab, encoded)
    expected, _ = compute_expected_counts_and_loglik(params, encoded)

    # Only compare features that actually fired on gold; features with
    # observed == 0 are skipped by the updater so they won't have converged.
    mask = observed.emission > 0
    assert np.allclose(
        observed.emission[mask], expected.emission[mask], atol=5e-2
    )


def test_predict_returns_label_strings_for_every_token():
    data = _tiny_dataset()
    vocab = Vocabulary.build(data, min_count=1)
    encoded = encode_sentences(vocab, data)
    params, _ = train(vocab, encoded, max_iters=5)
    predictions = predict(params, data)
    for sentence, prediction in zip(data, predictions):
        assert len(prediction) == len(sentence["words"])
        assert all(isinstance(label, str) for label in prediction)


def test_observed_counts_shapes_and_totals():
    data = _tiny_dataset()
    vocab = Vocabulary.build(data, min_count=1)
    encoded = encode_sentences(vocab, data)
    observed = compute_observed_counts(vocab, encoded)

    n_real = len(vocab.real_label_ids())
    assert observed.emission.shape == (vocab.n_observations, n_real)
    assert observed.transition.shape == (n_real, n_real)
    assert observed.start.shape == (n_real,)
    assert observed.stop.shape == (n_real,)

    # One start and one stop per sentence.
    assert observed.start.sum() == len(data)
    assert observed.stop.sum() == len(data)
    # Each sentence of length n contributes n-1 transitions.
    expected_transitions = sum(max(len(sentence["lid"]) - 1, 0) for sentence in data)
    assert observed.transition.sum() == expected_transitions


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_training_improves_initial_loglikelihood(seed):
    """One step of training must not make log-likelihood worse than the starting point."""
    rng = np.random.default_rng(seed)
    data = [
        {
            "words": [rng.choice(["hi", "hola", "tacos", "world"]) for _ in range(3)],
            "lid": [rng.choice(["lang1", "lang2"]) for _ in range(3)],
        }
        for _ in range(5)
    ]
    vocab = Vocabulary.build(data, min_count=1)
    encoded = encode_sentences(vocab, data)
    _, history = train(vocab, encoded, max_iters=5)
    assert history.loglik[-1] >= history.loglik[0] - 1e-8
