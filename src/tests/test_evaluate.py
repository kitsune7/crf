from crf.evaluate import evaluate, format_evaluation


def test_perfect_predictions_give_full_score():
    sentences = [
        {"words": ["a", "b"], "lid": ["lang1", "lang2"]},
        {"words": ["c"], "lid": ["lang2"]},
    ]
    predictions = [["lang1", "lang2"], ["lang2"]]
    result = evaluate(sentences, predictions)
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.weighted_f1 == 1.0


def test_switch_point_accuracy_counts_only_transitions():
    sentences = [
        {"words": ["a", "b", "c"], "lid": ["lang1", "lang2", "lang1"]},
    ]
    # Gold: lang1 lang2 lang1. Switch points: positions 1 and 2.
    # Predictions wrong at position 1 only.
    predictions = [["lang1", "lang1", "lang1"]]
    result = evaluate(sentences, predictions)
    # 1 of 2 switch points correct.
    assert result.switch_point_accuracy == 0.5


def test_no_switch_points_returns_none():
    sentences = [{"words": ["a", "b"], "lid": ["lang1", "lang1"]}]
    predictions = [["lang1", "lang1"]]
    result = evaluate(sentences, predictions)
    assert result.switch_point_accuracy is None


def test_format_contains_all_sections():
    sentences = [{"words": ["a", "b"], "lid": ["lang1", "lang2"]}]
    predictions = [["lang1", "lang1"]]
    out = format_evaluation("test", evaluate(sentences, predictions))
    assert "token accuracy" in out
    assert "macro F1" in out
    assert "confusion matrix" in out
    assert "per-class" in out
