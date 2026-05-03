from crf.baseline import predict_baseline, train_baseline


def _tiny_dataset():
    return [
        {"words": ["I", "love", "tacos"], "lid": ["lang1", "lang1", "lang2"]},
        {"words": ["hola", "amigo"], "lid": ["lang2", "lang2"]},
        {"words": ["hello", "world"], "lid": ["lang1", "lang1"]},
        {"words": ["comer", "tacos"], "lid": ["lang2", "lang2"]},
        {"words": ["I", "eat", "mucho"], "lid": ["lang1", "lang1", "lang2"]},
    ]


def test_baseline_predictions_shape_matches_input():
    data = _tiny_dataset()
    model = train_baseline(data, min_count=1, max_iter=200)
    predictions = predict_baseline(model, data)
    assert len(predictions) == len(data)
    for sentence, prediction in zip(data, predictions):
        assert len(prediction) == len(sentence["words"])


def test_baseline_overfits_tiny_dataset():
    """A sufficiently expressive baseline should get ~perfect training accuracy here."""
    data = _tiny_dataset()
    model = train_baseline(data, min_count=1, max_iter=500)
    predictions = predict_baseline(model, data)
    total = sum(len(sentence["lid"]) for sentence in data)
    correct = sum(
        1
        for sentence, prediction in zip(data, predictions)
        for gold, pred in zip(sentence["lid"], prediction)
        if gold == pred
    )
    assert correct / total >= 0.9


def test_baseline_handles_unknown_words_at_prediction_time():
    data = _tiny_dataset()
    model = train_baseline(data, min_count=1, max_iter=200)
    unseen = [{"words": ["martian", "ufo"], "lid": ["lang1", "lang1"]}]
    predictions = predict_baseline(model, unseen)
    assert len(predictions) == 1
    assert len(predictions[0]) == 2
    # Labels should all be known labels — we don't invent anything new.
    for label in predictions[0]:
        assert label in model.label_to_id
