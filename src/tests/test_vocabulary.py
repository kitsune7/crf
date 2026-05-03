from crf.vocabulary import START, STOP, Vocabulary


def _sentences():
    return [
        {"words": ["hola", "amigo"], "lid": ["lang2", "lang2"]},
        {"words": ["hello", "amigo"], "lid": ["lang1", "lang2"]},
    ]


def test_labels_include_start_and_stop():
    vocab = Vocabulary.build(_sentences(), min_count=1)
    assert START in vocab.label_to_id
    assert STOP in vocab.label_to_id
    assert vocab.start_id == vocab.label_to_id[START]
    assert vocab.stop_id == vocab.label_to_id[STOP]
    # Real labels should come before the sentinels.
    assert vocab.start_id > max(vocab.real_label_ids())
    assert vocab.stop_id > max(vocab.real_label_ids())


def test_min_count_prunes_rare_features():
    vocab_keep = Vocabulary.build(_sentences(), min_count=1)
    vocab_prune = Vocabulary.build(_sentences(), min_count=2)
    # "word=amigo" appears in both sentences → survives min_count=2.
    assert "word=amigo" in vocab_keep.observation_to_id
    assert "word=amigo" in vocab_prune.observation_to_id
    # "word=hola" appears once → dropped at min_count=2.
    assert "word=hola" in vocab_keep.observation_to_id
    assert "word=hola" not in vocab_prune.observation_to_id


def test_encode_features_drops_unknown_features():
    vocab = Vocabulary.build(_sentences(), min_count=1)
    # "martian" is unseen; encode should silently skip its word-specific features
    # but the prefix/suffix/context-word features still fire when they match.
    ids = vocab.encode_features(["martian"], 0)
    assert all(isinstance(i, int) for i in ids)
    # None of the returned IDs should be out of range.
    assert all(0 <= i < vocab.n_observations for i in ids)


def test_encode_labels_roundtrip():
    vocab = Vocabulary.build(_sentences(), min_count=1)
    encoded = vocab.encode_labels(["lang1", "lang2"])
    reverse = vocab.id_to_label()
    assert [reverse[i] for i in encoded] == ["lang1", "lang2"]


def test_real_label_ids_excludes_sentinels():
    vocab = Vocabulary.build(_sentences(), min_count=1)
    real_ids = vocab.real_label_ids()
    assert vocab.start_id not in real_ids
    assert vocab.stop_id not in real_ids
    # Every training label should appear as a real label.
    assert vocab.label_to_id["lang1"] in real_ids
    assert vocab.label_to_id["lang2"] in real_ids
