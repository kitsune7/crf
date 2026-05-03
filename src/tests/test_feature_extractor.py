from crf.feature_extractor import (
    extract_features,
    feature_strings,
    is_twitter_specific,
    sentence_feature_strings,
)


def test_extract_features_first_token():
    # "I" is capitalized and all-caps, no previous word, next word is "really"
    tokens = "I really like foxes".split()
    features = extract_features(tokens, 0)
    assert features == {
        "word": "i",
        "prefix3": "I",
        "suffix3": "I",
        "suffix2": "I",
        "is_cap": True,
        "is_all_caps": True,
        "has_digit": False,
        "is_twitter_specific": False,
        "prev_word": "<START>",
        "next_word": "really",
    }


def test_extract_features_middle_token():
    tokens = "I really like foxes".split()
    features = extract_features(tokens, 2)  # "like"
    assert features["word"] == "like"
    assert features["prev_word"] == "really"
    assert features["next_word"] == "foxes"


def test_extract_features_last_token_has_end_marker():
    tokens = "I really like foxes".split()
    features = extract_features(tokens, 3)  # "foxes"
    assert features["next_word"] == "<END>"


def test_extract_features_spanish_suffix():
    # Spanish-looking suffixes are the whole reason we include suffix features.
    tokens = ["encantaría"]
    features = extract_features(tokens, 0)
    assert features["suffix3"] == "ría"
    assert features["suffix2"] == "ía"


def test_is_twitter_specific_flags_mentions_hashtags_urls():
    assert is_twitter_specific("@YiseBabee") is True
    assert is_twitter_specific("#tbt") is True
    assert is_twitter_specific("http://t.co/abc") is True
    assert is_twitter_specific("www.example.com") is True
    assert is_twitter_specific("plainword") is False


def test_feature_strings_emits_true_bools_only():
    tokens = ["CAT"]
    active = feature_strings(tokens, 0)
    # CAT is both capitalized and all-caps; is_twitter_specific / has_digit are False.
    assert "is_cap=true" in active
    assert "is_all_caps=true" in active
    assert "has_digit=true" not in active
    assert "is_twitter_specific=true" not in active
    # Non-boolean features always fire with their value.
    assert "word=cat" in active
    assert "prev_word=<START>" in active
    assert "next_word=<END>" in active


def test_sentence_feature_strings_has_one_entry_per_token():
    tokens = "hello world".split()
    per_token = sentence_feature_strings(tokens)
    assert len(per_token) == 2
    assert all(isinstance(features, list) for features in per_token)
    assert any(feature.startswith("word=") for feature in per_token[0])
