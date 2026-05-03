from collections.abc import Sequence


def extract_features(tokens: Sequence[str], i: int) -> dict:
    """
    Extract features for a specific token in a sentence.
    """
    word = tokens[i]
    features = {
        "word": word.lower(),
        "prefix3": word[:3],
        "suffix3": word[-3:],
        "suffix2": word[-2:],
        "is_cap": word[0].isupper(),
        "is_all_caps": word.isupper(),
        "has_digit": any(char.isdigit() for char in word),
        "is_twitter_specific": is_twitter_specific(word),
        "prev_word": tokens[i - 1].lower() if i > 0 else "<START>",
        "next_word": tokens[i + 1].lower() if i < len(tokens) - 1 else "<END>",
    }

    return features


def is_twitter_specific(word: str) -> bool:
    if word.startswith("@"):
        return True
    if word.startswith("#"):
        return True
    if word.startswith("http"):
        return True
    if word.startswith("www"):
        return True
    return False


def feature_strings(tokens: Sequence[str], i: int) -> list[str]:
    """
    Flatten the feature dict at position `i` into a list of "key=value" strings.

    The CRF ultimately needs every feature to be a stable, unique identifier
    that can be mapped to an integer ID (e.g., "word=tacos", "suffix3=cos"),
    so the vocabulary and the scoring code never drift on formatting.
    Boolean features are emitted only when True — "this thing is present" is
    the signal we want; a feature that is absent simply doesn't fire.
    """
    features = extract_features(tokens, i)
    active: list[str] = []
    for key, value in features.items():
        if isinstance(value, bool):
            if value:
                active.append(f"{key}=true")
        else:
            active.append(f"{key}={value}")
    return active


def sentence_feature_strings(tokens: Sequence[str]) -> list[list[str]]:
    """Return the list of active feature strings for each token in a sentence."""
    return [feature_strings(tokens, i) for i in range(len(tokens))]
