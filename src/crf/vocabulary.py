"""
Vocabulary: maps observation feature strings and labels to integer IDs.

Why this file exists
--------------------
The CRF only ever deals in integers once we leave feature extraction. Two
distinct ID spaces:

1. **Observation IDs** — one per unique feature string ("word=tacos",
   "suffix3=cos", ...). The scoring code pairs an observation ID with a
   candidate label to look up an emission weight. We intentionally do *not*
   bake the label into the observation ID here; that pairing happens in the
   weight matrix (shape: num_observations x num_labels).

2. **Label IDs** — one per language-ID tag. We also reserve two sentinel
   IDs for START and STOP. Those are "virtual" labels used only at the
   boundaries of the sentence, so forward-backward / Viterbi never have to
   special-case position 0 or position n+1.

Rare observation features (seen fewer than `min_count` times in training)
are dropped — they're usually overfitting fodder and inflate the weight
matrix for no gain.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from crf.feature_extractor import feature_strings

START = "<START>"
STOP = "<STOP>"


@dataclass
class Vocabulary:
    """Bidirectional mapping for observation features and labels."""

    observation_to_id: dict[str, int] = field(default_factory=dict)
    label_to_id: dict[str, int] = field(default_factory=dict)

    # Derived / convenience attributes filled in after building:
    start_id: int = -1
    stop_id: int = -1

    # ------------------------------------------------------------------ lookup
    @property
    def n_observations(self) -> int:
        return len(self.observation_to_id)

    @property
    def n_labels(self) -> int:
        """Number of labels *including* the START and STOP sentinels."""
        return len(self.label_to_id)

    def id_to_label(self) -> list[str]:
        """Reverse lookup as a list indexed by label ID."""
        reverse = [""] * self.n_labels
        for label, idx in self.label_to_id.items():
            reverse[idx] = label
        return reverse

    def real_label_ids(self) -> list[int]:
        """All label IDs except START / STOP — the labels a token can emit."""
        return [idx for label, idx in self.label_to_id.items() if label not in (START, STOP)]

    def encode_features(self, tokens: Sequence[str], i: int) -> list[int]:
        """
        Active observation IDs at position `i`.

        Unknown features (not seen during training, or pruned for low count)
        are silently dropped. This is our simple OOV policy: a word never
        seen before just contributes whatever *other* features still fire
        (prefix/suffix/capitalization/etc.).
        """
        ids: list[int] = []
        for feature in feature_strings(tokens, i):
            idx = self.observation_to_id.get(feature)
            if idx is not None:
                ids.append(idx)
        return ids

    def encode_labels(self, labels: Sequence[str]) -> list[int]:
        return [self.label_to_id[label] for label in labels]

    # ----------------------------------------------------------------- builder
    @classmethod
    def build(
        cls,
        sentences: Iterable[dict],
        min_count: int = 2,
    ) -> "Vocabulary":
        """
        Build a Vocabulary from training sentences.

        `sentences` is an iterable of {"words": [...], "lid": [...]} dicts —
        the shape produced by `crf.data_loader.load_data`.
        """
        feature_counts: dict[str, int] = {}
        label_set: set[str] = set()

        for sentence in sentences:
            tokens = sentence["words"]
            labels = sentence["lid"]
            for i in range(len(tokens)):
                for feature in feature_strings(tokens, i):
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1
            label_set.update(labels)

        # Sort for deterministic IDs — makes tests and saved models stable.
        observation_to_id = {
            feature: idx
            for idx, feature in enumerate(
                sorted(feature for feature, count in feature_counts.items() if count >= min_count)
            )
        }

        # Real labels first (sorted), then START / STOP — keeps real IDs in
        # a contiguous range starting at 0, which is convenient when we want
        # to iterate only over emittable labels.
        label_to_id: dict[str, int] = {}
        for label in sorted(label_set):
            label_to_id[label] = len(label_to_id)
        start_id = len(label_to_id)
        label_to_id[START] = start_id
        stop_id = len(label_to_id)
        label_to_id[STOP] = stop_id

        return cls(
            observation_to_id=observation_to_id,
            label_to_id=label_to_id,
            start_id=start_id,
            stop_id=stop_id,
        )
