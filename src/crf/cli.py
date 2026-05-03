"""
Command-line interface for training and evaluating the CRF and baseline.

Usage examples::

    uv run crf label-stats
    uv run crf baseline --min-count 2
    uv run crf train --max-iters 30 --min-count 2
    uv run crf train --subset 500 --max-iters 50  # quick dev loop

All commands load data from ``data/lid_spaeng_*.csv`` by default.
"""

import argparse
import sys
import time
from collections import Counter
from collections.abc import Sequence

from crf import data_loader
from crf.baseline import predict_baseline, train_baseline
from crf.evaluate import evaluate, format_evaluation
from crf.train import encode_sentences, predict, train
from crf.vocabulary import Vocabulary


def _load_splits() -> dict[str, list[dict]]:
    return data_loader.load_data_split()


def _maybe_subset(sentences: Sequence[dict], subset: int | None) -> list[dict]:
    if subset is None or subset >= len(sentences):
        return list(sentences)
    return list(sentences[:subset])


# --------------------------------------------------------------------- commands

def cmd_label_stats(_args: argparse.Namespace) -> int:
    splits = _load_splits()
    for split_name, sentences in splits.items():
        counts: Counter[str] = Counter()
        for sentence in sentences:
            counts.update(sentence["lid"])
        total = sum(counts.values())
        print(f"[{split_name}] {len(sentences)} sentences, {total} tokens")
        for label, count in counts.most_common():
            pct = 100.0 * count / total
            print(f"    {label:>10}  {count:>7}  ({pct:5.2f}%)")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    splits = _load_splits()
    train_data = _maybe_subset(splits["train"], args.subset)
    print(f"training LR baseline on {len(train_data)} sentences")
    start = time.time()
    model = train_baseline(train_data, min_count=args.min_count, max_iter=args.max_iter)
    print(f"  done in {time.time() - start:.1f}s  (features={model.n_features})")

    for split_name in ("train", "validation"):
        sentences = splits[split_name]
        if split_name == "train":
            sentences = train_data
        predictions = predict_baseline(model, sentences)
        report = evaluate(sentences, predictions)
        print(format_evaluation(f"baseline / {split_name}", report))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    splits = _load_splits()
    train_data = _maybe_subset(splits["train"], args.subset)
    print(f"training CRF on {len(train_data)} sentences")

    build_start = time.time()
    vocab = Vocabulary.build(train_data, min_count=args.min_count)
    encoded = encode_sentences(vocab, train_data)
    print(
        f"  vocab: {vocab.n_observations} observation features, "
        f"{len(vocab.real_label_ids())} real labels "
        f"(+ START/STOP) in {time.time() - build_start:.1f}s"
    )

    train_start = time.time()
    params, history = train(
        vocab,
        encoded,
        max_iters=args.max_iters,
        verbose=True,
    )
    print(f"  training loop: {time.time() - train_start:.1f}s, "
          f"final log-likelihood={history.loglik[-1]:.4f}")

    for split_name in ("train", "validation"):
        sentences = splits[split_name]
        if split_name == "train":
            sentences = train_data
        predictions = predict(params, sentences)
        report = evaluate(sentences, predictions)
        print(format_evaluation(f"crf / {split_name}", report))
    return 0


# --------------------------------------------------------------------- wiring

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CRF for LinCE LID_spaeng")
    sub = parser.add_subparsers(dest="command", required=True)

    label_stats = sub.add_parser("label-stats", help="print label distributions per split")
    label_stats.set_defaults(func=cmd_label_stats)

    baseline = sub.add_parser("baseline", help="train the LR baseline and report metrics")
    baseline.add_argument("--subset", type=int, default=None, help="use only the first N training sentences")
    baseline.add_argument("--min-count", type=int, default=2)
    baseline.add_argument("--max-iter", type=int, default=200)
    baseline.set_defaults(func=cmd_baseline)

    train_cmd = sub.add_parser("train", help="train the CRF with Algorithm S")
    train_cmd.add_argument("--subset", type=int, default=None, help="use only the first N training sentences")
    train_cmd.add_argument("--min-count", type=int, default=2)
    train_cmd.add_argument("--max-iters", type=int, default=50)
    train_cmd.set_defaults(func=cmd_train)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
