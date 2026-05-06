# crf

A from-scratch linear-chain Conditional Random Field implementation, following [Lafferty, McCallum & Pereira (2001)](https://nlp.cs.nyu.edu/nycnlp/lafferty01conditional.pdf). Includes log-space forward-backward, Viterbi decoding, Algorithm S (iterative scaling) training, and an MEMM baseline that shares the same feature templates.

The task is token-level language identification on Spanish-English code-switched tweets (LinCE `LID_spaeng`).

## Project setup

### Prerequisites

- Python 3.13 (pinned in `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) for dependency management

### Install

```bash
# clone and enter the repo, then:
uv sync
```

`uv sync` creates `.venv/` and installs the runtime dependencies
(`numpy`, `pandas`, `scikit-learn`, `scipy`) plus the dev tools (`pytest`,
`ruff`) pinned in `uv.lock`.

### Reproducing the results

All commands read from `data/lid_spaeng_*.csv` by default and print metrics to stdout. To reproduce the numbers shown in `presentation/results.png`, run the CRF training command from the repo root:

```bash
uv run crf train
```

This matches the configuration used for the presentation:

- full training set (21,030 sentences)
- `--min-count 2` (drops observation features seen fewer than 2 times)
- `--max-iters 200` (Algorithm S iterations)

The command prints, for both `train` and `validation` splits:

- token accuracy, weighted and macro F1
- switch-point accuracy (accuracy at positions where the gold label changes)
- the full sklearn-style per-class classification report
- a gold-vs-predicted confusion matrix

Expect about **~1 hour** of wall-clock training time on a modern laptop. Log-likelihood should rise monotonically from `-inf` toward `-78,473` — this is the end-to-end correctness signal for the forward-backward and Algorithm S implementations.

### Other useful commands

```bash
# Label distribution across all three splits
uv run crf label-stats

# MEMM baseline sharing the CRF's feature extractor
uv run crf baseline

# Fast dev loop — first 500 sentences, 50 iterations
uv run crf train --subset 500 --max-iters 50

# Run the test suite
uv run pytest
```

## Dataset

The experiments use the `LID_spaeng` task from the [LinCE benchmark](https://ritual.uh.edu/lince/) — Spanish-English code-switched tweets annotated with token-level language IDs. The train and validation tokens are tagged with one of eight labels: `lang1` (English), `lang2` (Spanish), `other`, `ne` (named entity), `ambiguous`, `mixed`, `fw` (foreign word), or `unk`. The committed test split has tokens only; its labels are blank placeholders.

| Split      | Sentences | Tokens  |
|------------|----------:|--------:|
| Train      |    21,030 | 253,221 |
| Validation |     3,332 |  40,391 |
| Test       |     8,289 |  97,341 |

### Why the data is committed directly

The original LinCE distribution has not been reliably accessible during the life of this project — the download endpoint has been intermittent, and the upstream repository has shuffled file layouts. To keep the experiments reproducible, the three splits used here are committed directly into `data/` as CSVs (`lid_spaeng_train.csv`, `lid_spaeng_validation.csv`, `lid_spaeng_test.csv`). Each row stores a single tweet as a stringified list of tokens and a parallel `lid` list; train and validation contain gold language IDs, while test contains blank placeholders. `src/crf/data_loader.py` parses those stringified lists back into Python lists. The CSVs are static snapshots — no network access is required to run anything in this repo.

## Experiment Results

### What the data looks like

Each example is a short tweet (median length ~12 tokens) with parallel `words` and `lid` lists of equal length. The label distribution across the training split is extremely imbalanced:

| Label       | Share of tokens |
|-------------|----------------:|
| `lang2` (es)|          44.6 % |
| `lang1` (en)|          31.8 % |
| `other`     |          21.4 % |
| `ne`        |           2.1 % |
| `ambiguous` |         < 0.1 % |
| `mixed`     |         < 0.1 % |
| `fw`        |         < 0.1 % |
| `unk`       |         < 0.1 % |

97.7 % of the corpus is just `lang1` / `lang2` / `other`. The four rare
classes have single-digit to low-double-digit token counts in validation,
which is what makes macro-F1 brutal regardless of how well the head of
the distribution is modeled.

### Headline numbers

Training for 200 Algorithm-S iterations on the full training set
(~3,696 s) produced these numbers:

| Metric              | Train  | Validation |
|---------------------|-------:|-----------:|
| Token accuracy      | 0.9629 | **0.9599** |
| Weighted F1         | 0.9562 | **0.9521** |
| Macro F1            | 0.4019 |     0.3879 |
| Switch-point acc    | 0.9017 | **0.8916** |
| Final log-likelihood| \_     | –78,473.00 |

The near-identical train/validation numbers say the CRF is not badly
overfitting, though the MEMM baseline below shows that optimization is a
larger limitation than variance on this run.

### Per-class breakdown (validation)

| Label        | Precision | Recall |  F1  | Support |
|--------------|----------:|-------:|-----:|--------:|
| `lang1` (en) |      0.96 |   0.98 | 0.97 |  16,712 |
| `lang2` (es) |      0.94 |   0.99 | 0.96 |  14,955 |
| `other`      |      0.99 |   0.97 | 0.98 |   7,830 |
| `ne`         |      0.95 |   0.11 | 0.19 |     815 |
| `ambiguous`  |      0.00 |   0.00 | 0.00 |      39 |
| `mixed`      |      0.00 |   0.00 | 0.00 |       6 |
| `fw`         |      0.00 |   0.00 | 0.00 |       2 |
| `unk`        |      0.00 |   0.00 | 0.00 |      32 |

The head of the distribution is solved; `ne` is partially recovered, while the very rare tail is still all zeros.

### What the confusion matrix shows (validation)

Reading the most-populated rows of the gold-vs-predicted confusion matrix:

- `lang1 → lang1`: 16,353 / 16,712 correct
- `lang2 → lang2`: 14,736 / 14,955 correct — most of the residual 219
  errors go to `lang1`, a plausible English/Spanish ambiguity
- `other → other`: 7,597 / 7,830 correct — URLs, punctuation, emoji, and
  Twitter artifacts are picked up cleanly by the `is_twitter_specific`
  feature
- `ne → ne`: 86 / 815 correct — named entities are no longer entirely
  collapsed, but most still go to `lang1` or `lang2`. This remains the
  single biggest failure mode and dominates the macro-F1 drag.

![full metric dump from `uv run crf train`](presentation/results.png)

### Baseline comparison

The MEMM baseline uses the same observation features plus a `prev_label`
indicator, trains a multinomial logistic regression over gold previous labels,
and decodes with Viterbi under local normalization:

| Metric           | CRF validation | MEMM validation |
|------------------|---------------:|----------------:|
| Token accuracy   |         0.9599 |      **0.9762** |
| Weighted F1      |         0.9521 |      **0.9744** |
| Macro F1         |         0.3879 |      **0.4796** |
| Switch-point acc |         0.8916 |      **0.9369** |
| `ne` F1          |         0.19   |      **0.69**   |

![full metric dump from `uv run crf baseline`](presentation/baseline-results.png)

### When CRF works well

1. **On the dominant classes.** `lang1`, `lang2`, and `other` all sit at
   0.96–0.98 F1. The per-token features (`word`, `prefix3`, `suffix2/3`,
   `is_cap`, `is_all_caps`, `has_digit`, `is_twitter_specific`,
   `prev_word`, `next_word`) carry most of the load for these classes.

2. **At switch points.** Validation switch-point accuracy is **89.2 %**
   — positions where the gold label flips between tokens. The CRF's
   transition weights are being used, though the locally normalized MEMM
   baseline does even better on this run.

3. **As a correctness signal.** Log-likelihood rises monotonically for
   all 200 iterations of Algorithm S. Any sign error, marginal bug, or
   partition-function mistake breaks monotonicity immediately, so this
   serves as a built-in end-to-end correctness test for both
   forward-backward and the scaling update.

### When CRF struggles

1. **Named entities.** Recall on `ne` is still low at 0.11. Casual
   Twitter capitalization makes `is_cap` a weak signal, there's no
   gazetteer or character-level LM, and the `ne → ne` transition sees too
   little evidence to fire consistently. Fixing this would take richer
   features (gazetteers, character models), not just a better training
   algorithm.

2. **Rare classes.** `ambiguous`, `mixed`, `fw`, `unk` all collapse to
   zero F1. Algorithm S updates a feature's weight by
   `log(observed / expected) / S`; when observed counts are tiny and
   noisy, updates are tiny and the model learns the prior "just never
   predict this." Class reweighting or an L2-regularized L-BFGS
   objective with class weights would probably help marginally, but the
   bottleneck is data.

3. **Training cost.** ~62 min for 200 iterations on 21 k sentences.
   Algorithm S pays an O(N · R²) cost per iteration (R = number of
   labels) for forward-backward plus expected-count accumulation. This
   is historically faithful to the paper but is roughly 10–100× slower
   than L-BFGS on the same objective. The field moved off iterative
   scaling within a couple of years of the paper's publication for
   exactly this reason.
