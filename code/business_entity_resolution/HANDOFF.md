# Amazon ML Challenge handoff

Updated September 27, 2026. This document is the current state of the business
entity resolution project and the checklist for producing the first hackathon
submission.

## Current status

The pipeline has a complete trained baseline and a frozen test-submission runner.
The full-test run has already been started locally on the Mac and is currently in
the `prepare` stage. Do not launch another run against the same run directory.
Its live artifacts are under:

`artifacts/local-fast-pilot/`

The latest observed state was:

- test preparation was reading Source 2 and had passed 900,000 records;
- about 55 GiB of free disk remained;
- `reports/test_pipeline_runtime.json` reports `status: RUNNING` and records the
  active stage;
- the supervising script is `experiments/run_submission.py`.

If the process was interrupted, resume the same command. The driver verifies the
model, threshold, configuration, and source hashes before every stage and reuses
completed immutable artifacts.

## If another agent takes over

Start here instead of launching a new experiment:

1. Read this file, `README.md`, the repository-level `plan.md`, and `ps.md`.
2. Check `artifacts/local-fast-pilot/reports/test_pipeline_runtime.json` and the
   `test_*.log` files. If the status is `RUNNING`, monitor the existing process;
   do not start a second run in that directory.
3. If the status is `FAILED` or the process stopped, rerun the exact
   `experiments/run_submission.py` command below. It resumes completed stages.
4. If the status is `PASS`, inspect both output TSVs, confirm the validator output,
   and build the clean submission package. Do not retune the threshold after this
   point without creating a new run directory and preserving the frozen baseline.
5. If `local-fast-pilot` or its linked `local-full-pilot` corpus is missing, use
   the fresh-device instructions below. Do not attempt to reconstruct links by
   guessing paths.
6. Do not edit files under `src/` while the frozen test run is active. The runner
   hashes the source before every stage, and a source change invalidates the run.
7. Keep generated artifacts and the challenge dataset out of the Git commit.
   Only source, configs, tests, documentation, requirements, and licenses belong
   in the repository.

The immediate goal is one valid, reproducible submission. Model improvements,
GPU experiments, and candidate-policy changes come only after that result is
validated and archived.

## Starting from a fresh device

There are two valid starting points. Use the first when transferring the already
trained baseline; use the second when reproducing the model from source.

### Option A: transfer the frozen run

The Git repository intentionally excludes the dataset, challenge validator, and
generated artifacts. From the new device:

```bash
git clone https://github.com/Fane1824/amazon_ml.git
cd amazon_ml
```

Copy the challenge data so the layout is exactly:

```text
amazon_ml/
├── dataset/
│   ├── train/train_source1.tsv
│   ├── train/train_source2.tsv
│   ├── train/train_source3.tsv
│   ├── train/train_ground_truth.tsv
│   └── test/test_source1.tsv, test_source2.tsv, test_source3.tsv
└── student_resource/utils/validate_submission.py
```

Copy `code/business_entity_resolution/artifacts/local-fast-pilot/` from the
original device as well as `artifacts/local-full-pilot/`, because the linked train
preparation and indexes depend on the latter. Preserve the relative directory
layout. A standalone transfer can dereference the links first:

```bash
cp -RL code/business_entity_resolution/artifacts/local-fast-pilot \
  /path/to/standalone/local-fast-pilot
```

Install Python 3.12 and create the tested environment:

```bash
cd code/business_entity_resolution
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pytest tests -q
```

If the frozen artifacts are present, run the test stages with the environment's
`python` executable. Do not use the Mac-specific `/private/tmp/ber312/bin/python`
path from the original machine:

```bash
python experiments/run_submission.py \
  --run-dir artifacts/local-fast-pilot \
  --data-dir ../../dataset \
  --validator ../../student_resource/utils/validate_submission.py
```

The runner is resumable. It refuses to continue if the model, threshold, config,
or source code changed, and it stops before disk space falls below 5 GiB. Keep at
least 50 GiB free for this workflow; 80–100 GiB is more comfortable for a fresh
full run and temporary copies. An NVIDIA GPU is optional for this baseline; CPU
retrieval and CatBoost are the tested path.

### Option B: reproduce the trained baseline

If no model or run artifacts are transferred, prepare the full data and train a
new frozen baseline. The following commands assume the shell is in
`code/business_entity_resolution` with `.venv` activated:

```bash
python src/main.py prepare --data-dir ../../dataset --run-dir artifacts/fresh-baseline --split train
python src/main.py index --data-dir ../../dataset --run-dir artifacts/fresh-baseline --split train
python src/main.py retrieve --data-dir ../../dataset --run-dir artifacts/fresh-baseline --part train
python src/main.py features --data-dir ../../dataset --run-dir artifacts/fresh-baseline --part train
python src/main.py retrieve --data-dir ../../dataset --run-dir artifacts/fresh-baseline --part dev
python src/main.py features --data-dir ../../dataset --run-dir artifacts/fresh-baseline --part dev
python src/main.py retrieve --data-dir ../../dataset --run-dir artifacts/fresh-baseline --part threshold
python src/main.py features --data-dir ../../dataset --run-dir artifacts/fresh-baseline --part threshold
python src/main.py train --data-dir ../../dataset --run-dir artifacts/fresh-baseline --config configs/default.json
python src/main.py score --data-dir ../../dataset --run-dir artifacts/fresh-baseline --part threshold --config configs/default.json
python src/main.py calibrate --data-dir ../../dataset --run-dir artifacts/fresh-baseline --config configs/default.json
python src/main.py evaluate --data-dir ../../dataset --run-dir artifacts/fresh-baseline --part threshold --config configs/default.json
```

For the locked evaluation check, run `retrieve`, `features`, `score`, and
`evaluate` again with `--part eval`. Use a new run directory after any source or
configuration change. Once the evaluation is acceptable, prepare and index the
test split, then use the same frozen model and threshold through `predict` or the
staged test commands. The one-command `src/main.py run` also reproduces the full
workflow, but it runs training and test prediction together and is less useful
when diagnosing an interrupted stage.

### Minimal smoke check

Before committing to a full run on unfamiliar hardware, verify the installation
and country-transfer behavior with the reduced fixture:

```bash
python src/main.py smoke --data-dir ../../dataset --run-dir artifacts/fresh-smoke
python ../../student_resource/utils/validate_submission.py \
  --matching artifacts/fresh-smoke/output/matching_results.tsv \
  --candidate artifacts/fresh-smoke/output/candidate_pairs.tsv \
  --test-dir artifacts/fresh-smoke/smoke_dataset/test --check-ids
```

Smoke outputs are diagnostic only and must never be uploaded as the competition
submission.

## What has been completed

### Data and validation rules

- Read the supplied `plan.md` and `ps.md` and implemented the required two-file
  submission format.
- Strictly prepare UTF-8 TSV data with schema, ID, country, duplicate, and truth
  checks.
- Preserve every Source 1 reference, including references with no matches.
- Treat country as an open string label. US and India are present in training;
  France is present only in test and is not filtered out.
- No external business lookup, geocoding, API, or internet data was used.

### Retrieval and candidate generation

- Built disk-backed Tantivy indexes for every country, containing all Source 2 and
  Source 3 training targets.
- Search uses normalized, transliterated names and addresses, name grams, and
  numeric components through three channels: name, address, and combined.
- The current fast path searches both target sources together, partitions the
  global top-k results by source, and falls back to source-filtered search if a
  source quota is underfilled. This preserves source coverage while reducing
  Tantivy search overhead.
- The adaptive reducer retains the original rank-fusion candidates and adds
  strong text-similarity alternatives, with at most 32 final candidates per
  target source. Candidates are exactly the pairs scored by the classifier.
- The selected policy averages 46.4 candidates per reference and caps at 64.
- On 10,000 development references, retrieval recall was 97.34% and oracle macro
  F0.5 was 0.991311. No known positive pair was lost when comparing the fast path
  with the previous adaptive path.
- Normal integrated retrieval took 47.9 seconds for 10,000 references, versus
  247.9 seconds previously. The projected full-test retrieval time is about 2.3
  hours on this Mac, subject to actual test-country mix and disk behavior.

### Matcher and features

- Compute 53 numeric pair features from supplied text only: normalized and
  transliterated name/address similarity, weighted token overlap, numeric
  agreement/conflict, script information, missingness, channel ranks, fusion
  evidence, source indicator, and candidate count.
- Train one CatBoost classifier; no PyTorch, pretrained model, embedding service,
  or GPU is required for the current baseline.
- Training uses 100,000 references and 4,660,426 candidate pairs. CatBoost ran on
  CPU with 12 threads, 1,500 depth-6 trees, learning rate 0.05, and seed 2026.
- Development loss continued improving through the configured iteration limit;
  the saved model contains 1,500 trees.
- The model hash is:

  `cc2139d0dfc42292dcf15f62ee6a2272b301b3df93ecf2f488703a9dcec14e37`

### Frozen validation result

The threshold was selected on a separate 20,000-reference calibration partition,
then evaluated unchanged on another 20,000-reference partition:

| Metric | Calibration | Held-out evaluation |
|---|---:|---:|
| Macro F0.5 | 0.944408 | 0.943936 |
| US macro F0.5 | 0.951859 | 0.950624 |
| India macro F0.5 | 0.933183 | 0.933836 |
| Pair precision | 98.268% | 98.253% |
| Pair recall | 88.398% | 88.453% |
| Singleton false-positive rate | 8.304% | 9.610% |

The frozen threshold is **0.685**. It must not be retuned using test data.

The main remaining modeling weakness is classification of difficult close
negatives, especially singleton references and exactly-one-match references. On
held-out data, retrieval missed 1,705 true pairs, while the classifier rejected
6,311 retrieved true pairs. This baseline is suitable for a first submission but
has room for later hard-negative and threshold work.

## Important artifacts

- `artifacts/local-fast-pilot/model/model.cbm` — frozen CatBoost model.
- `artifacts/local-fast-pilot/model/schema.json` — feature schema.
- `artifacts/local-fast-pilot/training_config.json` — exact production config.
- `artifacts/local-fast-pilot/decision/threshold.json` — frozen threshold and
  model identity.
- `artifacts/local-fast-pilot/reports/matching_eval.json` — held-out metrics.
- `artifacts/local-fast-pilot/reports/diagnostics_eval.json` — slice metrics,
  error counts, feature importance, and false-positive examples.
- `artifacts/local-fast-pilot/reports/test_pipeline_runtime.json` — live/final
  test-run status and stage timings.
- `HANDOFF.md` — detailed model training, validation, speed, and submission guide.
- `README.md` — concise pipeline architecture and ordinary staged commands.

The train preparation and indexes in this run are verified links to the immutable
full-corpus artifacts in `artifacts/local-full-pilot/`. Keep both directories
together. The linked training corpus must not be rebuilt with the combined `run`
command.

## What remains before submitting

The active driver performs these stages in order:

1. Prepare the complete test Source 1/2/3 corpus.
2. Build country-partitioned test indexes.
3. Retrieve candidates for every test Source 1 reference.
4. Compute features for every candidate pair.
5. Score candidates with the frozen model and threshold.
6. Export `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
7. Run the internal provenance/ID/subset validator.
8. Run the official validator with `--check-ids` against `dataset/test`.

The command used for the complete frozen run, after activating the environment, is:

```bash
cd code/business_entity_resolution
python experiments/run_submission.py \
  --run-dir artifacts/local-fast-pilot \
  --data-dir ../../dataset \
  --validator ../../student_resource/utils/validate_submission.py
```

If the process stops, rerun the same command. Inspect:

```bash
cat artifacts/local-fast-pilot/reports/test_pipeline_runtime.json
tail -n 50 artifacts/local-fast-pilot/test_*.log
df -h .
```

Do not upload anything until the driver reports:

`PIPELINE COMPLETE: internal and official validators passed`

and `test_pipeline_runtime.json` reports `status: PASS`.

After the run passes, inspect the output counts and sizes:

```bash
wc -l artifacts/local-fast-pilot/output/*.tsv
du -sh artifacts/local-fast-pilot/output
head -n 3 artifacts/local-fast-pilot/output/matching_results.tsv
head -n 3 artifacts/local-fast-pilot/output/candidate_pairs.tsv
```

The matching file must have one row for every test Source 1 entity. Empty
`matched_entity_ids` fields are required for predicted singletons. Every matched
ID must be an existing Source 2 or Source 3 test ID and must also occur in the
candidate file. Candidate and match lists must have no duplicates.

## Submission package checklist

The challenge requires a single zip containing:

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

Before zipping:

- Copy the two validated TSVs into a clean package `output/` directory.
- Copy the runnable `code/business_entity_resolution/` project, including
  `src/`, `README.md`, `requirements.txt`, `requirements-lock.txt`, and the MIT
  license. Do not include the 14 GiB experimental `artifacts/` directory unless
  the challenge explicitly requests it.
- Copy `student_resource/Documentation_template.md` from the challenge bundle to
  the package root and fill it in with the actual
  methodology, candidate generation, model/features, validation result, and
  reproducibility command. The report files are the source of truth for numbers.
- Preserve CatBoost's Apache-2.0 license and the project MIT license. Keep the
  dependency/license list in the methodology package.
- Re-run the official validator on the clean package paths, not only on the run
  directory.
- Keep a copy of the exact model hash, threshold, config, source snapshot, and
  validator output alongside the package for auditability.

The challenge allows at most five submissions per day. Freeze the model and
submission files before the first upload, record the portal result, and reserve
time for a validator or packaging correction. Do not change the threshold or
candidate policy after seeing leaderboard feedback without creating a new,
explicitly versioned run.

## Possible later improvements

The first valid submission should use the frozen baseline above. If time remains
after obtaining a valid result, investigate improvements in this order:

1. Mine hard negatives and retrieved-but-rejected positives on development data;
   singleton precision is the clearest current weakness.
2. Tune threshold and candidate rules using only training partitions, then rerun
   the locked evaluation partition.
3. Add targeted multilingual/name-variant features and test them on held-out
   India slices.
4. Consider NVIDIA GPU CatBoost or a neural/PyTorch matcher only after measuring
   a concrete model change. GPUs are unnecessary for completing this baseline.

No external data augmentation is permitted. Any later model must preserve the
same strict train/calibration/evaluation separation and must produce the exact
candidate file used for inference.
