# Project understanding

Updated 27 September 2026. Code reviewed at commit `f9f6290`.

## What the project does

This is the Amazon ML Challenge business entity resolution task. For every
Source 1 business, find all matching businesses in Sources 2 and 3 using the
supplied name, address and country. A reference can have zero, one or several
matches. Training covers US and India; test also contains France.

The deliverables are `matching_results.tsv` and `candidate_pairs.tsv`, each with
one row for every test reference, including empty results. The candidate file
must contain exactly the pairs presented to the classifier, and every accepted
match must belong to that candidate set. The metric is macro F0.5, calculated
per reference, with special treatment of true singletons. Precision therefore
matters more than recall. External business data and lookup services are forbidden.

The existing data audit in `plan.md` reports:

| Split | Source 1 | Source 2 | Source 3 |
|---|---:|---:|---:|
| Train | 2,206,821 | 5,034,616 | 5,285,603 |
| Test | 1,732,544 | 4,887,273 | 5,082,316 |

There are 24,229,173 records overall, including 259,452 French test references.
Training uses a sample of reference queries but searches the complete target
corpus. Reducing the target corpus makes a test easier and is unsuitable for
estimating competition accuracy or realistic retrieval throughput.

## What changed in Git

There are three commits in the history available in this checkout:

| Commit | Author | Time (IST) | What it introduced |
|---|---|---|---|
| `2b70f5d` | Fane1824 / Ishaan | 26 Sep, 20:01 | Initial repository README. |
| `d232e25` | Fane1824 / Ishaan | 26 Sep, 20:02 | Problem statement, rules, video transcript, detailed implementation/data-audit plan and initial ignore rules. |
| `f9f6290` | Amol Vijayachandran | 27 Sep, 01:20 | The complete runnable package: 34 changed files, 2,875 added lines, implementation, tests, configurations, experiments, dependency pins, license and handoff. |

The teammate's commit is much more than its “Clean project structure” title
suggests: it introduces the entire implementation into this Git history.
Intermediate experiments and training runs are described in the handoff, rather
than represented as separate commits here.

## How the implementation works

1. **Prepare:** validate TSV schemas, IDs and ground truth; normalize the text;
   store original and normalized records in SQLite. Keep native and transliterated
   text views. Source 1 signatures determine train/dev/calibration/evaluation
   partitions in an 80/5/5/10 proportion, keeping exact normalized duplicates together.
2. **Index:** create a Tantivy lexical index per country and split over all S2/S3
   targets. Index name words, name trigrams, address words and numbers.
3. **Retrieve:** query name, address and combined channels. The fast implementation
   searches both sources together, then fills any underrepresented source using
   a source-filtered fallback. Adaptive selection preserves the original top 12
   per source and adds similar alternatives, up to 32 per source / 64 per reference.
4. **Features:** compute 53 numeric pair features: text similarities, rare-token
   overlap, numeric agreement, missingness, script information and retrieval
   evidence. IDs and country one-hot encoding are not classifier inputs.
5. **Train:** fit one CatBoost model using retrieved positives and plausible
   retrieved negatives, weighting each pair by the inverse candidate count.
   Default samples: 100,000 train, 10,000 dev, 20,000 calibration, 20,000 evaluation.
6. **Calibrate/evaluate:** select a global threshold on the calibration split using
   the US/India mixture proxy; report unweighted macro F0.5 too. Evaluate the frozen
   threshold separately on the held-out split. No labeled French accuracy is available.
7. **Predict/export/validate:** run all test references through the same candidate
   and feature stages, score on CPU, export the two TSVs and check coverage,
   source IDs, candidate provenance and match-subset constraints.

`src/main.py run` executes training, calibration and full test prediction, but
intentionally leaves locked evaluation to explicit commands. The official
validator is an additional step; `experiments/run_submission.py` orchestrates it
for a previously frozen run with the expected artifacts.

SQLite and NumPy files replace the Arrow/Parquet intermediates proposed in the
original plan. There is no neural embedding model, PyTorch dependency or external
API in the current implementation. GPUs only apply to CatBoost training;
preparation, retrieval, feature calculation and current scoring use CPUs.

## What the teammate reports having completed

These are historical results from `code/business_entity_resolution/HANDOFF.md`,
not measurements independently reproduced on this machine:

- Full-target development retrieval: 10,000 references, 97.34% positive-pair
  recall, oracle macro F0.5 of 0.991311, average 46.4 candidates per reference.
- Shared search reduced measured development retrieval from 247.9 to 47.9 seconds
  on the teammate's Mac, without losing known positives in that comparison.
- CPU training used 100,000 references / 4,660,426 pairs, 12 threads and 1,500
  depth-6 trees at learning rate 0.05. The model reached the iteration limit.
- A global threshold of **0.685** was frozen on a separate 20,000-reference set.

| Metric | Calibration | Held-out evaluation |
|---|---:|---:|
| Macro F0.5 | 0.944408 | 0.943936 |
| US macro F0.5 | 0.951859 | 0.950624 |
| India macro F0.5 | 0.933183 | 0.933836 |
| Pair precision | 98.268% | 98.253% |
| Pair recall | 88.398% | 88.453% |
| Singleton false-positive rate | 8.304% | 9.610% |

The handoff identifies hard negatives and singleton mistakes as important
remaining weaknesses. On its held-out set, retrieval missed 1,705 true pairs;
the classifier rejected a further 6,311 retrieved true pairs.

The reported frozen model SHA-256 is
`cc2139d0dfc42292dcf15f62ee6a2272b301b3df93ecf2f488703a9dcec14e37`.
Its model, threshold, reports and exact training configuration live in the
teammate's ignored `artifacts/local-fast-pilot/` directory, with preparation/index
links to `artifacts/local-full-pilot/`.

## What actually exists after the pull

- Source, tests, configs and documentation are present, and the initial working
  tree was clean at `f9f6290`.
- **The trained model, threshold and generated indexes/features are not in this
  checkout.** The artifacts directory contains only `.gitkeep`. Pulling Git does
  not transfer ignored model artifacts.
- Local raw data is under **`student_resource/dataset/`**, not `dataset/`.
  The official validator is `student_resource/utils/validate_submission.py`.
- The handoff's statement that a full test run is “currently” preparing data
  describes the teammate's machine at handoff time. It is not evidence of an
  active run here or on gnode.
- `configs/default.json` sets CatBoost to **4 CPU threads**, whereas the historical
  training report says 12. Exact reproduction requires the saved run config.
- Existing `experiments/speed_benchmark.py` needs a completed 10,000-reference dev
  candidate run and full prepared/indexed training corpus; it cannot run directly
  from a fresh checkout.

To preserve the exact validated baseline, transfer the teammate's frozen run and
its linked corpus. Otherwise retrain a fresh model and recalibrate its threshold;
do not reuse 0.685 blindly with a different model, especially after GPU training.

## Where to look in the code

| File under `code/business_entity_resolution/` | Purpose |
|---|---|
| `src/main.py` | CLI stages and complete workflow |
| `src/data.py`, `src/text.py` | Input validation, SQLite preparation, normalization and splits |
| `src/blocking.py`, `src/selection.py` | Indexing, search and adaptive candidate reduction |
| `src/features.py`, `src/pipeline.py` | Feature computation and resumable parallel shards |
| `src/learn.py`, `src/metrics.py` | CatBoost, scoring, calibration and evaluation |
| `src/export.py` | TSV export and internal validation |
| `src/common.py` | Hashes, manifests, dependency/source fingerprints |
| `src/smoke.py` | Small real-data fixture, explicitly unsuitable for submission |
| `experiments/run_submission.py` | Frozen test-run supervisor and official validation |
| `experiments/compare_*.py` | Candidate-reduction comparisons |
| `experiments/matching_diagnostics.py` | Model errors and feature diagnostics |
| `tests/` | Metrics, retrieval, selection, complete pipeline, resume and validation tests |

Completed artifacts are immutable and checksum-verified. Incomplete query work
resumes at shard boundaries; incomplete preparation restarts a split and incomplete
indexing restarts a country. Keep separate run directories when source or relevant
configuration changes. Do not modify `src/` during an active frozen run.

## Remote test and runtime estimate

Server access was verified using the existing `~/.ssh/config` alias `gnode`,
through `ada`, as `ishaan.romil` on `gnode074`.

Preflight at about 03:26 IST on 27 September:

- 38 logical CPUs available to the process (affinity: 0–18 and 20–38).
  The host has two Intel Xeon E5-2640 v4 processors: 20 physical cores / 40
  logical threads in total, with 38 threads exposed to this session.
- 125 GiB physical RAM, approximately 123 GiB available.
- Three idle NVIDIA RTX 2080 Ti GPUs, 11 GiB VRAM each.
- Approximately 149 GiB free on local `/scratch` (same filesystem as `/tmp`).
- Home storage is network-mounted; local scratch is preferable for SQLite/index I/O.
- System Python is 3.10.12; the project requests an isolated Python 3.12 environment.

Isolated working directory:
`/scratch/ish/amazon_ml_20260927/`.
No existing project job or frozen model was found in the checked remote locations.
The project was initially staged under `/scratch/ishaan.romil`, then moved to
`/scratch/ish` at the user's request. The environment, Python runtime, package
cache and temporary files are now configured under that task directory. Existing
home files are to remain untouched; no home-directory cleanup is authorized.

### Tests completed on gnode

The user selected **rebuild from the pulled code**, rather than transferring the
teammate's trained artifacts. Python 3.12.12 and all 24 dependencies from
`requirements-lock.txt` are installed under the scratch workspace.

| Check | Result |
|---|---|
| Original test suite | 29 passed in 10.61 seconds |
| Original CPU end-to-end smoke | Passed in 17.84 seconds |
| Original three-GPU training smoke | Passed in 4.13 seconds |
| Test suite after the preparation change below | **30 passed in 8.42 seconds** |
| Revised CPU end-to-end smoke | **Passed in 16.19 seconds** |
| Revised three-GPU training smoke | **Passed in 2.73 seconds** |
| Official validator with `--check-ids` | Passed on all 60 fixture references |

The fixture contains 2,000 training references, all their labeled targets plus
3,000 distractors per target source, and 60 test references (20 each from US,
India and France). These results validate execution, not competition accuracy.
The validator's generic “safe to submit” message applies only to the fixture;
these reduced outputs must never be submitted for the full competition.

Local copies of measured reports are kept under the ignored
`code/business_entity_resolution/artifacts/gnode-results/` directory. Server
reports live in `/scratch/ish/amazon_ml_20260927/runs/`.

### Preparation change made during this session

Hardware inspection showed that `/scratch/ish` is on a **rotational Toshiba
hard disk**, not an SSD. Initial full-test preparation progressed from 100,000
to 400,000 records while its per-100,000 interval grew from roughly 27 to 46
seconds. That initial process was stopped before changing the source.

`src/data.py` now gives the temporary preparation writer a 512 MiB SQLite cache,
keeps its rollback journal in memory, and avoids per-batch disk synchronization.
It closes and explicitly synchronizes the completed database before renaming it
and recording the completion manifest. An interrupted temporary build is discarded
and rebuilt. Read-only worker caches remain at 64 MiB each. No normalization,
candidate selection, model features, labels or decision rules were changed.

On the same 220,000-record preparation fixture (20,000 references and 100,000
records from each target source), measured wall time changed from **38.97 to
24.37 seconds**, a **1.60× speedup**. Parent peak RAM rose from 207 to 319 MiB.
This is one before/after measurement; disk-cache and concurrent-transfer effects
were not controlled. The new recovery test also checks SQLite integrity after
replacing an interrupted temporary database.

### Initial runtime estimate

**Planning estimate: roughly 8–14 hours for a fresh rebuild through validated
full-test outputs, including contingency. This is provisional, not a measured
full-corpus completion time.** Installation and data transfer have already finished.

| Work | Initial allowance before contingency | Basis / limitation |
|---|---:|---|
| Train + test preparation | 0.75–1.5 h | 24.37 s / 220,000 records projects about 45 minutes; larger database indexes and truth checks add uncertainty. |
| Train + test indexing | 0.5–1 h | Small-fixture indexing extrapolation; large-index merging remains unmeasured. |
| Retrieval, including training partitions and full test | 2–4 h | Teammate's 2.3-hour Mac test projection is only an anchor; gnode full-corpus rate is still unmeasured. |
| Features | 0.5–1.5 h | Real-data smoke throughput, adjusted conservatively for roughly 80 million test pairs and disk I/O. |
| GPU training, calibration, scoring, export and both validators | 1–1.5 h | Broad allowance; the GPU smoke verifies compatibility, not large-pool scaling. |

These component allowances total approximately 4.75–9.5 hours; adding 50%
contingency gives about 7–14 hours, rounded to an 8–14-hour planning window.
No completion time or score is guaranteed by the smoke run. The primary
uncertainties are full-corpus retrieval on the older Xeons, hard-disk I/O,
large-pool GPU training and the French test distribution.

`experiments/remote_pilot.py` is the next timing test. It prepares and indexes
**all training targets**, then times retrieval/features on 10,000 train and
10,000 development references, trains a small-sample GPU model with up to 1,500
trees, and times scoring. Its `runtime_estimate.json` replaces the rough component
assumptions with measured full-corpus rates and a stated 50% allowance. It remains
an extrapolation, especially for France and larger GPU pools.

`experiments/remote_rebuild.py` continues only after that pilot passes. It reuses
verified train preparation/indexes and the identical development sample, then
retrieves 100,000 training references, fits a fresh full-size model, calibrates a
new threshold, checks locked evaluation and runs the frozen test-submission driver.
It never promotes the pilot model. Both drivers record commands, timing and failures.

**Launch order requested by the user:** commit and push these code/documentation
updates first, then start the full-corpus pilot and the dependent rebuild. At the
time this section was written, smoke tests were complete and the new full-corpus
pilot had not yet started. No home-directory files were deleted.

## Remaining work

1. Push the verified changes, then launch the full-corpus timing pilot.
2. Refine the provisional runtime estimate from its measured stage timings.
3. Train, calibrate and evaluate the fresh model; the user chose rebuilding.
4. Produce and validate both full-test output files, including every French reference.
5. Complete the methodology template, preserve model/configuration/provenance and
   licenses, and build the submission package.
6. Upload through the challenge portal and record the result. No upload has been
   performed as part of this task.

The repository plan records a deadline of 27 September 2026, 23:59 IST, and a
target upload time of 22:00 IST. These are repository notes, not a fresh portal check.
