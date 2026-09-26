# Business entity resolution baseline

A runnable first baseline for Amazon ML Challenge 2026. It uses only supplied
records and labels. No PyTorch, pretrained models, external business lookup, or
online service is needed. CPU execution works on macOS; CatBoost training can
switch to NVIDIA CUDA on Linux. Apple MPS is not used.

`HANDOFF.md` is the operational guide: it explains fresh-device setup, the
frozen baseline, measured validation, the resumable test run, and final packaging.
The current baseline reached 0.943936 macro F0.5 on 20,000 held-out references,
with a frozen 0.685 threshold. Historical performance notes were consolidated
there so this package has one current handoff instead of several overlapping
reports.
`configs/default.json` uses that blocker; `configs/baseline.json` preserves the
original fixed 24-candidate configuration.

The `code/business_entity_resolution/` directory is intentionally the package
boundary required by the challenge submission format. It contains all runnable
source, configuration, dependency pins, tests, experiments, and licenses. The
repository-level `dataset/`, `student_resource/`, and generated `artifacts/`
directories are supplied separately and are ignored because they are large or
challenge-specific.

## Environment

Tested with Python 3.12. Use the same version on the cluster.

```bash
cd code/business_entity_resolution
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest tests -q
```

`requirements.txt` is the small runtime install for production execution.
`requirements-lock.txt` is the complete tested environment: it includes runtime
dependencies, CatBoost's transitive packages, and pytest for verification. Use
the lock file when reproducing this run; use the smaller file when you only need
to execute an existing model. CatBoost GPU mode additionally requires a working
NVIDIA driver. Neither CUDA training nor cluster throughput has been tested by
the Mac smoke run.

## Short Mac execution test

From `code/business_entity_resolution`:

```bash
python src/main.py smoke --data-dir ../../dataset --run-dir artifacts/mac-smoke-v1
```

This scans the original training files to construct a separate fixture:

- 2,000 training references, split into train/dev/threshold/eval groups.
- Every labeled target for those references, plus 3,000 extra targets per source.
- 60 test references: 20 each from US, India, and France.
- A reduced test target corpus of 3,000 records per target source.

It then prepares data, builds indexes, retrieves candidates, computes 53 features,
trains CatBoost on CPU, calibrates a threshold, predicts, exports both TSVs, and
validates them. A separate evaluation split exercises the locked-evaluation path.

**Smoke metrics are not estimates of competition accuracy.** Its target corpus
is intentionally much easier, and its outputs are not valid submissions for the
full competition test set. `SMOKE_ONLY.json` marks these runs. The original
dataset is never modified.

Run the supplied official validator against the smoke fixture, not the full data:

```bash
python ../../student_resource/utils/validate_submission.py \
  --matching artifacts/mac-smoke-v1/output/matching_results.tsv \
  --candidate artifacts/mac-smoke-v1/output/candidate_pairs.tsv \
  --test-dir artifacts/mac-smoke-v1/smoke_dataset/test --check-ids
```

## First full cluster run

Copy this project and the full `dataset/` directory to the GPU host. Use a new
run directory, not the Mac smoke artifacts. `--data-dir` must contain `train/`
and `test/`. CLI paths work from any directory.

Before a full run, inspect hardware and benchmark retrieval against the full
training target corpus. In the commands below, replace `/data/dataset` and
`/scratch/ber-v1` with actual paths on your cluster.

```bash
python src/main.py prepare --data-dir /data/dataset --run-dir /scratch/ber-v1 --split train
python src/main.py index --data-dir /data/dataset --run-dir /scratch/ber-v1 --split train
python src/main.py retrieve --data-dir /data/dataset --run-dir /scratch/ber-v1 --part dev
python src/main.py features --data-dir /data/dataset --run-dir /scratch/ber-v1 --part dev
```

Inspect `reports/blocking_dev.json`, stage timings, disk usage, and process memory.
The default development sample is 10,000 references; retrieval still searches all
training S2/S3 records. Do not treat the current candidate limit as tuned. If the
pilot looks feasible, this command resumes the completed stages and continues:

```bash
python src/main.py run --data-dir /data/dataset --run-dir /scratch/ber-v1 --device GPU --workers 12
```

For CPU training, omit `--device GPU`. To select an individual CUDA device, add
`"devices": "0"` under `model` in a copied JSON config and pass `--config` on every
command. `workers` controls retrieval/feature processes; `model.thread_count`
controls CatBoost CPU threads. Defaults use 12 workers and 256-reference shards,
tested on the 14-core, 48 GB M4 Pro. Use fewer workers on smaller machines;
workers have separate SQLite caches, term-frequency caches, and index readers.

The default configuration trains on up to 100,000 references, uses up to 10,000
for early stopping and 20,000 for threshold selection, and **predicts every test
reference**. Training query sampling never reduces the target retrieval corpus.
There is no fixed 128 GB RAM or 200 GB scratch requirement. Measure before scaling.

## Pipeline and validation

1. `prepare`: strict UTF-8 TSV/schema/ID checks; disk-backed SQLite records;
   complete ground-truth reference coverage, target existence and country checks.
   Original names/addresses and multiple normalized views are retained.
2. `index`: a disk-backed Tantivy index per country and dataset split, with
   name words, name trigrams, address words, and numeric components. Query tokens
   use document frequency, with no English stemming. Unknown country strings work.
3. `retrieve`: three shared search channels, partitioned into target sources,
   with a source-filtered fallback whenever a source's raw quota is not filled.
   Rank fusion reserves per-channel slots. Defaults: 32 raw hits/channel/source.
   Adaptive selection retains
   the original top 12 per source and adds text-similar alternatives up to 32 per
   source. It averaged 46.4 total candidates on the full-corpus development pilot.
   Source and country are hard filters; address numbers are not.
4. `features`: bounded batches of numerical similarities, rare-token overlap,
   missingness, script, numeric agreement, source and retrieval evidence. IDs and
   country one-hot features are excluded; empty-empty comparisons earn no credit.
5. `train`: one CatBoost classifier, real retrieved negatives, all retained
   positives, and per-pair weight `1 / candidate_count`. A reference-signature
   hash makes 80/5/5/10 partitions and keeps exact normalized duplicates together.
6. `score`, `calibrate`, `evaluate`: CPU model scoring, a threshold sweep at 0.001
   resolution plus all-empty sentinel, and exact singleton-aware macro F0.5.
   Selection uses the configured US/India test-mixture proxy; reports also expose
   unweighted macro F0.5. France has no labeled validation score.
7. `predict`, `export`, `validate`: full-test batching, exact scored candidates,
   thresholded matches, strict ID/subset/provenance checks. Both TSVs are under
   `output/`, with one row for every reference, including zero-candidate queries.

References are selected by deterministic ID hashes within their assigned split;
this approximates natural proportions, rather than enforcing exact country and
match-count quotas. IDs are not classifier features. Each query is evaluated
against its complete truth set, so blocking misses remain false negatives.

`run` deliberately does not consume the locked evaluation partition. After the
configuration is chosen, run the following once for the final check:

```bash
python src/main.py retrieve --data-dir /data/dataset --run-dir /scratch/ber-v1 --part eval
python src/main.py features --data-dir /data/dataset --run-dir /scratch/ber-v1 --part eval
python src/main.py score --data-dir /data/dataset --run-dir /scratch/ber-v1 --part eval
python src/main.py evaluate --data-dir /data/dataset --run-dir /scratch/ber-v1 --part eval
```

These commands reuse the saved threshold; evaluation never retunes it.

## Artifacts and restart behavior

```
run/
  prepared/{train,test}/records.sqlite
  indexes/{train,test}/<country-hash>/
  candidates/<partition>/<shard>/queries.jsonl
  features/<partition>/<shard>/features.npz
  model/{model.cbm,schema.json,config.json,manifest.json}
  decision/{threshold.json,curve.json,manifest.json}
  scores/<partition>/<shard>/scores.jsonl
  output/{matching_results.tsv,candidate_pairs.tsv}
  reports/
  invocations/
```

Artifacts carry source/configuration/dependency fingerprints and file checksums.
Shard files are written through temporary paths; a completion manifest is written
last. Repeat a command with the same inputs to reuse intact shards. Incompatible
or corrupted completed artifacts fail explicitly. Incomplete prepare/index work
restarts at the split/country level; retrieval, features, and scoring restart at
the shard level. A worker error aborts the stage instead of emitting empty matches.

Use a new run directory after source changes or changes to stage-relevant settings.
Keep the exact source version, environment, model and threshold for each submission.
GPU retraining need not reproduce identical model bytes. Exporting from the same
frozen scores does reproduce identical TSV bytes.

SQLite and NumPy shards replace the planned Arrow/Parquet intermediates to keep
the first implementation small. Strings live in the record database; feature
shards contain arrays. Training assembles memory-mapped arrays, but CatBoost pools
still consume RAM; GPU mode is not unlimited-memory training. This baseline retains
test feature shards and score provenance on disk. Disk usage needs a full-corpus
pilot before final sizing. Checksum verification also adds sequential disk reads.

## Remaining competition work

- Full-corpus retrieval recall and candidate-limit selection; current limits are defaults.
- Cluster CPU/RAM/disk/GPU throughput measurements and a full test run.
- Country-transfer diagnostics, deeper error slices, and evidence-based improvements.
- Optional neural matcher/retriever after measuring the baseline's limitations.
- Final methodology template, model/dependency license inventory, submission ZIP,
  and actual portal upload. No submission or packaging is performed automatically.

The original project code is MIT licensed; CatBoost is Apache-2.0. This baseline
has no pretrained weights. Preserve third-party notices in the final package.
