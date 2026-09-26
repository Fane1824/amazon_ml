# Amazon ML Challenge 2026: detailed implementation plan

The following specifies the contents of `plan.md` and the separate `reject_ideas.md`. Neither file has been written because this task is currently in Plan mode.

## Contents of `plan.md`

### 1. Objective and chosen approach

Build a reproducible business entity resolution pipeline that:

1. Generates a small, auditable candidate set for every test Source 1 entity.
2. Scores those candidates using one supervised matching model.
3. Optimizes the challenge’s exact **macro F0.5**, including singleton behavior.
4. Handles US, India, France, and unfamiliar country labels without dropping records.
5. Produces valid leaderboard predictions and the complete final submission package.

**Chosen architecture:**

> Conservative text normalization → country-partitioned lexical retrieval → deterministic candidate reduction → numerical similarity features → one CatBoost classifier → validation-selected threshold → TSV exports.

Use an existing machine with substantial CPU, RAM, and storage. Use one L40S for classifier training if available on that machine. Keep AWS credits as a fallback for data processing.

Do not make embeddings, language models, distributed training, or a large ensemble prerequisites. The main challenges are retrieval recall, distinguishing common business names, and transferring to France.

The plan deliberately separates **measured findings** from **implementation targets**. No matching model has been trained during this review, so model scores and runtime estimates remain unverified.

### 2. Findings from the supplied materials and data

Reviewed:

- `guidelines.md`
- `ps.md`
- `video_transcript.md`
- Resource README and documentation template
- Complete submission validator
- Archive inventory
- All seven TSV files through streaming audits
- A deterministic sample of 4,517 training reference entities and all 15,578 of their labeled matches

#### Dataset size

| Split | Source 1 | Source 2 | Source 3 | Total records |
|---|---:|---:|---:|---:|
| Train | 2,206,821 | 5,034,616 | 5,285,603 | 12,527,040 |
| Test | 1,732,544 | 4,887,273 | 5,082,316 | 11,702,133 |
| Total | 3,939,365 | 9,921,889 | 10,367,919 | 24,229,173 |

An unrestricted test Cartesian product would require approximately **17.27 trillion comparisons**.

The workspace contains the starter validator, but no existing training or inference implementation.

#### Ground-truth integrity

The audit established:

- Exactly one ground-truth row per training Source 1 ID.
- No duplicate source IDs within any input file.
- No train/test ID overlap within a source.
- All referenced ground-truth target IDs exist.
- No duplicate IDs within ground-truth match lists.
- No target is assigned to multiple training reference entities.
- No cross-country positive pairs.
- **7,638,365 positive pairs**.
- **2,681,854 training target records** have no reference match.
- **123,247 singleton references**, representing **5.5848%** of training Source 1.
- Mean matches per reference: **3.4613**.
- Observed total match counts range from zero to eleven.

The unlinked target records are important distractors. They must remain in validation retrieval pools.

#### Country shift

| Country | Training Source 1 | Test Source 1 | Test share |
|---|---:|---:|---:|
| US | 1,323,633 | 663,106 | 38.27% |
| India | 883,188 | 809,986 | 46.75% |
| France | 0 | 259,452 | 14.98% |

Training is approximately 60% US and 40% India. Test has a different mixture and a substantial unseen-country component.

#### Text and ambiguity findings

- Source 1 names and addresses are populated throughout the supplied data.
- Approximately 3.34% of training targets and 2.66% of test targets have empty addresses.
- Indian target names use multiple scripts, including Devanagari, Kannada, Bengali, Tamil, Gujarati, Malayalam, Telugu, Odia, and Gurmukhi.
- True matches include abbreviations, initials, unrelated-looking trade names, appended websites, phone-like strings, legal-suffix changes, and reordered addresses.
- Some true matches contain conflicting or additional address numbers. Number disagreement cannot be an unconditional rejection.
- About **35.8% of US** and **44.4% of Indian** training references share a normalized name with another reference.
- Two French test references have the same normalized name and address. Preserve both IDs.

These findings rule out exact-name matching as a sufficient decision rule and make address-independent and name-independent retrieval both necessary.

### 3. Competition rules and operational constraints

Use the supplied problem statement for task-specific behavior. The AWS tutorial provides infrastructure guidance, not a replacement scoring or output specification.

#### Required outputs

`matching_results.tsv`:

```text
source1_entity_id<TAB>matched_entity_ids
```

`candidate_pairs.tsv`:

```text
source1_entity_id<TAB>candidate_entity_ids
```

Both must contain exactly **1,732,544 data rows**, one per test Source 1 ID.

Requirements:

- UTF-8, tab-separated, exact headers.
- Empty second field for no matches or no candidates.
- Comma-separated target IDs without spaces or duplicate IDs.
- Only existing test S2/S3 IDs.
- No Source 1 IDs in target lists.
- Every final match must belong to its exported candidate list.

**Candidate provenance is a hard invariant:** export the exact pairs presented to the matching classifier. Do not shrink the candidate file after classifier scoring.

#### Fair play and licensing

- Use only the provided records and labels for learning business relationships.
- Do not query business directories, geocoders, search engines, entity resolution APIs, or external business datasets.
- Documentation lookup and installing ordinary software libraries do not supply business labels.
- Preserve dependency licenses and document the final model’s license.
- Use CatBoost, which is Apache-2.0 licensed; distribute the trained model and original project code under a compatible permitted license. No pretrained model is required. [CatBoost repository and license](https://github.com/catboost/catboost)

#### Submission and documentation constraints

- Maximum five submissions per day.
- No submissions have been made yet.
- Do not assume unused earlier-day slots carry forward.
- Deadline: **September 27, 2026, 11:59 PM IST**.
- Preserve every submitted file, its corresponding candidates, model, configuration, and hashes.

The general guidelines request a 1–2-page explanation, while the detailed problem statement allows a longer methodology document. Satisfy both conservatively: a concise filled template, with full technical details in the project README.

### 4. Execution environment and AWS fallback

#### Primary machine

Use one existing GPU host for the entire pipeline.

Resource targets:

- Linux and Python 3.11.
- At least 16 CPU threads.
- Prefer 128 GB system RAM; 64 GB is acceptable if the pilot confirms sufficient headroom.
- At least 200 GB free scratch storage.
- One L40S for training.
- CPU-based normalization, indexing, feature extraction, and final inference.

Inspect actual hardware before starting. GPU memory is not a substitute for system RAM.

The local Mac has approximately 39 GiB free disk and lacks most required modeling libraries. Use it for development and small checks, not full intermediate storage.

#### Dependencies

Use:

- NumPy
- Polars
- PyArrow
- Tantivy Python bindings
- RapidFuzz
- AnyAscii
- CatBoost
- pytest

Resolve stable Python 3.11-compatible packages once, run the smoke test, and pin the complete resolved environment. Do not upgrade packages during the challenge.

Tantivy provides a local full-text index without operating a search service. RapidFuzz supplies compiled string comparisons. [Tantivy](https://github.com/quickwit-oss/tantivy-py), [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz)

#### AWS fallback

Use AWS only if the existing host fails the resource or throughput checks.

Follow the supplied guide’s notebook-and-batch-prediction approach. Do not deploy an inference endpoint. The tutorial’s small notebook instance is not an appropriate sizing reference for this dataset. [AWS preparation guide](https://builder.aws.com/content/3HiM6zDmFrF98fRzOUETnGFDoqz/amazon-ml-challenge-2026-your-complete-prep-guide-with-live-demo)

Fallback specification:

- One SageMaker Notebook Instance.
- Preferred size: `ml.r5.4xlarge`, subject to account availability and current pricing.
- 200 GB attached storage.
- Prefer `us-east-1` as the guide suggests, unless access or quotas make another region necessary.
- Run the same ordinary Python pipeline inside the notebook’s terminal.
- Train on CPU if moving features to the existing GPU host would take longer.
- No Studio migration, managed endpoint, feature store, or orchestration service.

Before launch, verify available credit balance, eligible services, quotas, storage charges, and current regional rates. [Notebook creation API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateNotebookInstance.html), [SageMaker pricing](https://aws.amazon.com/sagemaker/ai/pricing/)

Budget policy:

| Allocation | Maximum planned amount |
|---|---:|
| Compute | $150 |
| Storage and transfers | $20 |
| Uncommitted reserve | $30 |

Reduce these limits if the available credit balance is below $200. Calculate an explicit maximum runtime from the quoted hourly rate before launch.

Set budget alerts, but use a runtime limit as well: billing notifications can arrive after additional costs accrue. Stop compute after copying results; account separately for retained storage. [AWS Budgets limitations](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html), [Stopping notebook compute](https://docs.aws.amazon.com/botocore/latest/reference/services/sagemaker/client/stop_notebook_instance.html)

### 5. Code organization, interfaces, and artifacts

Keep this as a small batch-processing project.

```text
code/business_entity_resolution/
├── src/
│   ├── main.py
│   ├── data.py
│   ├── text.py
│   ├── blocking.py
│   ├── features.py
│   ├── learn.py
│   ├── metrics.py
│   └── export.py
├── tests/
├── configs/
│   └── default.json
├── artifacts/
├── README.md
├── requirements.txt
└── LICENSE
```

Expose one command-line entry point:

```text
python src/main.py <stage> --data-dir <directory> --run-dir <directory> --config <file>
```

Stages:

| Stage | Responsibility |
|---|---|
| `audit` | Validate input schema, IDs, labels, counts, and input hashes |
| `prepare` | Normalize records, assign internal row IDs, create split manifests |
| `index` | Build train or test retrieval indexes |
| `retrieve` | Produce candidate-pair shards and blocking statistics |
| `features` | Generate numeric pair features |
| `train` | Fit and save the classifier |
| `evaluate` | Score complete reference match sets and select the threshold |
| `predict` | Retrieve, score, and save test pair predictions in batches |
| `export` | Create both required TSVs |
| `validate` | Run strict internal checks and the supplied validator |
| `package` | Assemble the final reproducible archive |

Each stage records:

- Input and configuration hashes.
- Code version or source snapshot hash.
- Dependency versions.
- Random seeds.
- Start/end time and record counts.
- Output paths and hashes.
- Completion status.

Write outputs to temporary paths and rename them only after successful completion. Resume only shards whose input/configuration hashes match.

#### Internal data contracts

**Record table:** split, source, original entity ID, internal row index, country, original text, normalized text views, missingness and script flags.

**Candidate table:** reference row index, target source, target row index, retrieval-channel flags, channel ranks, fusion rank.

**Feature table:** candidate identifiers plus an ordered, versioned list of float32 features.

**Score table:** candidate identifiers, classifier score, model hash.

**Run metadata:** data hashes, split manifest, feature schema, retrieval policy, model parameters, threshold, metrics, resource usage.

Keep strings in record tables rather than duplicating them in every pair row.

### 6. Data preparation and normalization

Read TSVs with an explicit tab separator. Preserve IDs as strings and empty values as empty strings.

Create sequential internal integer indexes for efficient joins. Never use the numerical portion of an entity ID as a predictive feature.

Preserve original fields and derive separate text views.

| View | Purpose |
|---|---|
| Original text | Inspection, reproducibility, script-sensitive comparison |
| Conservative normalized text | Unicode normalization, case folding, whitespace normalization |
| Latin-folded text | Accent-insensitive comparisons |
| Transliterated text | Additional cross-script retrieval and features |
| Core business name | Legal-suffix and obvious decoration normalization |
| Address tokens and numeric components | Order-insensitive similarity and weak structural evidence |

#### Name handling

- Normalize punctuation and whitespace.
- Standardize `&` and `and` in a secondary view.
- Canonicalize legal suffix variants such as `pvt/private` and `ltd/limited`.
- Preserve a full-name view; do not rely exclusively on suffix-stripped names.
- Extract obvious website/domain components as additional weak evidence.
- Remove clearly appended phone-like or `ID:` decorations only in the secondary view.
- Preserve genuine business-name numbers.
- Compute initials/acronym compatibility as a feature, never an automatic match.
- Treat OCR-style substitutions as a supplementary comparison, not a destructive rewrite.

#### Address handling

- Retain both order-sensitive strings and order-insensitive tokens.
- Canonicalize ordinary address abbreviations conservatively.
- Extract all numeric/alphanumeric components, not just the first number.
- Preserve compound numbers, ranges, unit markers, and tokens such as `bis`.
- Treat an isolated `null` placeholder as missing noise.
- Do not require a postcode, city parser, or state parser.
- Do not use geocoding or external locality dictionaries.
- Never hard-reject solely on a number, postcode, or unit disagreement.

#### Unicode and transliteration

Use AnyAscii as an additional deterministic view. Preserve native Unicode because transliteration is approximate and can remove distinctions. Its character-based output must not be mistaken for a verified alias. [AnyAscii behavior and license](https://github.com/anyascii/anyascii)

Do not strip combining marks indiscriminately from native Indic text. Apply accent folding to the Latin comparison view.

All country labels must pass through a generic mapping. France must follow the same pipeline even though it has no training labels.

### 7. Validation design

Validation is the main protection against wasting submissions.

#### Reference-level partitions

Split by Source 1 entity, never by candidate pair:

- 80% training partition.
- 5% early-stopping/development partition.
- 5% threshold-selection partition.
- 10% locked evaluation partition.

Use a fixed seed and persist the assignments.

Keep a reference and every associated positive pair in one partition. Group exact duplicate normalized name-and-address reference signatures together if encountered.

#### Initial working samples

Use deterministic, proportionally stratified samples:

| Purpose | References |
|---|---:|
| Blocking development | 10,000 from the development partition |
| Classifier training | 400,000 |
| Early stopping | 20,000 |
| Threshold selection | 50,000 |
| Locked final evaluation | 100,000 |

Stratify by country and match-count buckets: zero, one, two-to-four, five-plus. Preserve natural proportions for aggregate scores.

The remaining records remain available as background retrieval records and for a justified training expansion.

#### Full distractor pool

For every training, development, or validation query, retrieve against the **complete training S2/S3 population**.

Do not restrict the target corpus to targets owned by sampled references. Such a shortcut would make retrieval and classification artificially easy.

The search index can contain held-out target text because candidate text is available at inference. Held-out labels must not be used to build features, aliases, thresholds, or training examples.

Use complete ground-truth sets for scoring. A positive pair missed by blocking remains a false negative.

#### Additional robustness checks

1. **US → India transfer:** fit a diagnostic model on US training queries and evaluate Indian held-out queries.
2. **India → US transfer:** perform the reverse.
3. **Name-disjoint slice:** evaluate references whose normalized names do not occur in the classifier’s training sample.
4. **Perturbed-reference slice:** apply fixed identity-preserving typos, word-order changes, or legal-suffix changes to held-out queries.
5. **Missing-address and script slices:** inspect performance separately.

Use 100,000 training queries and 10,000 evaluation queries per country-transfer direction to keep these checks bounded.

These checks provide evidence about transfer; they do not establish French accuracy.

### 8. Exact metric implementation

For a reference with truth set \(T\) and predicted set \(P\):

\[
F_{0.5}(P,T)=
\begin{cases}
1, & |P|=0,\ |T|=0\\
0, & |T|=0,\ |P|>0\\
\frac{1.25\,TP}{1.25\,TP+FP+0.25\,FN}, & |T|>0
\end{cases}
\]

Average over **every evaluated Source 1 entity**.

Do not use pairwise accuracy, micro F0.5, ordinary F1, or macro averaging over binary class labels as the challenge score.

Required sanity cases:

| Truth | Prediction | Expected score |
|---|---|---:|
| Empty | Empty | 1 |
| Empty | One incorrect ID | 0 |
| Two IDs | Empty | 0 |
| Two IDs | Both correct | 1 |
| Two IDs | One correct | 0.833333… |
| Two IDs | Both correct plus one false positive | 0.714285… |

An all-empty prediction over the complete training set must score **0.0558482088…**.

Report:

- Official macro F0.5.
- Per-country macro F0.5.
- Singleton false-positive rate.
- Non-singleton macro F0.5.
- Exact match-set accuracy.
- Micro precision/recall as diagnostics.
- Performance by match count, script, missingness, and name frequency.

Also report a known-country test-mixture proxy using normalized US/India test proportions, approximately 45%/55%. Label this clearly; it excludes unknown French performance.

### 9. Candidate generation

Use one disk-backed Tantivy index per dataset split and country, containing both target sources.

Index fields:

- Source identifier and target row index.
- Native and transliterated name tokens.
- Normalized name character trigrams.
- Native and transliterated address tokens.
- Address numeric tokens.

Use consistent tokenization for indexing and querying, without English stemming. Construct query objects rather than passing raw business strings into a query-language parser. Tantivy supports persistent indexes, Boolean queries, and boosted fields. [Tantivy documentation](https://tantivy-py.readthedocs.io/en/latest/tutorials.html)

#### Retrieval channels

For each reference and each target source, retrieve through three channels:

| Channel | Evidence | Purpose |
|---|---|---|
| Name-led | Name words and name trigrams | Missing or heavily altered addresses |
| Address-led | Address words and numeric components | Transliterated, abbreviated, or unrelated-looking names |
| Combined | Name words, name trigrams, address words | Ordinary noisy records and common-name disambiguation |

Initial relative field boosts:

- Name-led: name words 3, name trigrams 1.
- Address-led: address words 3, numeric tokens 1.
- Combined: name words 2, name trigrams 1, address words 2.

When native and transliterated text are identical, avoid duplicate clauses.

Use up to twelve informative word tokens per field and twelve informative name trigrams per query. Select by corpus document frequency, preserving numeric evidence separately. Very common character grams should not dominate retrieval.

No address component is a mandatory filter. Country and target source are the only hard partitions for the supplied data, supported by the complete positive-pair country audit.

#### Raw retrieval and final candidate reduction

1. Retrieve up to 32 hits from each channel for each target source.
2. Union and deduplicate hits by target ID.
3. Combine channel ranks using reciprocal-rank fusion with constant 60.
4. For a final limit \(L\) per source, reserve up to \(\lfloor L/4\rfloor\) leading hits from each channel.
5. Fill remaining places using fusion rank.
6. Break ties deterministically using original target ID.

This preserves address-only recovery candidates that might otherwise disappear beneath common-name hits.

Evaluate only these initial final limits:

```text
L per source: 8, 12, 20, 32
Total maximum per reference: 16, 24, 40, 64
```

Do not pad candidate lists when fewer records qualify.

#### Blocking measurements

Measure before fitting the classifier:

- Positive-pair recall.
- Mean per-reference recall among non-singletons.
- Fraction of non-singletons with every true match retrieved.
- Fraction with no true match retrieved.
- Mean, median, p95, p99, and maximum candidate count.
- Candidate precision.
- Retrieval runtime and peak memory.
- Channel-specific incremental recall.
- Country, source, script, and missing-address breakdowns.

Compute the **oracle macro ceiling** by predicting \(T_q \cap C_q\) for each reference. This measures the best score achievable using that candidate set.

Selection rule:

- Prefer the smallest candidate limit within **0.001 absolute oracle macro score** of the largest tested limit.
- Aim for at least 99.5% positive-pair recall and 0.995 oracle macro F0.5, but treat these as targets, not established performance.
- Reject a smaller limit if it disproportionately damages a country or script slice.
- If raw recall is inadequate, permit one expansion from 32 to 64 hits per channel and diagnose the newly recovered positives.
- If targets remain unmet, document the measured ceiling and use the best feasible retrieval policy. Do not conceal misses or begin an unbounded search.

At an average of 24 candidates, test inference would involve approximately **41.58 million pairs**. Candidate count is a genuine computational and ranking consideration.

### 10. Pair features and classifier

Compute compact numerical features rather than feeding raw business strings into the classifier.

#### Feature groups

| Group | Features |
|---|---|
| Name equality | Full-name, core-name, transliterated-name equality |
| Name similarity | Edit similarity, Jaro–Winkler, token-sort ratio, token overlap, trigram overlap |
| Distinctiveness | Rare-token overlap, inverse-frequency weighted overlap, normalized name frequency |
| Name structure | Length ratio, token-count ratio, acronym compatibility, numeric-name agreement |
| Address similarity | Token overlap, weighted overlap, edit similarity, token-sort similarity |
| Address structure | Numeric overlap, compound-number agreement, unit evidence, missing components |
| Joint evidence | Minimum/product of name and address similarities; strong-one-field/weak-other-field patterns |
| Missingness | Empty field flags, truncated text indicators, failed transliteration indicators |
| Script | Native-script compatibility and improvement from transliteration |
| Retrieval | Channel membership, reciprocal ranks, fused rank, candidate count |
| Source | Source 2 versus Source 3 |

Important restrictions:

- No entity IDs, row numbers, or file order as model features.
- No raw country one-hot features.
- No hand-labeled French pseudo-labels.
- No blanket exact-name or exact-address acceptance.
- Empty-empty fields must not count as positive evidence.
- Favor symmetric similarity features to reduce dependence on Source 1 being cleaner.

Use float32 feature shards and bounded per-worker caches. Do not store Python token sets for all 24 million records simultaneously.

#### Training examples

For each sampled training reference:

- Generate candidates using the selected inference policy.
- Label a candidate positive only when its ID belongs to that reference’s truth set.
- Retain all candidate negatives, including plausible unlinked targets.
- Keep singleton queries.
- Do not construct a balanced set of random easy negatives.

Use per-pair weight:

\[
w_{q,c} = \frac{1}{|C_q|}
\]

This gives each reference approximately equal total training weight. It is a surrogate; threshold selection still uses the exact challenge metric.

Do not use automatic class balancing.

#### Initial CatBoost configuration

```text
loss_function       = Logloss
eval_metric         = Logloss
iterations          = 1500
depth               = 6
learning_rate       = 0.05
l2_leaf_reg         = 5
border_count        = 128
early_stopping      = 100 rounds
random_seed         = 2026
use_best_model      = true
task_type           = GPU on one L40S; CPU fallback
```

Use the early-stopping partition only for checkpoint selection. Use the separate threshold partition for decision calibration. These settings are supported by CatBoost’s documented training interface. [Training parameters](https://catboost.ai/docs/en/references/training-parameters/)

Save the trained model immediately. GPU training is not bitwise deterministic, so retain the exact model artifact used for submission rather than relying on retraining to reproduce identical bytes. Score validation and final outputs using the same CPU inference path. [GPU reproducibility behavior](https://catboost.ai/docs/en/features/training-on-gpu)

### 11. Threshold selection, singletons, and France

#### One global threshold first

Use one global decision threshold across countries and target sources.

Procedure:

1. Score every candidate for the threshold-selection queries.
2. Sweep thresholds from 0 to 1 in increments of 0.01, plus an all-empty sentinel above 1.
3. Refine the best region in increments of 0.001.
4. Evaluate complete predicted sets, including references with zero candidates.
5. Prefer the higher threshold when scores differ by no more than 0.0002.
6. Save the selected threshold and the complete score curve.

Select using the known-country test-mixture proxy, while reporting the official unweighted validation macro score.

Do not force a fixed number of matches. If no candidate exceeds the threshold, output an empty list.

#### France policy

- Use the same normalizer, retrieval channels, numeric features, model, and threshold.
- Retain accent-sensitive and accent-folded views.
- Preserve French address and legal-form tokens.
- Do not require recognized US states or Indian PIN codes.
- Do not infer a French threshold from leaderboard changes.
- Do not force France’s predicted match-count distribution to resemble another country.

Inspect unlabeled French records to detect parsing and retrieval failures, not to create guessed ground truth.

#### Ambiguous ownership

Do not impose a global one-to-one assignment.

Source 1 can legitimately have many matches, and the French duplicate-reference finding makes forced target ownership unsafe without further evidence.

Report targets accepted for multiple references as an ambiguity diagnostic. Do not add graph propagation or an ownership rule in the initial pipeline.

### 12. Bounded improvement process

Keep experimentation local and limited.

| Run | Purpose | Decision |
|---|---|---|
| E0 | Data, metric, and retrieval audit | Establish a valid pipeline and measured recall ceiling |
| E1 | One classifier with complete feature groups | Establish the first submission-quality model |
| E2 | Two country-transfer diagnostics and stress slices | Identify failure under domain shift |
| E3 | At most one corrective model revision | Fix the largest observed error category |
| E4 | Locked evaluation | Confirm the frozen candidate/model/threshold configuration |

Allowed E3 changes:

- Repair a demonstrated normalization problem.
- Improve a retrieval channel responsible for measured misses.
- Add a feature addressing a repeated false-positive pattern.
- Add mild, identity-preserving query perturbations if clean-query training fails stress tests.

If augmentation is used:

- Apply it to at most 20% of sampled training references.
- Use one perturbed view per selected reference.
- Restrict transformations to typos, reordering, and observed formatting/suffix noise.
- Regenerate candidates for the perturbed query.
- Keep original and augmented variants in the same partition.
- Divide the reference’s total training weight across its variants.

Do not change several unrelated components at once.

#### Promotion gate

Promote the corrective version only if:

- Threshold-set macro improvement is at least 0.001.
- A paired, reference-level bootstrap gives a positive lower 95% confidence bound for the improvement.
- Neither US nor India loses more than 0.002 absolute macro F0.5.
- Singleton false-positive rate does not increase by more than 0.2 percentage points.
- Candidate size and runtime remain acceptable.
- The country-transfer and stress checks show no material new failure.

Use 500 paired bootstrap resamples. Compare per-reference scores, not independent candidate pairs.

Evaluate the locked partition only after choosing the final configuration. A failed final gate triggers rollback to the already validated baseline, not further tuning on the locked labels.

### 13. Runtime, memory, and recovery

#### Pilot before full execution

Run a 10,000-query pilot against the full appropriate target index.

Measure:

- Index-build time and size.
- Queries per second.
- Feature pairs per second.
- Peak RAM.
- Candidate count.
- Output bytes per reference.

Project the complete run:

\[
T_{\text{total}} =
T_{\text{index}}+
T_{\text{retrieve}}+
T_{\text{features}}+
T_{\text{score}}+
T_{\text{export}}
\]

Add a 50% contingency to the projection.

Do not start a full run whose projected completion leaves insufficient validation and upload time.

#### Batching

- Process countries separately.
- Use reference batches of approximately 5,000.
- Write completed candidate and score shards immediately.
- Limit workers to available CPU and measured memory.
- Avoid nested thread oversubscription.
- Bound feature caches.
- Use disk-backed indexes and Arrow/Parquet intermediate data.
- Never materialize dense cross-products or all pair strings.

Retain enough score data to change an already validated threshold without rerunning retrieval or features.

#### Recovery rules

- Resume only compatible completed shards.
- Never silently skip failed references.
- Do not convert a worker failure into empty predictions.
- Reduce worker count after memory pressure; do not arbitrarily reduce candidates.
- If the latest model cannot finish safely, use the last complete validated submission.
- Preserve raw inputs and submission artifacts throughout.

### 14. Testing and submission validation

#### Unit tests

Cover:

- Exact macro F0.5, especially empty sets.
- Tabs, commas inside fields, and empty output lists.
- Native-script preservation.
- Accent folding and transliteration fallback.
- Punctuation and legal suffix variants.
- Missing-address comparisons.
- Address ranges and compound numbers.
- Deterministic candidate deduplication and ties.
- Unknown country labels.
- Candidate/match subset enforcement.

#### Integration fixtures

Include:

1. A singleton.
2. Several true matches from each target source.
3. Two different businesses with the same name.
4. Different businesses sharing an address.
5. A cross-script true pair.
6. A trade-name pair recovered through its address.
7. A target with no address.
8. A true match with conflicting address numbers.
9. A France-like unseen-country record.
10. Two indistinguishable reference records with different IDs.
11. A reference with zero retrieved candidates.
12. A batch interruption and restart.

Verify that changing IDs or input row order does not change semantic match decisions.

#### Full-output checks

Internal validation must fail on:

- Missing or duplicate reference rows.
- Unknown target IDs.
- Wrong-source IDs.
- Duplicate IDs in a list.
- Malformed columns or encoding.
- Any match outside its candidate set.
- Any discrepancy between exported candidates and classifier-scored pairs.
- Incomplete prediction shards.

The supplied validator is insufficient as the only gate: ID existence checking is off by default, and candidate inconsistencies can produce only warnings.

Run the official validator with both files and `--check-ids` when memory permits. If necessary, run its matching-only ID check and perform the complete candidate checks through the memory-efficient internal validator. Do not omit those checks.

Also verify:

- Exactly 1,732,544 reference rows.
- France contributes exactly 259,452 rows.
- Candidate statistics agree with inference logs.
- Output files round-trip through a fresh parser.
- Repeated export from frozen scores gives identical bytes.

### 15. Submission schedule and deadline management

At review time, approximately 28 hours remained before the stated deadline. The following are work budgets, not measured runtime promises.

| Elapsed time | Milestone |
|---|---|
| 0–2 hours | Environment, input audit, metric tests, normalization, index build started |
| 2–4 hours | Retrieval pilot, recall ceiling, candidate-limit selection |
| 4–7 hours | Features, first classifier, threshold selection |
| 7–12 hours | Full test prediction, validation, first leaderboard submission |
| 12–18 hours | Country-transfer checks and at most one corrective revision |
| 18–23 hours | Final prediction and locked evaluation |
| Final 4 hours | Packaging, reproducibility checks, uploads, contingency |

Recalculate the schedule from actual start time.

#### Submission policy

- **Submission 1:** the first complete, locally validated classifier pipeline.
- **Submission 2:** only a materially improved version that passes the promotion gate.
- **Submission 3:** a validated final refinement or correction if genuinely necessary.
- Preserve at least two slots on the final day for operational corrections.
- Never submit an all-empty or knowingly weak file merely to test formatting.
- Never upload repeated threshold variants solely to probe the leaderboard.
- Confirm `SCORED` status after each upload.

If the first valid result is ready before September 26 ends, submit it then. Do not rush an invalid result to use an expiring slot.

Freeze new modeling work at least four hours before the deadline. Target final leaderboard upload by **10:00 PM IST on September 27**, retaining the remaining time for failures.

### 16. Final package and reproducibility

Package:

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── tests/
│       ├── configs/
│       ├── artifacts/
│       ├── README.md
│       ├── requirements.txt
│       └── LICENSE
└── Documentation_template.md
```

Include in `artifacts/`:

- Exact trained model.
- Selected threshold.
- Feature schema.
- Normalization configuration.
- Retrieval configuration.
- Split/sample manifests or deterministic reconstruction instructions.
- Input and output hashes.
- Validation summary.
- Dependency/license manifest.

Exclude raw datasets, large rebuildable indexes, temporary features, and credentials.

Document two reproduction paths:

1. Regenerate both outputs from the frozen model and supplied data.
2. Rebuild preparation, retrieval, features, and model training from the supplied data.

The first path must reproduce the submitted outputs. Explain expected GPU retraining nondeterminism for the second path.

For each submission, keep a separate immutable run directory containing both TSVs, model/configuration hashes, validation metrics, candidate statistics, public score, and upload time.

Final acceptance:

- Required TSVs pass strict checks.
- Predictions include every reference and country.
- Candidate file accurately represents model inference.
- The selected model has passed the frozen evaluation gate.
- The archive contains the exact leaderboard matching file selected for final delivery.
- Reproduction instructions work in a clean environment.
- Methodology documents measured results and limitations without claiming unmeasured French accuracy.
- Cloud resources used for the task are stopped after artifacts are preserved.

### 17. Assumptions and explicit defaults

- Ground truth is exhaustive for each training reference, as the supplied description states.
- Existing GPU hosts are available before the deadline and have substantial non-GPU resources; confirm exact capacity during preflight.
- One machine and one classifier are the default.
- AWS expenditure defaults to zero unless a measured resource shortfall justifies it.
- No external business lookup or external labeled dataset is used.
- Candidate limits and score thresholds are selected through the specified finite local procedures.
- No final-score guarantee is possible before implementing and measuring the pipeline.
- Team name and member details are supplied during packaging; they do not block technical work.

---

## Contents of `reject_ideas.md`

# Rejected and deferred approaches

“Rejected” means excluded from this implementation. “Deferred” means potentially useful in a longer project, but outside the challenge’s remaining time budget.

| Idea | Status | Rationale |
|---|---|---|
| Full pairwise comparison | Rejected | Approximately 17.27 trillion test comparisons; incompatible with scalable blocking. |
| Exact-name matching as the final rule | Rejected | Normalized names repeat frequently across distinct references. |
| Exact-address matching as the final rule | Rejected | Different businesses can share premises, and addresses can be incomplete. |
| Name-only candidate generation | Rejected | Misses trade names, initials, transliteration, and substantial name corruption. |
| Address-only candidate generation | Rejected | Missing addresses and shared or generic premises create failures. |
| Hard postcode or house-number blocking | Rejected | Components are missing, reordered, or inconsistent even in true matches. |
| One match per reference or per source | Rejected | The task is explicitly one-to-many; training contains up to eleven total matches. |
| Forced global target assignment | Rejected for this version | Identical normalized French references make arbitrary ownership decisions unsafe. |
| Connected-component or transitive graph merging | Rejected | A single false edge can create many false matches; the task is reference-centered. |
| Random pair-level train/validation split | Rejected | Leaks variants of the same reference identity across partitions. |
| Validation against a sampled target corpus | Rejected | Removes realistic distractors and inflates retrieval and matching performance. |
| Adding true matches to validation candidates | Rejected | Conceals blocking misses and overstates end-to-end performance. |
| Training mainly on random negatives | Rejected | Random negatives do not represent the plausible errors the model must reject. |
| Arbitrary positive/negative class balancing | Rejected as default | Distorts the candidate distribution and complicates threshold behavior. |
| Optimizing accuracy, micro F-score, or ordinary F1 | Rejected | These differ from the official singleton-aware macro F0.5. |
| A fixed 0.5 classifier threshold | Rejected | The challenge metric requires reference-level local threshold selection. |
| Country-specific models as the main design | Rejected | France has no labels, and multiple models increase calibration complexity. |
| Country one-hot encoding | Rejected | Encourages dependence on seen-country categories and handles France poorly. |
| Discarding non-Latin characters | Rejected | Removes useful evidence from Indian scripts and French text. |
| Transliteration as verified identity evidence | Rejected | Character transliteration is approximate and can introduce collisions. |
| External geocoding, business registries, or ER APIs | Rejected | Explicitly prohibited by the challenge. |
| Entity-ID or row-order features | Rejected | They do not represent business identity and risk brittle artifact exploitation. |
| Lookup tables transferring training matches to test | Rejected | IDs and normalized reference name/address signatures do not overlap in the audited US/India data. |
| Global dense embedding retrieval | Deferred | Adds model selection, encoding, indexing, and multilingual validation costs before establishing a lexical baseline. |
| An LLM or large cross-encoder for every pair | Rejected for this challenge run | Tens of millions of candidate pairs make inference and calibration costly. |
| Fine-tuning an 8B model because GPUs are available | Rejected | Compute availability alone does not justify the implementation and licensing burden. |
| Broad hyperparameter searches or large ensembles | Rejected | Diverts time from retrieval quality, error analysis, and reliable final execution. |
| Test-set pseudo-labeling | Deferred | Can reinforce false merges under country shift and weakens confidence in validation. |
| Synthetic French business generation | Rejected | Invented examples are not verified labels and may misrepresent the real test distribution. |
| Forcing test match counts or singleton rates to match training | Rejected | The test country mixture differs; its true match-count distribution is unknown. |
| Learned filtering followed by misleading candidate exports | Rejected | Candidate provenance must honestly describe the records scored by the matching model. |
| Dense all-corpus character-similarity matrices | Rejected | Truncating to top-k afterward does not prevent costly intermediate computation. |
| Spark, Kubernetes, distributed model training, or a feature store | Rejected | A single substantial machine should handle the proposed pipeline with much less setup. |
| SageMaker real-time endpoint | Rejected | The deliverable is batch TSV predictions, not an online service. |
| The tutorial’s smallest notebook instance for full processing | Rejected | Its demonstration workload is much smaller than the supplied 24.23-million-record dataset. |
| Depending solely on the starter validator | Rejected | ID checking is optional and some important candidate inconsistencies are warnings. |
| Repeated leaderboard threshold probing | Rejected | Consumes scarce submissions and risks fitting the public subset. |
| Last-minute unvalidated full retraining | Rejected | Changes scores and threshold behavior after the submission has already been verified. |
