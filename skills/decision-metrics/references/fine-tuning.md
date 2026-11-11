# Prepare and evaluate fine-tuning data

Use this reference when the user wants to improve a decision model from recorded
examples. DecisionMetrics prepares labeled datasets and records evaluation runs;
it does not supply an optimizer, trainer, or fine-tuning service. Keep dataset
preparation separate from launching a training job. Follow the user's requested
scope and any already authorized compute budget.

The maintained [training-data guide](https://github.com/banjtheman/decision-metrics/blob/main/docs/training-data.md)
documents the export schema and custom exporter interface. Check the installed
version before using commands or assuming trainer compatibility.

## Record inputs that can become training examples

Record the state, original ordered options or levels, instructions, model result,
and the application's observed action status. Set
`metadata={"episode_id": "episode-01"}` on the `Harness`; all runs belonging to
the same episode must share that ID. Choose an equivalent grouping unit for the
application, such as a conversation or ticket, so related observations stay
together. Without `episode_id`, the exporter groups by `run_id`.

Training export needs `record_inputs=True`, the default. Journals recorded with
`record_inputs=False` lack the requests and cannot supply training examples;
their rows are excluded as `inputs_not_recorded`. Do not reconstruct missing
inputs by guessing. Decide which inputs the project can retain, redact or
minimize them before recording, and keep resulting traces and datasets in the
project's intended storage. Changing recorded requests later breaks packet-hash
validation.

Eligible rows require a successful provider decision, at least two options or
levels where applicable, no recorded deadline miss, and an `applied` or `skipped`
action. Labeled replay is therefore usable. `--require-applied` excludes skipped
actions. Errors, forced results, expired/rejected/unknown actions, and missing
action records are excluded with reasons. Do not relabel an execution failure as
success to make its row eligible.

## Review and assign explicit targets

Create a review queue in a fresh directory:

```bash
decision-metrics export-training output/run-01.jsonl --output output/review-01
```

Without labels, `train.jsonl`, `validation.jsonl`, and `test.jsonl` are empty.
Edit `label` in `review.jsonl`, preserving its IDs and packet hash, or supply a
separate annotation file. Each annotation identifies the source `run_id` and
`decision_id` and includes a label with the correct target field:

| Request | Target field | Meaning |
| --- | --- | --- |
| Choice | `"choice": "technical"` | An existing option ID, not its position or display text |
| Noul | `"answer": true` | A Boolean target, not a probability |
| Score | `"level": 2` | A zero-based integer level, not the expected fractional score |

Every label also needs `source` and `labeler`. Preserve the actual origin:

- `human`: an identified person reviewed and supplied the target.
- `teacher`: another model supplied an imitation target; identify its checkpoint.
- `policy`: a documented deterministic rule supplied the target; identify the
  rule version. An agent applying such a rule must not call it human review.
- `self`: the target model supplied its own prediction. Use this only as an
  explicit experiment and pass `--allow-self-labels` when exporting those labels.

Optional `details` can record the rationale, policy, annotation tool version, or
other JSON metadata. Optional `weight` must be finite and positive. Do not invent
human labels, reinterpret confidence as correctness, or infer the best action
from acceptance or a successful episode.

Export reviewed labels:

```bash
decision-metrics export-training output/run-01.jsonl \
  --labels output/review-01/review.jsonl --output output/labeled-01
```

When the user explicitly wants to imitate recorded predictions, select the exact
backend name present in the journal:

```bash
decision-metrics export-training output/jev-replay.jsonl \
  --teacher-backend jev --output output/jev-teacher-data
```

This reads existing responses; it makes no teacher API calls. Explicit annotations
override the generated teacher targets. Noul teacher targets use probability
`>= 0.5`; Score targets use the most probable level, or rounded expected score
when no distribution was recorded. These remain teacher predictions. If they came
from the target model itself, annotate `source: self` and use the explicit
self-label opt-in rather than disguising them as independent supervision.

## Split whole episodes before training

Without `--splits`, every group goes to `train`. Supply a JSON mapping that covers
exactly the source episode groups, for example:

```json
{"episode-01":"train","episode-02":"validation","episode-03":"test"}
```

```bash
decision-metrics export-training output/episode-01.jsonl output/episode-02.jsonl \
  output/episode-03.jsonl --labels output/labels.jsonl \
  --splits output/splits.json --output output/dataset-01
```

Keep related conversations, sessions, or near-duplicate cases in the same split;
do not randomly divide adjacent rows. The exporter rejects identical packets
across splits and conflicting targets for identical packets. It does not detect
semantic similarity or near duplicates. Repeated packets within a split remain,
so inspect class balance, repeated examples, phase coverage, and weighting.

Inspect `manifest.json`: label sources, exclusions, split counts, question types,
episode mapping, and file hashes. `evaluation_ready: true` only establishes that
training data and at least one labeled held-out split exist; it does not certify
sample quality, adequate size, or an untouched final test set.

## Match the dataset to a real training runtime

The default format is `decision-jsonl-v1` (`--format decision-jsonl`), containing
the request, candidate order, explicit label, and provenance. It is a portable
dataset, not a promise that an arbitrary trainer accepts it directly.

The built-in Strands export is:

```bash
decision-metrics export-training output/jev-replay.jsonl \
  --teacher-backend jev --format strands --output output/strands-data-01
```

Its format name remains `strands-choice-jsonl`. Choice targets become candidate
indices; Noul options are false then true; Score options are levels in ascending
order. Choice rows have been checked with strands-decider 0.1.0's native loader
and pointer collator. Noul and Score rows follow the documented shape but have
not been verified with that loader. Verify the exact loader and question types
before training them.

For another runtime, inspect its current official model/trainer documentation
after selecting a compatible base checkpoint, tokenizer, decision head, and
training method. Implement a `TrainingExporter` when a conversion is needed:

```bash
decision-metrics export-training output/run-01.jsonl --labels output/labels.jsonl \
  --exporter my_exporters:MyExporter --output output/trainer-data-01
```

`my_exporters:MyExporter` denotes an implementation the project must supply, not
a built-in exporter. Preserve candidate order, correct target binding, and label
provenance through conversion. Current exports contain one supervised target per
question; soft probability targets, preference data, and reinforcement learning
require separate designs.

Run the chosen trainer's dataset loader and collation on a small sample first.
Check rendered token lengths for every example, target indices, option shuffling,
and train/held-out separation. Do not silently truncate the state or candidates.
Use the selected checkpoint's actual limits; the published Strands v19 checkpoint,
for example, has a 4,096-token window.

Before starting training, establish the authorized step/time/cost budget, output
checkpoint location, and stop condition. A request to set up telemetry or export
data alone does not request a training job. DecisionMetrics has no training CLI,
and its Ollama adapter provides inference only. Do not invent an Ollama
fine-tuning endpoint or assume an externally trained artifact is importable;
verify the selected training and serving formats when that deployment is chosen.

## Compare the baseline and tuned checkpoint

Preserve the original checkpoint. Configure distinct backend names for the
baseline and tuned model, then replay the same held-out packets, options, order,
and instructions with the same request limits. Keep independently supplied
evaluation targets separate from the model's answers and from training labels
used to tune the model. Compare Choice selection against its target, Noul results
with a declared threshold and probability metric, and Score results against the
rubric using a declared error or level-agreement metric.

The replay CLI and `summary` report validity, usage, latency, costs, and action
statuses; they do not compute accuracy. Join results to held-out targets by stable
case ID or packet hash in project evaluation code. Count failures and missing
responses explicitly, and distinguish teacher agreement from independent target
accuracy. Tune on validation data and reserve final test data for the final
comparison; successful replay alone does not prove improvement.

Record the tuned checkpoint/revision in backend metadata alongside the base
checkpoint, dataset manifest hash, label provenance, trainer/version, training
configuration, seed, and evaluation split. Carry that identity into subsequent
DecisionMetrics traces. Report the measured quality and latency tradeoffs and
any unsupported question types or deployment limitations before claiming the
fine-tuned model is ready for use.
