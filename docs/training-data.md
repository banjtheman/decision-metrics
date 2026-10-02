# From decision traces to training data

DecisionMetrics records what a model saw, what it chose, whether the application
executed that choice, and what happened during the run. These are observations.
A supervised dataset also needs an explicit target and an account of who supplied
it. An action that was accepted, or a run that ended well, does not establish the
best choice among actions that were never taken.

The dataset pipeline is independent of a model's training runtime. It uses the
standard library and writes a model-neutral format by default. A serializer adapts
the labeled records to a runtime such as Strands.

## Review and label

Record with `Harness(..., record_inputs=True)` (the default), then export:

```bash
decision-metrics export-training output/run-01.jsonl --output output/review-01
```

The new directory contains `review.jsonl`, `train.jsonl`, `validation.jsonl`,
`test.jsonl`, and `manifest.json`. Without labels, the data files are empty and the
review rows have `label: null`. Edit those fields, or create a separate JSONL file
with annotations like this:

```json
{"run_id":"episode-01","decision_id":"COPY_FROM_TRACE","label":{"choice":"technical","source":"human","labeler":"reviewer-v1","details":{"reason":"The ticket describes a service outage."}}}
```

`choice` must be a legal option ID, not its display name or numeric position.
`source` is `human`, `teacher`, `policy`, or `self`. `labeler` identifies the reviewer,
teacher checkpoint/version, or policy. Optional `weight` is finite and positive;
optional `details` holds JSON metadata such as an explanation or annotation tool
version. Unknown fields in a label are rejected. Review rows include a packet hash;
it is checked when they are read back.

```bash
decision-metrics export-training output/run-01.jsonl \
  --labels output/review-01/review.jsonl --output output/labeled-01
```

Human labels can correct the recorded model's choice. The original response,
action result, observation ID, run metadata, and episode outcome remain in
provenance. The model-neutral `decision-jsonl-v1` row contains `request`, explicit
`candidate_order`, `label`, and `provenance`; structured option values stay intact.

To explicitly distill recorded predictions from a named backend:

```bash
decision-metrics export-training output/jev-replay.jsonl \
  --teacher-backend jev --output output/jev-teacher-data
```

Those targets have `source: teacher` and identify the returned model. Explicit
annotations override automatic teacher labels. A teacher prediction remains an
imitation target; teacher agreement is not an optimal-action accuracy score. If
using the target model's own predictions, annotate them as `self` and pass
`--allow-self-labels` so that the experiment records this choice explicitly.

The default eligibility rules require a successful provider decision with at least
two options, no recorded deadline miss, and an `applied` or `skipped` action. This
allows labeled offline replay. Add `--require-applied` to require real execution.
Errors, forced choices, missing inputs, missing acknowledgements, rejected actions,
expired actions, and other unknown statuses are excluded. Their review rows record
the reason. A label for an excluded decision is rejected rather than silently used.

## Hold out whole episodes

Supply `metadata={"episode_id": "episode-01"}` when constructing a Harness. Logs
from the same episode must share that ID. If it is absent, the run ID is the group.
Split mappings operate on these groups, so adjacent observations stay together.

Create a JSON object covering every source episode group:

```json
{"episode-01":"train","episode-02":"validation","episode-03":"test"}
```

```bash
decision-metrics export-training output/episode-01.jsonl output/episode-02.jsonl \
  output/episode-03.jsonl --labels output/labels.jsonl \
  --splits output/splits.json --output output/dataset-01
```

An identical packet cannot occur in different splits, even if the run IDs differ.
Conflicting targets for the same packet are rejected. Repeated packets within one
split remain present, so sampling and weighting remain the trainer's responsibility.
Packet hashes identify exact inputs; they do not detect semantically similar states.

Without a split map, all groups go to `train`. The manifest reports
`evaluation_ready: false` until labeled training data and at least one labeled
held-out split exist. This is a structural check, not a claim of sample adequacy.
It records eligibility/exclusion counts, label sources, phase counts, source-event
hash, episode mapping, exporter details, and SHA-256 hashes for each data file.
Use a fresh output directory for every export.

## Strands adapter

```bash
decision-metrics export-training output/jev-replay.jsonl \
  --teacher-backend jev --format strands --output output/strands-data-01
```

`StrandsChoiceExporter` writes the native `Example.from_dict` fields: `kind`,
`state`, `instructions`, ordered `[option_id, description]` pairs, numeric `label`,
`weight`, and `task`. Extra provenance is retained and ignored by that loader.
The label index follows the original candidate order. Structured descriptions use
the same `json-descriptions-v1` rendering as HTTP inference.

The adapter has been checked with strands-decider 0.1.0's native loader and pointer
collator. The published v19 checkpoint has a 4,096-token window. Token counting and
training remain runtime-specific: validate every rendered prompt before passing
it to a trainer that may truncate. Use native option shuffling to preserve target
binding, and supply explicit episode-separated files rather than randomly
splitting adjacent rows. Keep the published checkpoint as the baseline before
training or resetting its adapter/head. [Strands source](https://github.com/strands-labs/strands-decider).

## Add another exporter

Implement the `TrainingExporter` protocol. A serializer receives independent
copies of the original `ChoiceRequest`, validated label, and joined provenance.
Return a finite JSON object in the format your trainer expects:

```python
# my_exporters.py
class MyExporter:
    name = "my-choice-jsonl-v1"
    details = {"trainer_version": "1.0"}  # Optional manifest metadata.

    def serialize(self, request, label, provenance):
        return {
            "observation": request.state,
            "question": request.instructions,
            "candidates": list(request.options.items()),
            "target": label["choice"],
            "sample_weight": label["weight"],
            "provenance": provenance,
        }
```

```bash
decision-metrics export-training output/run-01.jsonl --labels output/labels.jsonl \
  --exporter my_exporters:MyExporter --output output/my-data-01
```

The module must be importable in the active Python environment. The CLI loads a
trusted no-argument class or an exported instance. The Python API also accepts an
instance directly:

```python
from decision_metrics import export_training, read_annotations
from decision_metrics.report import read_events
from my_exporters import MyExporter

manifest = export_training(
    read_events("output/run-01.jsonl"), "output/my-data-01",
    annotations=read_annotations("output/labels.jsonl"),
    exporter=MyExporter(),
)
```

The built-in serializers are model-neutral JSONL and Strands Choice. Kev and Clef
can use the same inputs, labels, and split pipeline after their trainer-specific
schemas are verified; this release does not claim native exporter support for
those runtimes. This API covers supervised Choice decisions. Reward/reinforcement
learning datasets, preferences, Score/Noul targets, optimizers, and fine-tuning
jobs are separate work.
