# DecisionMetrics

DecisionMetrics is a Python library for recording traces and metrics from
decision models such as Jev, Cloudflare Clef, Perplexity Decisions, and Strands
Decider.

A decision model chooses from a set of options supplied by your application.
Those options might be support queues, agent tools, or game movements.
DecisionMetrics records the inputs, available options, selected choice, and
probabilities when the model provides them. It measures response latency, token
usage, estimated API cost, and deadline misses, then links each decision to the
action your application applied, rejected, or skipped.

Use these records to debug an application, compare models and local or hosted
deployments, replay the same inputs across providers, or prepare reviewed training
datasets. Save decision traces to JSONL and export spans and metrics through your
application's OpenTelemetry setup.

Python 3.11+. The core uses only the standard library. OpenTelemetry is optional.

## Install and try it offline

With your Python 3.11+ environment active:

```bash
python -m pip install decision-metrics
decision-metrics demo --output output/demo.jsonl
decision-metrics summary output/demo.jsonl
```

For OpenTelemetry support, install `decision-metrics[otel]`. This is an alpha
release.

The demo is synthetic and makes no model or game calls. Output files are created
exclusively: use a fresh name for each run.

## Application examples

| Application | Model decision | Action recorded |
| --- | --- | --- |
| [Support-ticket routing](https://github.com/banjtheman/decision-metrics/blob/main/examples/support_routing.py) | Choose billing, technical support, or account access | Assign the ticket to an in-memory queue |
| [Agent tool selection](https://github.com/banjtheman/decision-metrics/blob/main/examples/agent_tools.py) | Choose documentation search, a status check, or a follow-up question | Execute a local example tool and record its result |
| Game controller | Choose from available movements or upgrades | Record whether the game accepted the action |

Clone the repository to run the application examples:

```bash
git clone https://github.com/banjtheman/decision-metrics.git
cd decision-metrics
python examples/support_routing.py --output output/support-demo.jsonl
python examples/agent_tools.py --output output/tools-demo.jsonl
decision-metrics summary output/support-demo.jsonl output/tools-demo.jsonl
```

Both scripts use synthetic inputs and deterministic offline rules by default.
To measure a decision model instead, set its credentials and select a configured
backend. For example, with `JEV_KEY` set:

```bash
python examples/support_routing.py --backend jev --output output/support-jev.jsonl
python examples/agent_tools.py --backend jev --output output/tools-jev.jsonl
```

The actions stay within the local examples. The journals record the selected
model's responses, timings, available token usage, and estimated API cost.
See the [examples guide](https://github.com/banjtheman/decision-metrics/blob/main/examples/README.md)
for configuration and recorded fields.

## Record a decision and its action

This example records a game controller's movement decision and whether the game
accepted it. Download the [sample provider configuration](https://github.com/banjtheman/decision-metrics/blob/main/examples/providers.toml)
to `examples/providers.toml` and start a local Strands decision server before
running it. The same recording pattern applies to other applications and models.

```python
from decision_metrics import ChoiceRequest, Harness, JsonlSink
from decision_metrics.providers import load_backend

backend = load_backend("examples/providers.toml", "strands-local")
try:
    with JsonlSink("output/run-01.jsonl") as sink:
        bench = Harness(backend, sink, metadata={"application": "my_game"})
        request = ChoiceRequest(
            state={"health": 5, "threat": "east"},
            options={"west": {"clearance": 150}, "east": {"clearance": 10}},
            instructions="Choose a movement that helps the player survive.",
        )
        decision = bench.choose(request, phase="combat", deadline_ms=850)
        if decision.expired():
            bench.record_action(decision, "expired", reason="combat_deadline")
        else:
            # Replace with the application's authoritative action execution.
            accepted = game.apply(decision.result.choice)
            bench.record_action(decision, "applied" if accepted else "rejected")
        bench.outcome("session_complete", metrics={"waves_completed": 9})
finally:
    backend.close()
```

`game.apply` is an application hook, not a library function. Pass the observation's
actual `time.monotonic()` timestamp as `observed_at` when it predates the call.
Action timestamps measure when the application records the result, which can be
after a bridge acknowledgement; they do not reveal the exact simulation tick.

Your application supplies the available actions and executes the selected one.
Record the execution result separately so the trace shows whether the decision
led to an action. Action statuses are `applied`, `rejected`, `expired`, `skipped`,
and `unknown` for delivery without a reliable acknowledgement.

## Providers

[examples/providers.toml](https://github.com/banjtheman/decision-metrics/blob/main/examples/providers.toml) contains separate entries for
Jev, Clef, Clef-flash, Perplexity, Strands local/remote, Kev, and meraGPT.
Download that file to `examples/providers.toml` before using the examples below,
or clone the repository to get all sample files. They are not installed with the
library.

| Transport | Configuration and credential environment |
| --- | --- |
| Jev System One | `provider = "jev"`; `JEV_KEY` |
| Cloudflare Workers AI | `provider = "clef"` or `"clef-flash"`; `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_AUTH_TOKEN` |
| Perplexity Decisions | `provider = "perplexity"`; `PERPLEXITY_API_KEY` |
| Local Strands / Kev | `provider = "strands"` or `"kev"`; local HTTP endpoint |
| meraGPT System One | `provider = "meragpt"`; `MERAGPT_API_KEY`; 2–10 options |
| Another compatible server | `provider = "systemone"`; explicit endpoint and model |

Hosted prices are dated defaults, with a source stored beside each rate. Override
them in TOML when they change. Local entries record zero API charges while
excluding hardware/energy costs. Remote Strands is a configuration template;
replace its endpoint and record the actual server hardware before using it.
No remote deployment is created by this library.

Use the same checkpoint for both Strands entries, and start its server with strict
window validation. The v19 checkpoint has a 4,096-token window; the server's
default truncation would change the benchmark input. The example's checkpoint,
device, and strict-window fields describe the intended server configuration;
they are not a runtime attestation. [Strands serving instructions](https://github.com/strands-labs/strands-decider).

Option descriptions supplied as objects, arrays, numbers, booleans, or null are
encoded as canonical JSON text for HTTP adapters; existing text stays unchanged.
Option IDs and presentation order are preserved. The journal keeps the original
structured request and records the rendering version in backend metadata.

The transports are tested against fixtures and a local HTTP server. Jev 1.13 and
the published Strands v19 checkpoint have also passed a 16-case saved-game replay.
Other live vendor integrations remain unverified.

## Freeze inputs and replay them

A packet is one JSON object per line with `state`, ordered `options`,
`instructions`, and optional `phase` and `case_id`. The [sample cases](https://github.com/banjtheman/decision-metrics/blob/main/examples/packets.jsonl) are synthetic.

```bash
decision-metrics replay examples/packets.jsonl \
  --config examples/providers.toml --backend strands-local \
  --limit 3 --max-cost 0.50 --output output/strands-local-replay.jsonl
decision-metrics summary output/strands-local-replay.jsonl
```

Replay is sequential and records actions as skipped. It measures response
validity, usage, latency, and distributions. It supplies no ground-truth action
labels or gameplay win rate. The cost limit uses known API usage and can be
crossed by the final request; unknown usage/prices remain explicit. Separate
request limits still apply.

Use separate entries for deployment comparisons. Local response speed is part of
the result; a faster local model can make more timely decisions in a live game.
A second experiment with a shared request cadence can help isolate choice
quality from deployment speed. Record cadence, hardware, precision, checkpoint,
and server settings in run metadata.

For full episodes, set call and billing limits high enough to finish. An identical
call ceiling can stop a faster controller earlier; budget stops belong in a
separate outcome category from game deaths.

## Export training data

Turn journals into a review queue, then supply human or teacher labels. The default
dataset preserves the original structured inputs, ordered candidates, and label
provenance:

```bash
decision-metrics export-training output/run-01.jsonl --output output/review-01
# Edit the label fields in output/review-01/review.jsonl.
decision-metrics export-training output/run-01.jsonl \
  --labels output/review-01/review.jsonl --output output/labeled-01
```

No labels are inferred from a model's confidence, a successful action, or an
episode outcome. To deliberately imitate another model, pass
`--teacher-backend jev`; those labels remain marked as teacher predictions.
Self-labels require an explicit opt-in. Stale, rejected, failed, or unacknowledged
decisions are excluded from training and retained in the review file with reasons.

Choose `--format strands` for Strands' native Choice examples, or
`--exporter your_package:YourExporter` for another runtime. Exporters control the
row format; shared code handles labels, outcome joins, whole-episode splits,
duplicate-input leakage checks, and file hashes. See the
[training-data guide](https://github.com/banjtheman/decision-metrics/blob/main/docs/training-data.md)
for annotations, held-out episodes, and the Python extension interface.

## Bring another backend

Implement `info`, `choose(request) -> DecisionResult`, and `close()`:

```python
from decision_metrics import BackendInfo, DecisionResult

class MyBackend:
    info = BackendInfo("my_backend", "checkpoint-v1", "local_cpu")

    def choose(self, request):
        selected_id = my_model.select(request.state, request.options, request.instructions)
        return DecisionResult(choice=selected_id)

    def close(self):
        pass
```

Return `Usage` when available. Missing probabilities, confidence, and usage stay
`None`; a generative or heuristic backend does not need to fabricate them.
Native HTTP decision adapters require the promised complete option distribution.
One legal option is recorded as forced without calling the backend.

Validation checks legal IDs, finite probabilities, complete distributions, sums,
and agreement between the provider's selected choice and its argmax. Declared
rounding precision permits bounded rounding error; raw values are retained.
Valid usage from invalid responses is still accounted for. Provider confidence
describes its decision distribution, not the probability of winning a game.

## OpenTelemetry

Install the `otel` extra and wrap the local sink after your application configures
its OpenTelemetry SDK providers/exporters:

```bash
python -m pip install 'decision-metrics[otel]'
```

```python
from decision_metrics import Harness, JsonlSink
from decision_metrics.otel import OpenTelemetrySink

with JsonlSink("output/run-otel.jsonl") as local:
    sink = OpenTelemetrySink(local)  # Uses the application's tracer and meter.
    bench = Harness(backend, sink)
    # Call bench.choose and bench.record_action as above.
```

It emits `decision.choose` spans and these application-defined metrics:

| Metric | Unit |
| --- | --- |
| `decision.requests`, `decision.actions`, `decision.deadline_misses` | count |
| `decision.duration`, `decision.observation_age` | seconds |
| `decision.tokens` | tokens, split by input/output |
| `decision.api_cost` | USD, known API cost only |

Metric attributes include provider, deployment, requested model, phase, and
status. Decision/run IDs and packet hashes are trace attributes rather than
metric dimensions. Raw prompt contents and probability vectors remain in JSONL.
These names are DecisionMetrics conventions, not official OTel semantic
conventions. Without an application SDK provider, the API is a no-op. [Official
Python instrumentation guide](https://opentelemetry.io/docs/languages/python/instrumentation/).

The local journal is written first. Export failures leave it intact and increment
`sink.export_failures`. The library does not select a collector or initialize
global SDK providers. Sink writing is synchronous; exporter overhead can affect
the control loop, so use the same export setup across benchmark entries.

## Events and summaries

Schema v1 uses `run_start`, `decision`, `action`, and `run_outcome` events. Each
decision carries a UUID, common-packet SHA-256, candidate order, exact prompt,
requested/returned model, deployment, timestamps, measured latency, deadline,
result/error, usage, and estimated API cost. Probabilities and provider confidence
remain in the result. Authentication headers and HTTP bodies from failures are
not recorded. Disable prompt contents with `Harness(..., record_inputs=False)`.

Summaries separate deployments and phases, preserve unknown usage/cost counts,
exclude forced actions from model latency and requests, and reject duplicate
decision/action IDs. Median and p95 use attempted requests; p95 is the nearest-rank
percentile. An application outcome is whatever the application actually observed.
No aggregate accuracy score is inferred from the model's own selections.

## Development

```bash
git clone https://github.com/banjtheman/decision-metrics.git
cd decision-metrics
python -m pip install -e '.[otel]' opentelemetry-sdk
python -m unittest discover -s tests -v
```

The OpenTelemetry tests require `opentelemetry-sdk`; the core tests do not. Tests
make no vendor calls. The project is MIT licensed.

See [publishing instructions](https://github.com/banjtheman/decision-metrics/blob/main/docs/publishing.md)
for release checks and PyPI Trusted Publishing setup.
