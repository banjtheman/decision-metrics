# DecisionMetrics

Standalone telemetry and replay for models that choose among supplied actions.
Record what the model chose, how long it took, what it cost, and whether the
application actually applied the action.

Python 3.11+. The core uses only the standard library. OpenTelemetry is optional.
The library has its own provider contract and no Vibecheck dependency.

## Try it offline

With your Python 3.11+ environment active:

```bash
git clone https://github.com/banjtheman/decision-metrics.git
cd decision-metrics
python -m pip install -e .
decision-metrics demo --output output/demo.jsonl
decision-metrics summary output/demo.jsonl
```

The demo is synthetic and makes no model or game calls. Output files are created
exclusively: use a fresh name for each run.

## Record a decision and its action

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

The game owns legal action generation, execution, freshness, resets, and scoring.
DecisionMetrics records those facts without executing an action or retrying a
stale model request. Action statuses are `applied`, `rejected`, `expired`, `skipped`,
and `unknown` for delivery without a reliable acknowledgement.

## Providers

[examples/providers.toml](examples/providers.toml) contains separate entries for
Jev, Clef, Clef-flash, Perplexity, Strands local/remote, Kev, and meraGPT.

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

The HTTP transports are implemented and tested against fixtures and a local HTTP
server. Live vendor compatibility, account access, and game performance still
need smoke tests. OpenAI Decisions is pending a verified account/API contract.

## Freeze inputs and replay them

A packet is one JSON object per line with `state`, ordered `options`,
`instructions`, and optional `phase` and `case_id`. The sample cases are synthetic.

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
python -m pip install -e '.[otel]'
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
python -m unittest discover -s tests -v
```

The OpenTelemetry tests require `opentelemetry-sdk`; the core tests do not. Tests
make no vendor calls. The project is MIT licensed.
