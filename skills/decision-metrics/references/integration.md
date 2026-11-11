# Integration reference

This reference describes the current Python API. Check the installed version
before using provider-specific capabilities. Adapt the examples to the project's
dependency manager, layout, data policy and execution lifecycle.

## A runnable offline integration

Save this as a local example and run it with the project's Python interpreter.
It uses an explicitly named rules backend, makes no network calls, and records a
real assignment into an in-memory queue. Replace `RulesBackend.choose` with the
existing decision call or load a configured HTTP backend for a live integration.
The trace's run metadata and model label distinguish this example from inference.

```python
from pathlib import Path
from uuid import uuid4

from decision_metrics import (
    BackendInfo, ChoiceRequest, DecisionResult, Harness, JsonlSink,
)
from decision_metrics.report import read_events, summarize


class RulesBackend:
    info = BackendInfo("offline-rules", "rules-v1", "local-example")

    def choose(self, request):
        message = request.state["message"].lower()
        choice = "billing" if "charged" in message else "support"
        # No fabricated model probabilities, token usage, or API cost.
        return DecisionResult(choice=choice, model="rules-v1")

    def close(self):
        pass


backend = RulesBackend()
path = Path("output") / f"decision-smoke-{uuid4().hex}.jsonl"
queues = {"billing": [], "support": []}
try:
    with JsonlSink(path) as sink:
        bench = Harness(backend, sink, metadata={"application": "routing-smoke",
                                                "synthetic_inputs": True,
                                                "offline_demo": True})
        decision = bench.choose(ChoiceRequest(
            state={"id": "T-1", "message": "I was charged twice."},
            options={"billing": "Payments and refunds", "support": "Other help"},
            instructions="Choose the appropriate queue.",
        ), phase="routing")
        queues[decision.result.choice].append("T-1")
        bench.record_action(decision, "applied", details={"ticket_id": "T-1"})
        bench.outcome("example_complete", metrics={"tickets_routed": 1})
finally:
    backend.close()

report = summarize(read_events(path))
assert queues["billing"] == ["T-1"]
assert report["entries"][0]["actions"] == {"applied": 1}
print(path)
print(report)
```

The smoke example verifies the integration, not a model's quality. Built-in CLI
checks are also available; choose new output filenames for each invocation:

```bash
python -m decision_metrics demo --output output/demo-01.jsonl
python -m decision_metrics summary output/demo-01.jsonl
```

## Backend contract and action lifecycle

A custom backend exposes `info: BackendInfo`, `choose(request) -> DecisionResult`,
and `close()`. Optional methods are `noul(request) -> NoulResult` and
`score(request) -> ScoreResult`. The call to the real model must happen inside the
backend method so the harness measures it. Do not issue a second inference just
to generate telemetry. Keep any available model label, probabilities, confidence
and `Usage(input_tokens, output_tokens)` from the actual response; leave absent
fields as `None`. A Noul has no separate confidence field.

Check result semantics before adapting an existing model. With a Choice
distribution, validation requires the chosen option to have maximum probability.
A sampled choice may legitimately differ: keep the application's chosen option,
omit the incompatible optional distribution in the custom adapter, and document
why it remains in separate application telemetry. Never reselect the argmax just
to satisfy validation. A Score must be an expected zero-based level and agree
with any supplied level probabilities; an arbitrary existing rating is not
automatically a Score. Use a faithful documented mapping or leave the existing
path intact and provide an isolated example. These allowances apply to custom
adapters; malformed native HTTP responses remain errors, not stripped-down successes.

`DecisionError` takes a fixed lowercase diagnostic code, such as
`DecisionError("provider_unavailable", kind="transport")`. If the provider supplied
valid usage with an invalid result, include `usage=...` and `model=...` so cost
accounting can retain it. Never use exception messages or response bodies as codes.
The harness journals errors before raising them; arbitrary exceptions become
`unexpected_backend_error` without their text. Follow the existing application's
error policy instead of returning an invented option or rerunning the model.
If existing callers depend on specific exceptions, preserve that public error
contract after the harness has journaled the failure; test it explicitly, since
the harness normally raises `DecisionError` rather than the original exception.

Choose journal-failure behavior to match the application's logging and reliability
policy. `JsonlSink` file creation and writes are synchronous and I/O errors can
propagate; instrumentation is not automatically fail-open. An action may already
have executed when writing its action record fails. Do not retry the model or
action to repair missing telemetry. Document how the application surfaces or
handles storage failure, and test that path when it affects normal operation.

`bench.choose`, `bench.noul`, `bench.score` and `bench.ask` return a `Decision`;
the typed model response is `decision.result`. The action belongs to the
application. On each successful decision, record the actual action status once,
using that same decision object and harness:

| Status | Meaning |
| --- | --- |
| `applied` | The application confirmed execution or acceptance. |
| `rejected` | The application rejected the action or knows execution failed. |
| `expired` | The decision was too stale to apply. |
| `skipped` | Intentionally unused, including replay or observation-only runs. |
| `unknown` | Delivery/execution is uncertain; no reliable acknowledgement. |

If execution raises, record rejected only when failure is known; a timeout after
dispatch may require unknown. Then follow the application's error policy.
An inference error raises without a usable `Decision`: do not call
`record_action` with a fabricated ticket. Keep application action details free of
secrets and unneeded user content. `record_inputs=False` omits the request but does
not sanitize metadata, results, action details, or the state sent to the model.

For time-sensitive actions, pass `deadline_ms` and, if available, the original
observation's `time.monotonic()` value as `observed_at`. Check `decision.expired()`
before applying the action. Record an end-of-run result with
`bench.outcome(name, metrics={...})` only when the application has observed it.

## Request types

| Request | Input fields | Main result |
| --- | --- | --- |
| `ChoiceRequest` | `state`, ordered `options` map, `instructions` | `.choice`, `.probabilities`, `.confidence` |
| `NoulRequest` | `state`, `instructions`, optional `criteria` for true/false | `.probability` |
| `ScoreRequest` | `state`, `levels` lowest first, `instructions` | `.score`, `.probabilities` list, `.confidence` |

Inputs must be JSON-compatible and finite. HTTP adapters render structured option
descriptions as canonical JSON text. Candidate IDs and presentation order stay
intact. Score supports 1–10 levels; provider capability checks can be stricter.
One alternative is forced locally. Native providers require complete valid
distributions; custom backends can omit probabilities when unavailable.

Native Noul and Score depend on provider support. OpenAI Decisions rejects Noul
criteria; OpenRouter and Kev default to Choice only unless capabilities are
explicitly configured. Read the installed provider implementation/configuration
instead of assuming every backend supports all three types.

## Provider configuration

Use `from decision_metrics.providers import load_backend, make_backend`.
`load_backend("config/decisions.toml", "local-decider")` selects this example:

```toml
[backends.local-decider]
provider = "ollama"
model = "tev1:0.8b"
endpoint = "http://localhost:11434/v1/systemone"
deployment = "ollama-local"
timeout = 120
keep_alive = "5m"
input_per_million_usd = 0
output_per_million_usd = 0
price_source = "Local API charges only; excludes compute and energy"
```

Use this only when local Ollama is appropriate or requested, with a package revision
that includes the provider. Confirm the server and installed models; download
`tev1:0.8b` if a model installation is in scope. Its format and download size vary
by platform. Local requests need no key. Choice accepts 2–26 options; Score accepts
2–10 levels. The HTTP timeout allows model loading; application action deadlines
may be shorter. This adapter exposes text/JSON state and one question per call,
not image input or batched questions.

For other providers, copy only the relevant entry from the repository's
`examples/providers.toml`. Environment variable **names**, not values, belong in
`api_key_env`. Direct Ollama cloud uses endpoint `https://ollama.com/v1/systemone`,
model `clef`, and `OLLAMA_API_KEY`. The local cloud proxy uses `clef:cloud` through
a signed-in local server. Cloud pricing stays unknown unless configured; never
reuse the local zero-cost setting for it. Honor the chosen deployment and avoid
automatic paid-model substitution when credentials are missing.

## Replay, training, and OpenTelemetry

Replay packets are JSONL objects with `state`, `instructions`, `options` for a
Choice (default type), `type: "noul"` with optional `criteria`, or `type: "score"`
with `levels`. Optional `case_id` and `phase` identify cases.

```bash
python -m decision_metrics replay cases.jsonl --config config/decisions.toml \
  --backend local-decider --limit 3 --max-cost 0.50 --output output/replay-01.jsonl
python -m decision_metrics summary output/replay-01.jsonl
```

Replay makes real provider calls but records actions as skipped. `--max-cost`
uses known estimated charges, may be crossed by the final call, and cannot cap
unknown costs. Retain a small request limit for smoke checks. Export training only
when requested; human labels or explicit teacher choices are separate from the
observed application outcome.

For requested OpenTelemetry integration, install the `otel` extra and wrap the
local journal with `decision_metrics.otel.OpenTelemetrySink(local_sink)`. Pass
that wrapper to `Harness` and retain the owning context manager for `local_sink`.
The application configures the SDK/providers/exporters; the package does not.
Check that traces arrive and `sink.export_failures` remains zero. Raw prompts and
probability vectors stay in JSONL; decision IDs are trace attributes rather than
metric dimensions.
