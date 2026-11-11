---
name: decision-metrics
description: Integrate DecisionMetrics into an application's model decisions, verify traces and replay, or prepare reviewed datasets for fine-tuning. Use when setting up or extending DecisionMetrics, including with an existing custom backend or Ollama.
license: MIT
---

# DecisionMetrics

Implement a working integration at an actual application decision boundary.
DecisionMetrics is a Python 3.11+ library for Choice, Noul (yes/no probability),
and Score (ordered rating) decisions. It records what the model saw, its answer,
latency, available usage/cost, and what the application subsequently did. It does
not execute application actions or automatically instrument arbitrary model SDKs.

## Inspect and choose the integration

Read the project's instructions, dependency manifests, model call sites, action
execution, tests, and logging conventions. Preserve the user's provider and the
existing decision behavior. Choose a small, useful decision boundary: routing a
ticket, selecting a tool, scoring an item, or choosing a game action.

- **Existing decision code:** wrap its call in a custom backend; call it once,
  inside the measured backend method. Do not call the model first and then time
  returning a cached answer. Check that its answer semantics fit the typed result
  before adapting it: Choice with probabilities requires an argmax selection,
  and Score means an expected zero-based level. Preserve application behavior;
  follow the reference's compatibility guidance for sampled choices or other ratings.
- **New decision model integration:** use a documented provider via
  `make_backend(settings)` or `load_backend(path, name)`. Reuse the application's
  credentials and configuration conventions.
- **Non-Python project:** use an existing Python component if it owns the decision.
  Otherwise propose the smallest Python worker or service boundary that fits the
  architecture; there is no JavaScript SDK in this repository. Keep end-to-end
  application latency separate from Python-side inference latency. Do not rewrite
  the application or introduce a production service without that scope being agreed.
- **No natural decision boundary:** create an isolated runnable example related to
  the project and explain where a future integration would attach. Do not inject
  model decisions into unrelated application behavior just to create a trace.

Proceed with reasonable defaults. Ask only for missing choices that change the
integration materially; missing credentials should not block configuration,
offline checks, or documentation.

## Install the matching version

Use the project's existing Python environment and dependency manager. Add
`decision-metrics` through its normal manifest/lockfile workflow. The core has no
third-party runtime dependencies; `decision-metrics[otel]` is optional. Noul and
Score require 0.4.0 or later.

Check the installed API against the version being documented. Ollama and Google
Decisions support are on repository `main` and unreleased; do not assume
the PyPI 0.4.0 package contains them. If the required provider is absent from
`python -m decision_metrics providers`, inspect the current release or use a
reviewed repository revision, recorded in the project's dependencies. Avoid
silently selecting a floating development version.

Canonical sources (use the installed source or matching Git revision when available):

- [README and Python interface](https://github.com/banjtheman/decision-metrics/blob/main/README.md)
- [Provider configuration](https://github.com/banjtheman/decision-metrics/blob/main/examples/providers.toml)
- [Application examples](https://github.com/banjtheman/decision-metrics/tree/main/examples)
- [Training export](https://github.com/banjtheman/decision-metrics/blob/main/docs/training-data.md), only when requested.

If fetched as a single raw file, retrieve the linked reference relative to this
file's URL. When installed, keep this whole skill directory together.

## Implement and verify

Read [integration reference](references/integration.md) for exact APIs, a runnable
offline example, provider configuration, and lifecycle/error handling.

For requested fine-tuning or distillation work, read the
[fine-tuning reference](references/fine-tuning.md). It covers recording usable
inputs, reviewing labels, splitting and exporting data, selecting a compatible
trainer, and comparing the tuned model with a baseline. Training is a separate
workflow, not part of a routine telemetry setup.

1. Connect `Harness` to a backend and `JsonlSink` at the chosen boundary. Build
   `ChoiceRequest`, `NoulRequest`, or `ScoreRequest` from the application's actual
   inputs. Keep option IDs stable and Score levels in their original order.
2. Handle `DecisionError` using the application's existing failure policy. A
   failed model response must not become a fabricated successful decision. If the
   application uses a fallback, identify it separately in application telemetry.
3. Execute the application's action once, then record its real result with
   `record_action`: applied, rejected, expired, skipped, or unknown. Record failed
   execution too; dispatch without acknowledgement is not proof of application.
4. Respect the project's data policy. Inputs are stored by default; set
   `record_inputs=False` when content storage is unwanted, or redact inputs before
   constructing requests if appropriate. This flag does not prevent sending state
   to the configured model. Keep credentials out of journals, metadata and source.
5. Run an offline example and a representative integration check using the
   project's existing test approach. Inspect the JSONL and summary, not just exit
   status: legal answer, action outcome, latency, usage and missing-data behavior.
   Check relevant failure/deadline paths for the actual integration. If a configured
   live model is available and in scope, make a small synthetic-input smoke call.
   Label offline versus live results explicitly.
6. Provide the changed files, exact run/summary commands, selected provider/model,
   tested behavior, and any remaining setup. Add brief project documentation and
   an environment-variable template if needed. Explain when an integration is only
   an offline example or needs credentials; do not claim deployment or live success.

## Metrics and operational rules

- Missing probabilities, confidence, token usage or prices remain unknown. Local
  zero API cost must be explicit and excludes compute/energy. Never infer accuracy
  from the model's confidence or from an action being accepted.
- One option or one Score level produces a forced result without calling the
  backend. To test inference and timing, supply at least two alternatives.
- A Noul returns a probability, not an application decision threshold. A Score is
  an expected zero-based level, which may be fractional; it is not normalized to 0–1.
- A deadline marks stale results; it does not cancel HTTP inference. Check expiry
  before applying an action. Transport timeout and action deadline are different.
- The backend and sink are synchronous. Avoid blocking an async event loop, and
  isolate connections/harnesses per sequential worker rather than concurrently
  sharing one backend. Close the backend and sink at the owning lifecycle boundary.
- Output journals are created exclusively. Use a fresh filename for each run.
  Replay marks actions skipped; it does not execute the application's actions.
- Reuse the application's OpenTelemetry SDK/exporters when requested. Installing
  the optional API alone does not configure collection. Training export requires
  explicit reviewed labels or an explicitly chosen teacher; do not invent labels.
