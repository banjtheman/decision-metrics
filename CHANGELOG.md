# Changelog

## Unreleased

- Record Noul (yes/no) and Score (ordered levels) questions next to Choice:
  `NoulRequest`/`NoulResult`, `ScoreRequest`/`ScoreResult`, `Harness.noul()`,
  `Harness.score()`, and `Harness.ask()` for any type. Events carry
  `question_type`; journals without it are read as Choice. A one-level Score is
  forced like a single option.
- Validate Noul probabilities, and Score ranges, per-level distributions, sums and
  the expected level within the provider's rounding.
- Send Noul and Score natively to Jev, Perplexity, Clef, Clef-flash, meraGPT,
  Strands and generic System One servers. Map them to OpenAI's `predicate` and
  `score` questions; a Noul with criteria is refused there before any request.
  OpenRouter and Kev stay Choice-only unless a backend sets `question_types`. A
  backend without a type records `unsupported_question_type` without a request.
- Replay `"type": "noul"` and `"type": "score"` packets; summaries count decisions
  by question type; OpenTelemetry emits `decision.noul` and `decision.score` spans
  and a `decision.question_type` attribute.
- Export Noul (`answer`) and Score (`level`) labels and teacher targets, in the
  model-neutral format and as Strands rows (`false`/`true` options, levels by
  number).
- Jev's default key is now `TYPESAFE_API_KEY`, TypeSafe's documented name;
  `JEV_KEY` still works when it is unset.
- Jev's default model is now `jev-1.14.0`, the current `jev-latest`. Set
  `model = "jev-1.13.0"` to keep comparing against earlier runs.
- Validated live with Jev 1.13 and 1.14 for Noul and Score; the other providers'
  support follows their documentation and is tested against fixtures.

## 0.3.0 — 2026-10-11

- Add the native OpenRouter Decisions transport with configurable model IDs,
  credential environment variables, and explicit prices. Disable provider
  fallbacks and preserve the original choices, distributions and token usage.
- Add Microsoft Decision-1 configuration and validate 16 frozen Brotato inputs
  through OpenRouter; these probes are not optimal-action labels.
- Add the native OpenAI Decisions transport, including ordered typed choices,
  refusal handling, strict distribution validation and its endpoint-specific
  input-only rate. Both transports use the standard library without a new SDK.
- Validate both transports in live Brotato episodes and retain the existing
  model-neutral telemetry and training-data export formats.

## 0.2.0

- Add model-neutral training-data export with explicit labels and complete
  decision/action/episode provenance, plus a review queue for unlabeled traces.
- Add whole-episode split mappings, checks for duplicate-input leakage and
  conflicting labels, eligibility filters, and hashed dataset manifests.
- Add the native Strands Choice serializer and a public `TrainingExporter`
  interface. Custom serializers work through Python or `--exporter module:Class`.
- Render structured HTTP option descriptions as canonical JSON text, preserving
  option IDs, order, and fields. This fixes Strands 0.1.0's text-only criteria
  contract and records the rendering version in traces.
- Validate Jev 1.13 and the published Strands v19 model on 16 frozen Brotato
  inputs; validate the exported teacher labels with Strands' native data pipeline.
