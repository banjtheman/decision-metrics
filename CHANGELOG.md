# Changelog

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
