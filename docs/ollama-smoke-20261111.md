# Ollama live integration check — 2026-11-11

All nine live requests passed through DecisionMetrics with valid responses,
recorded token usage, and complete action records. Six synthetic cases exercised
Choice, Noul and Score through the replay CLI. The support-routing example then
assigned all three tickets to the expected in-memory queues.

The full automated suite also passed: 84 tests, including the optional
OpenTelemetry checks, with `python -m unittest discover -s tests -v`.

## Environment

- Ollama 0.40.16, local `http://localhost:11434/v1/systemone`, no API key.
- Model: `tev1:0.8b`, MLX runner, BF16 safetensors, reported 752M parameters.
- Downloaded model size: 1,524,372,929 bytes (approximately 1.52 GB).
- Model digest: `b661d56508169d2b6c4cab5a047409f50bd9694a215bc232469d44291f4bdbf1`.
- Hardware: Apple M3 Max, 64 GiB memory.
- Backend: `ollama-tev-small`; timeout 120 seconds, keep-alive five minutes.
- Library: local checkout with the unreleased Ollama provider, Python 3.11.

Ollama chose this model variant for the Mac. Other platforms or tags may download
a different format or size. Model documentation: [Tev1](https://ollama.com/library/tev1).

## Observed results

| Case | Question type | Observed answer | Expected check |
| --- | --- | --- | --- |
| Duplicate payment, refund requested | Choice | `billing`, probability 0.999499 | Billing selected |
| Checkout returns 500 errors | Choice | `technical`, probability 0.625910 | Technical selected |
| Explicit refund request | Noul | P(true) = 0.999668 | Above 0.5 |
| Everything works; no refund wanted | Noul | P(true) = 0.086489 | Below 0.5 |
| Thank-you ticket, no action needed | Score | 0.051025 on a 0–2 scale | Most likely level 0 |
| Production unavailable to everyone | Score | 1.989211 on a 0–2 scale | Most likely level 2 |
| Example ticket T-001 | Choice + action | `billing`, applied | Expected queue |
| Example ticket T-002 | Choice + action | `technical`, applied | Expected queue |
| Example ticket T-003 | Choice + action | `account_access`, applied | Expected queue |

The adapter validated the full probability distributions and score expectations,
preserved model labels and confidence where provided, and recorded 1,417 input
tokens and zero output tokens. All nine requests had known usage and no errors.
Replay recorded six skipped actions; the application recorded three applied
actions. The configured local API charge was zero, excluding hardware and energy.

The first request, including model loading, took 2,182 ms. The remaining eight
took 23–143 ms, with an 83 ms median. These are smoke-test observations on one
machine, not a latency benchmark or evidence of general model accuracy.

## Reproduce

From the repository root, with Ollama running:

```bash
python -m pip install -e .
ollama pull tev1:0.8b
python examples/support_routing.py --backend ollama-tev-small \
  --deadline-ms 120000 --output output/support-tev-fresh.jsonl
decision-metrics summary output/support-tev-fresh.jsonl
```

The original local artifacts are under
`output/ollama-tev-smoke-20261111-01/` (ignored by Git): `packets.jsonl` contains
the six exact synthetic replay inputs, `replay.jsonl` and `support-routing.jsonl`
contain the journals, and `environment.json` and `summary.json` preserve the model
metadata and combined summary. To repeat those six inputs in the same checkout:

```bash
decision-metrics replay output/ollama-tev-smoke-20261111-01/packets.jsonl \
  --config examples/providers.toml --backend ollama-tev-small --limit 6 \
  --max-cost 1 --output output/tev-replay-fresh.jsonl
```

Use fresh output filenames for each run. Nimble, direct cloud calls, and cloud
proxy calls were not exercised. This check covers the existing text/JSON,
single-question interface; images and multi-question requests are not exposed.
