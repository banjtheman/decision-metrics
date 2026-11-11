# Google Decisions live integration check — 2026-11-11

All eleven live requests passed through DecisionMetrics with valid responses,
known token usage, and complete action records. Eight synthetic replay checks
covered Choice, Noul, Score, and one-sided Noul criteria. The support-routing
example assigned all three tickets to the expected in-memory queues. No adapter
changes were needed after the live checks.

The automated suite passed 96 tests before the live run, including local HTTP
fixtures and the optional OpenTelemetry checks. These live checks complement
that coverage; they do not establish general model accuracy.

## Configuration

- Requested model: `gemini-decision-1`.
- Endpoint: `https://generativelanguage.googleapis.com/v1beta/decisions`.
- Authentication: API key in the `x-goog-api-key` header.
- Backend: `google-decisions`, deployment `google-hosted`, HTTP timeout 30 seconds.
- Library: local checkout with the unreleased Google Decisions adapter.
- Input: synthetic text or structured data serialized as JSON text, one question
  per request. The requests did not use media or multi-question payloads.

Google omitted a returned model label, so journals retain the requested model
but leave `returned_model` and the result's `model` as null. No model revision is
inferred from the request. API documentation: [Google Decisions](https://aistudio.google.com/docs/decisions-api).

## Observed results

| Case | Type | Observed answer | Expected check |
| --- | --- | --- | --- |
| Duplicate payment, refund requested | Choice | `billing`, probability 0.997031 | Billing selected |
| Checkout returns 500 errors | Choice | `technical`, probability 0.959309 | Technical selected |
| Explicit refund request | Noul | P(true) = 0.961059 | Above 0.5 |
| Everything works; no refund wanted | Noul | P(true) = 0.010000 | Below 0.5 |
| Thank-you ticket, no action needed | Score | 0.068636 on a 0–2 scale | Most likely level 0 |
| Production unavailable to everyone | Score | 1.997861 on a 0–2 scale | Most likely level 2 |
| Refund request, true criterion only | Noul | P(true) = 0.973411 | Above 0.5 |
| Refund request, false criterion only | Noul | P(true) = 0.989470 | Above 0.5 |
| Example ticket T-001 | Choice + action | `billing`, applied | Expected queue |
| Example ticket T-002 | Choice + action | `technical`, applied | Expected queue |
| Example ticket T-003 | Choice + action | `account_access`, applied | Expected queue |

The eleven requests comprised five Choice, four Noul, and two Score decisions.
The adapter validated full distributions and Score expectations, mapped native
usage totals to 1,434 input tokens and zero output tokens, and preserved Choice
and Score confidence. Replay recorded eight skipped actions; the example
recorded three applied actions. There were no errors, deadline misses, missing
action records, or unknown-usage requests.

Measured request latency ranged from 231–333 ms, with a 257 ms median. These are
observations from a small integration check, not a latency benchmark. The cost
estimate was $0.0001434 using the configured paid-tier rate of $0.10 per million
input tokens. The account's actual billing tier and charges were not verified;
free-tier usage may have no charge. See [Google's pricing](https://aistudio.google.com/docs/pricing#gemini-decision).

## Reproduce

From the repository root, with `GEMINI_API_KEY` already set in the environment:

```bash
python -m pip install -e .
python examples/support_routing.py --backend google-decisions \
  --deadline-ms 30000 --output output/support-google-fresh.jsonl
decision-metrics summary output/support-google-fresh.jsonl
```

The original local artifacts are under
`output/google-decisions-live-20261111-01/` (ignored by Git): `packets.jsonl`
and `partial-packets.jsonl` contain the exact replay inputs, `replay.jsonl`,
`partial-criteria.jsonl`, and `support-routing.jsonl` contain the journals, and
`summary.json` combines their metrics. To repeat the six main replay inputs in
the same checkout:

```bash
decision-metrics replay output/google-decisions-live-20261111-01/packets.jsonl \
  --config examples/providers.toml --backend google-decisions --limit 6 \
  --max-cost 0.01 --output output/google-replay-fresh.jsonl
```

Use fresh output filenames for every run. The applied actions in this check
only modified the example's in-memory queues. No training or production actions
were performed. Media, multiple questions in one request, and Google's larger
Score scales remain outside this adapter's current interface.
