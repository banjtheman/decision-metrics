# Application examples

These examples show how to record model choices and the actions your application
takes after them. Each uses synthetic inputs and applies actions locally.

Clone the repository and install DecisionMetrics in a Python 3.11+ environment:

```bash
git clone https://github.com/banjtheman/decision-metrics.git
cd decision-metrics
python -m pip install decision-metrics
```

## Route support tickets

```bash
python examples/support_routing.py --output output/support-demo.jsonl
```

The example chooses between billing, technical support, and account access, then
adds each ticket to an in-memory queue. Its traces connect the ticket and queue
options to the decision and assignment. The run outcome records the queues and
number of tickets routed.

## Select an agent's next tool

```bash
python examples/agent_tools.py --output output/tools-demo.jsonl
```

The example chooses between documentation search, a service-status check, and a
request for more details. It executes a local fixture function and records its
result alongside the selection. The status data and documentation are synthetic;
the script makes no external tool calls.

## Use a decision model

Both scripts default to deterministic offline rules labelled `offline-demo` in
the telemetry. Their timings describe local example code. They do not supply
model probabilities or token usage.

To use Jev, set `JEV_KEY` in your environment and run:

```bash
python examples/support_routing.py --backend jev --output output/support-jev.jsonl
python examples/agent_tools.py --backend jev --output output/tools-jev.jsonl
```

Each script makes up to three decision requests and applies the selected actions to
the local example. The native adapter records returned probabilities and usage,
and the configured pricing supplies estimated API cost. Errors and expired
decisions are recorded without replacing them with offline rules.

Use `--backend` and `--config` to select another entry from
[providers.toml](providers.toml), including Clef, Perplexity, or a running local
Strands server. Set `--deadline-ms` to match the time budget for your application.

## Inspect the telemetry

```bash
decision-metrics summary output/support-demo.jsonl output/tools-demo.jsonl
```

The summary reports latency, action counts, errors, deadline misses, and available
usage and cost, with a breakdown by decision phase. Each JSONL journal retains
the inputs, option order, choice, and action details. Use a fresh output filename
for every run.

For OpenTelemetry spans and metrics, use the same calls with an
`OpenTelemetrySink` as shown in the main [README](../README.md).
