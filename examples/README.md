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

To use Jev, set `TYPESAFE_API_KEY` in your environment (the older `JEV_KEY` name
also works) and run:

```bash
python examples/support_routing.py --backend jev --output output/support-jev.jsonl
python examples/agent_tools.py --backend jev --output output/tools-jev.jsonl
```

Each script makes up to three decision requests and applies the selected actions to
the local example. The native adapter records returned probabilities and usage,
and the configured pricing supplies estimated API cost. Errors and expired
decisions are recorded without replacing them with offline rules.

Use `--backend` and `--config` to select another entry from
[providers.toml](providers.toml), including Clef, Perplexity, OpenAI Decisions,
Microsoft Decision-1 through OpenRouter, Ollama, or a running local Strands server.
The OpenAI and OpenRouter adapters require DecisionMetrics 0.3.0 or later:
`python -m pip install --upgrade 'decision-metrics>=0.3.0'`.
Set `--deadline-ms` to match the time budget for your application.

## Use Ollama locally or in the cloud

Ollama support is unreleased, so install from this checkout. Start Ollama v0.35.0+
and download Nimble before running the examples:

```bash
python -m pip install -e .
ollama pull nimble
python examples/support_routing.py --backend ollama-nimble \
  --deadline-ms 60000 --output output/support-ollama-01.jsonl
python examples/agent_tools.py --backend ollama-nimble \
  --deadline-ms 60000 --output output/tools-ollama-01.jsonl
```

The `ollama-nimble` entry calls `http://localhost:11434/v1/systemone` without a
key. Its 60-second HTTP timeout accommodates model loading, and `keep_alive`
keeps the model loaded for five minutes after a request. The example deadline is
also 60 seconds; reduce it to your application's actual budget for measurements.
Choice, Noul and Score all work with `provider = "ollama"`; choices are limited
to 26 options. See the main [README](../README.md#ollama-decision-models) for the
Python request interface and scope of support.

For a smaller local model, use [Tev1 0.8B](https://ollama.com/library/tev1:0.8b):

```bash
ollama pull tev1:0.8b
python examples/support_routing.py --backend ollama-tev-small \
  --deadline-ms 120000 --output output/support-ollama-tev-01.jsonl
```

The `ollama-tev-small` entry allows 120 seconds for model loading and inference,
keeps the model loaded for five minutes, and records zero API charges. It uses
the same Choice, Noul and Score interfaces.

To use `clef:cloud` through a local Ollama v0.40.5+ server signed in to ollama.com:

```bash
python examples/support_routing.py --backend ollama-cloud \
  --deadline-ms 60000 --output output/support-ollama-cloud-01.jsonl
```

For direct cloud calls, set `OLLAMA_API_KEY` in your environment and run:

```bash
python examples/support_routing.py --backend ollama-cloud-direct \
  --deadline-ms 60000 --output output/support-ollama-direct-01.jsonl
```

Both cloud entries require access to Ollama's cloud. The local entries record
zero API charges, excluding hardware and energy costs; cloud costs stay unknown
until their rates are configured. Each command sends the synthetic tickets or
tool context to the selected server. See the
[Ollama Decision guide](https://docs.ollama.com/capabilities/decision) for setup.

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
