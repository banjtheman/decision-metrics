"""Shared CLI setup for synthetic application examples."""
import argparse
import json
import math
from pathlib import Path

from decision_metrics import BackendInfo, DecisionResult, Pricing
from decision_metrics.providers import load_backend
from decision_metrics.report import read_events, summarize


class DemoBackend:
    """Deterministic local rules, explicitly labelled as a simulated backend."""

    info = BackendInfo(
        "offline-demo", "deterministic-rules", "local-example",
        pricing=Pricing(0, source="Offline example; no model API calls"),
        details={"simulated": True},
    )

    def __init__(self, select):
        self.select = select

    def choose(self, request):
        # These rules provide no model probabilities or token usage.
        return DecisionResult(self.select(request), model=self.info.requested_model)

    def close(self):
        pass


def configure(description, select):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--backend", help="Configured model name; omit for offline rules")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("providers.toml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-ms", type=float, default=1000)
    args = parser.parse_args()
    if not math.isfinite(args.deadline_ms) or args.deadline_ms <= 0:
        parser.error("deadline-ms must be finite and positive")
    backend = load_backend(args.config, args.backend) if args.backend else DemoBackend(select)
    return args, backend


def print_results(output, results):
    print(json.dumps({"results": results, "telemetry_file": str(output),
                      "summary": summarize(read_events(output))}, indent=2))
