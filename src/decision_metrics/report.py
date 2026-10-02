"""Summaries from recorded events, with missing usage and costs kept explicit."""
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from .types import SCHEMA_VERSION


def read_events(path: str | Path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                raise ValueError(f"Invalid JSON event on line {number}") from None
            if not isinstance(event, dict) or event.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"Unsupported event schema on line {number}")
            yield event


def _summarize(decisions, actions):
    evaluations = [event for event in decisions if event["source"] == "provider"]
    attempts = [event for event in evaluations if event["request_attempted"]]
    latencies = sorted(event["latency_ms"] for event in attempts)
    known_costs = [event["estimated_api_cost_usd"] for event in attempts
                   if event.get("estimated_api_cost_usd") is not None]
    usage = [event["usage"] for event in attempts if event.get("usage") is not None]
    action_counts = Counter(actions[event["decision_id"]]["status"] for event in decisions
                            if event["decision_id"] in actions)
    failures = Counter(event["error"]["code"] for event in evaluations if event["status"] == "error")
    return {"decisions": len(decisions), "backend_evaluations": len(evaluations), "model_requests": len(attempts),
            "forced_decisions": sum(event["source"] == "forced" for event in decisions),
            "valid_responses": sum(event["status"] == "ok" for event in attempts),
            "errors": dict(failures), "actions": dict(action_counts),
            "decisions_without_action_result": sum(event["status"] == "ok" and
                event["decision_id"] not in actions for event in decisions),
            "deadline_misses": sum(bool(event.get("deadline_missed")) for event in attempts),
            "action_deadline_misses": sum(bool(actions[event["decision_id"]].get("deadline_missed"))
                for event in decisions if event["decision_id"] in actions),
            "input_tokens": sum(value["input_tokens"] for value in usage),
            "output_tokens": sum(value["output_tokens"] for value in usage),
            "unknown_usage_requests": len(attempts) - len(usage),
            "known_estimated_api_cost_usd": sum(known_costs),
            "unknown_cost_requests": len(attempts) - len(known_costs),
            "median_latency_ms": statistics.median(latencies) if latencies else None,
            "p95_latency_ms": latencies[max(0, math.ceil(.95 * len(latencies)) - 1)] if latencies else None,
            "returned_models": sorted({event["returned_model"] for event in attempts
                                        if event.get("returned_model")})}


def summarize(events):
    decisions = {}
    actions = {}
    runs = {}
    for event in events:
        kind = event["event"]
        if kind == "decision":
            key = event["decision_id"]
            if key in decisions:
                raise ValueError("Duplicate decision ID would double-count usage")
            decisions[key] = event
        elif kind == "action":
            key = event["decision_id"]
            if key in actions:
                raise ValueError("Duplicate action result")
            actions[key] = event
        elif kind == "run_start":
            runs.setdefault(event["run_id"], {})["start"] = event
        elif kind == "run_outcome":
            runs.setdefault(event["run_id"], {})["outcome"] = event
    if any(key not in decisions for key in actions):
        raise ValueError("Action references an unknown decision")
    groups = {}
    for event in decisions.values():
        backend = event["backend"]
        key = (backend["name"], backend["deployment"], backend["requested_model"])
        groups.setdefault(key, []).append(event)
    entries = []
    for (name, deployment, model), values in groups.items():
        phases = sorted({event["phase"] for event in values})
        entries.append({"backend": name, "deployment": deployment, "requested_model": model,
                        **_summarize(values, actions),
                        "phases": {phase: _summarize([e for e in values if e["phase"] == phase], actions)
                                   for phase in phases}})
    return {"schema_version": SCHEMA_VERSION, "entries": entries,
            "runs": [{"run_id": run_id, "metadata": value.get("start", {}).get("metadata", {}),
                      "outcome": value.get("outcome", {}).get("outcome"),
                      "metrics": value.get("outcome", {}).get("metrics", {})}
                     for run_id, value in runs.items()]}
