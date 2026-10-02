"""Small offline demo, sequential replay, and JSONL summary CLI."""
import argparse
import json
import math
from pathlib import Path
import sys

from . import BackendInfo, ChoiceRequest, DecisionError, DecisionResult, Harness, JsonlSink
from .providers import PROVIDERS, load_backend
from .report import read_events, summarize
from .training import export_training, read_annotations
from .exporters import EXPORTERS, load_exporter


class DemoBackend:
    info = BackendInfo("demo", "deterministic-demo", "offline")

    def choose(self, request):
        return DecisionResult(next(iter(request.options)), model="deterministic-demo")

    def close(self):
        pass


def packets(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                request = ChoiceRequest(value["state"], value["options"], value["instructions"])
            except (ValueError, KeyError, TypeError):
                raise ValueError(f"Invalid decision packet on line {number}") from None
            yield request, value.get("phase", "decision"), value.get("case_id", number)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Standalone decision telemetry and replay")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("providers", help="List configured transport types; no API calls")
    demo = commands.add_parser("demo", help="Write synthetic offline events; no game/API calls")
    demo.add_argument("--output", type=Path, default=Path("output/demo.jsonl"))
    summary = commands.add_parser("summary", help="Summarize one or more event files")
    summary.add_argument("events", type=Path, nargs="+")
    training = commands.add_parser("export-training", help="Create a review queue or labeled training dataset")
    training.add_argument("events", type=Path, nargs="+")
    training.add_argument("--output", type=Path, required=True, help="New dataset directory")
    training.add_argument("--labels", type=Path, help="Edited review JSONL or explicit annotation JSONL")
    training.add_argument("--teacher-backend", help="Explicitly imitate recorded choices from this named backend")
    training.add_argument("--splits", type=Path, help="JSON map of episode group IDs to train/validation/test")
    training.add_argument("--require-applied", action="store_true", help="Exclude offline replay and other skipped actions")
    training.add_argument("--allow-self-labels", action="store_true")
    formats = training.add_mutually_exclusive_group()
    formats.add_argument("--format", choices=sorted(EXPORTERS), default=None,
                         help="Dataset format (default: decision-jsonl)")
    formats.add_argument("--exporter", help="Custom trusted Python serializer, module:attribute")
    replay = commands.add_parser("replay", help="Send frozen packets sequentially to one backend")
    replay.add_argument("packets", type=Path)
    replay.add_argument("--config", type=Path, required=True)
    replay.add_argument("--backend", required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--limit", type=int, default=20)
    replay.add_argument("--max-cost", type=float, default=.50,
                        help="Stop after known API cost reaches this USD estimate")
    args = parser.parse_args(argv)
    if args.command == "replay" and (args.limit <= 0 or not math.isfinite(args.max_cost) or args.max_cost <= 0):
        parser.error("limit and max-cost must be positive and finite")
    try:
        if args.command == "providers":
            print(json.dumps(PROVIDERS, indent=2))
            return 0
        if args.command == "summary":
            print(json.dumps(summarize(event for path in args.events for event in read_events(path)), indent=2))
            return 0
        if args.command == "export-training":
            splits = json.loads(args.splits.read_text(encoding="utf-8")) if args.splits else None
            result = export_training((event for path in args.events for event in read_events(path)), args.output,
                annotations=read_annotations(args.labels) if args.labels else (),
                teacher_backend=args.teacher_backend, splits=splits,
                require_applied=args.require_applied, allow_self_labels=args.allow_self_labels,
                exporter=load_exporter(args.exporter or args.format or "decision-jsonl"))
            print(json.dumps(result, indent=2))
            return 0
        backend = DemoBackend() if args.command == "demo" else load_backend(args.config, args.backend)
        failures = 0
        known_cost = 0.0
        stopped_for_cost = False
        try:
            with JsonlSink(args.output) as sink:
                harness = Harness(backend, sink, metadata={"mode": args.command,
                    "note": "Replay has no ground-truth action labels or live gameplay score."})
                cases = [(ChoiceRequest({"example": "synthetic"}, {"left": {}, "right": {}},
                                        "Choose one option."), "demo", "synthetic-1")] if args.command == "demo" else packets(args.packets)
                for count, (request, phase, case_id) in enumerate(cases):
                    if args.command == "replay" and count >= args.limit:
                        break
                    if args.command == "replay" and known_cost >= args.max_cost:
                        stopped_for_cost = True
                        break
                    try:
                        decision = harness.choose(request, phase=phase, observation_id=case_id)
                        harness.record_action(decision, "skipped", reason="offline_replay")
                        usage = decision.result.usage
                    except DecisionError as error:
                        failures += 1
                        usage = error.usage
                    price = backend.info.pricing
                    if price:
                        known_cost += price.cost(usage) or 0.0
                harness.outcome("cost_limit" if stopped_for_cost else "replay_complete",
                                metrics={"failed_requests": failures, "known_api_cost_usd": known_cost})
        finally:
            backend.close()
        print(json.dumps(summarize(read_events(args.output)), indent=2))
        return 1 if failures else 0
    except (OSError, ValueError, DecisionError) as error:
        # Configuration/file errors are local; server messages are never used.
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
