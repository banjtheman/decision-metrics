"""Turn decision journals and explicit labels into auditable training datasets.

No model dependency, optimizer, inferred optimal-action label, or random row split.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from .exporters import DecisionJSONLExporter, TrainingExporter
from .types import ChoiceRequest, SCHEMA_VERSION, VERSION, json_copy

SPLITS = ("train", "validation", "test")
LABEL_SOURCES = {"human", "teacher", "policy", "self"}


def read_annotations(path: str | Path):
    """Read edited review rows or {run_id, decision_id, label} annotation rows."""
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError
                yield json_copy(row)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid annotation on line {number}") from None


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _label(value, request, allow_self_labels):
    if not isinstance(value, dict) or set(value) - {"choice", "source", "labeler", "weight", "details"}:
        raise ValueError("Label requires choice, source, labeler, and optional weight/details")
    choice = value.get("choice")
    if not isinstance(choice, str) or choice not in request.options:
        raise ValueError("Label choice is not an available option")
    source = value.get("source")
    if not isinstance(source, str) or source not in LABEL_SOURCES:
        raise ValueError("Label source must be human, teacher, policy, or self")
    if source == "self" and not allow_self_labels:
        raise ValueError("Self-labels require allow_self_labels=True / --allow-self-labels")
    labeler = _identifier(value.get("labeler"), "Labeler")
    weight = value.get("weight", 1.0)
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("Label weight must be finite and positive")
    details = value.get("details", {})
    if not isinstance(details, dict):
        raise ValueError("Label details must be an object")
    return {"choice": choice, "source": source, "labeler": labeler,
            "weight": weight, "details": json_copy(details)}


def export_training(events, output: str | Path, *, annotations=(),
                    teacher_backend: str | None = None, splits: dict | None = None,
                    require_applied: bool = False, allow_self_labels: bool = False,
                    exporter: TrainingExporter | None = None) -> dict:
    """Write review.jsonl, three model-specific data files, and a hashed manifest.

    Labels are absent by default. ``teacher_backend`` explicitly labels eligible
    responses from that backend as teacher imitation; annotations take precedence.
    ``splits`` maps whole episode groups to train/validation/test. A run's
    metadata.episode_id defines its group, falling back to run_id. Without a map,
    every group goes to train and the manifest marks evaluation as unavailable.
    The default exporter preserves original JSON inputs. Use StrandsChoiceExporter
    or implement TrainingExporter for a different training runtime.
    """
    exporter = DecisionJSONLExporter() if exporter is None else exporter
    _identifier(getattr(exporter, "name", None), "Exporter name")
    if not callable(getattr(exporter, "serialize", None)):
        raise ValueError("Exporter must implement serialize(request, label, provenance)")
    exporter_details = json_copy(getattr(exporter, "details", {}))
    if not isinstance(exporter_details, dict):
        raise ValueError("Exporter details must be an object")
    events = json_copy(list(events))
    decisions, actions, starts, outcomes = {}, {}, {}, {}
    run_ids = set()
    seen_decision_ids = set()
    for event in events:
        if not isinstance(event, dict) or event.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported event schema")
        run_id = _identifier(event.get("run_id"), "Run ID")
        run_ids.add(run_id)
        kind = event.get("event")
        if kind in {"decision", "action"}:
            decision_id = _identifier(event.get("decision_id"), "Decision ID")
            key = (run_id, decision_id)
            store = decisions if kind == "decision" else actions
            if key in store or (kind == "decision" and decision_id in seen_decision_ids):
                raise ValueError(f"Duplicate {kind} ID")
            store[key] = event
            if kind == "decision":
                seen_decision_ids.add(decision_id)
        elif kind in {"run_start", "run_outcome"}:
            store = starts if kind == "run_start" else outcomes
            if run_id in store:
                raise ValueError(f"Duplicate {kind} for run")
            store[run_id] = event
    if any(key not in decisions for key in actions):
        raise ValueError("Action references an unknown decision/run")
    if not decisions:
        raise ValueError("No decisions to export")

    groups = {}
    for run_id in sorted(run_ids):
        metadata = starts.get(run_id, {}).get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Run metadata must be an object")
        groups[run_id] = _identifier(metadata.get("episode_id", run_id), "Episode group")
    group_ids = set(groups.values())
    if splits is None:
        split_map = dict.fromkeys(sorted(group_ids), "train")
    else:
        if not isinstance(splits, dict) or set(splits) != group_ids:
            raise ValueError("Split map must cover exactly the source episode groups")
        if any(value not in SPLITS for value in splits.values()):
            raise ValueError("Split must be train, validation, or test")
        split_map = dict(splits)

    labels = {}
    for row in annotations:
        row = json_copy(row)
        if not isinstance(row, dict):
            raise ValueError("Annotation must be an object")
        key = (_identifier(row.get("run_id"), "Annotation run ID"),
               _identifier(row.get("decision_id"), "Annotation decision ID"))
        if key not in decisions:
            raise ValueError("Annotation references an unknown decision/run")
        if row.get("packet_hash") is not None and row["packet_hash"] != decisions[key].get("packet_hash"):
            raise ValueError("Annotation packet hash does not match its decision")
        if row.get("label") is None:
            continue  # A review row that has not been annotated yet.
        if key in labels:
            raise ValueError("Duplicate annotation")
        labels[key] = row["label"]
    if teacher_backend is not None:
        _identifier(teacher_backend, "Teacher backend")
        if not any(e.get("backend", {}).get("name") == teacher_backend for e in decisions.values()):
            raise ValueError("Teacher backend does not occur in these journals")

    review, datasets = [], {name: [] for name in SPLITS}
    excluded, sources, phases = Counter(), Counter(), Counter()
    packet_splits, packet_labels = {}, {}
    eligible_count = 0
    for key, event in decisions.items():
        run_id, decision_id = key
        group_id = groups[run_id]
        split = split_map[group_id]
        action = actions.get(key)
        request_data = event.get("request")
        phase = _identifier(event.get("phase", "decision"), "Phase")
        request = None
        if request_data is not None:
            try:
                request = ChoiceRequest(**request_data)
            except (ValueError, TypeError):
                raise ValueError("Invalid recorded request") from None
            if event.get("packet_hash") != request.packet_hash:
                raise ValueError("Recorded request does not match its packet hash")
            if event.get("candidate_order") != list(request.options):
                raise ValueError("Recorded candidate order does not match the request")
            prior = packet_splits.setdefault(request.packet_hash, split)
            if prior != split:
                raise ValueError("Identical decision packet crosses dataset splits")
        reason = None
        if request is None:
            reason = "inputs_not_recorded"
        elif len(request.options) < 2 or event.get("source") != "provider":
            reason = "forced_or_nonprovider"
        elif event.get("status") != "ok":
            reason = "decision_error"
        elif not isinstance(event.get("result"), dict) or not isinstance(event["result"].get("choice"), str) or event["result"]["choice"] not in request.options:
            raise ValueError("Recorded choice is not an available option")
        elif event.get("deadline_missed"):
            reason = "decision_deadline_missed"
        elif action is None:
            reason = "action_missing"
        elif action.get("deadline_missed"):
            reason = "action_deadline_missed"
        elif action.get("status") not in ({"applied"} if require_applied else {"applied", "skipped"}):
            reason = "action_" + str(action.get("status", "invalid"))

        label = None
        if reason is not None:
            excluded[reason] += 1
            if key in labels:
                raise ValueError(f"Annotation targets an ineligible decision: {reason}")
        else:
            eligible_count += 1
            if key in labels:
                label = _label(labels[key], request, allow_self_labels)
            elif teacher_backend is not None and event.get("backend", {}).get("name") == teacher_backend:
                backend = event["backend"]
                label = _label({"choice": event["result"]["choice"], "source": "teacher",
                    "labeler": event.get("returned_model") or backend.get("requested_model") or teacher_backend,
                    "details": {"backend": teacher_backend, "method": "recorded_provider_choice"}},
                    request, allow_self_labels)

        provenance = {"run_id": run_id, "decision_id": decision_id, "episode_id": group_id,
            "split": split, "phase": phase, "request": request_data,
            "observation_id": event.get("observation_id"), "packet_hash": event.get("packet_hash"),
            "backend": event.get("backend"), "model_result": event.get("result"),
            "action": action, "run_outcome": outcomes.get(run_id),
            "run_metadata": starts.get(run_id, {}).get("metadata", {}),
            "decision_deadline_missed": event.get("deadline_missed"),
            "latency_ms": event.get("latency_ms"), "observation_age_ms": event.get("observation_age_ms"),
            "source_library_version": event.get("library_version"),
            "source_option_rendering": event.get("backend", {}).get("details", {}).get("option_rendering"),
            "label": label}
        review.append({"run_id": run_id, "decision_id": decision_id,
            "packet_hash": event.get("packet_hash"), "eligible": reason is None,
            "excluded_reason": reason, "request": request_data, "label": label,
            "provenance": provenance})
        if label is None:
            continue
        prior = packet_labels.setdefault(request.packet_hash, label["choice"])
        if prior != label["choice"]:
            raise ValueError("Conflicting labels for an identical decision packet")
        phases[provenance["phase"]] += 1
        sources[label["source"]] += 1
        serialized = json_copy(exporter.serialize(ChoiceRequest(**request.to_dict()),
                                                 json_copy(label), json_copy(provenance)))
        if not isinstance(serialized, dict):
            raise ValueError("Exporter must return a finite JSON object")
        datasets[split].append(serialized)

    counts = {name: len(rows) for name, rows in datasets.items()}
    notes = ["Teacher agreement and episode outcomes do not establish an optimal-action label."]
    if not any(counts[name] for name in ("validation", "test")):
        notes.append("No held-out labeled episodes: this export cannot measure generalization.")
    if counts["train"] == 0:
        notes.append("No labeled training examples yet; annotate review.jsonl and export again.")
    manifest = {"dataset_schema_version": 1, "library_version": VERSION,
        "format": exporter.name, "exporter_details": exporter_details,
        "source_event_count": len(events), "decisions": len(decisions),
        "source_events_sha256": hashlib.sha256(json.dumps(events, sort_keys=True,
            ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest(),
        "eligible_decisions": eligible_count, "unlabeled_eligible_decisions": eligible_count - sum(counts.values()),
        "excluded": dict(excluded), "examples": counts, "label_sources": dict(sources),
        "phases": dict(phases), "run_to_episode": groups, "episode_splits": split_map,
        "require_applied": require_applied, "allow_self_labels": allow_self_labels,
        "teacher_backend": teacher_backend,
        "evaluation_ready": bool(counts["train"] and (counts["validation"] or counts["test"])),
        "notes": notes, "files": {}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in {"review": review, **datasets}.items():
        encoded = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows).encode("utf-8")
        path = output / f"{name}.jsonl"
        with path.open("xb") as stream:
            stream.write(encoded)
        manifest["files"][path.name] = {"sha256": hashlib.sha256(encoded).hexdigest(), "rows": len(rows)}
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return manifest
