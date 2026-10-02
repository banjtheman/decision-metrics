"""Dataset serializers; provenance, labels, and episode splits live in training.py."""
from __future__ import annotations

import importlib
from typing import Protocol

from .rendering import OPTION_RENDERING, render_options
from .types import ChoiceRequest, json_copy


class TrainingExporter(Protocol):
    """Implement name and serialize; optional details are copied into the manifest."""
    name: str

    def serialize(self, request: ChoiceRequest, label: dict, provenance: dict) -> dict: ...


class DecisionJSONLExporter:
    """Model-neutral inputs, ordered option IDs, explicit target, and provenance."""
    name = "decision-jsonl-v1"
    details = {"schema_version": 1, "options": "original_json"}

    def serialize(self, request, label, provenance):
        return {"schema_version": 1, "request": request.to_dict(),
                "candidate_order": list(request.options), "label": json_copy(label),
                "provenance": json_copy(provenance)}


class StrandsChoiceExporter:
    """Native Example.from_dict shape, with extra provenance ignored by Strands."""
    name = "strands-choice-jsonl"
    details = {"option_rendering": OPTION_RENDERING,
               "validated_data_loader": "strands-decider 0.1.0"}

    def serialize(self, request, label, provenance):
        options = list(render_options(request.options).items())
        return {"kind": "choice", "state": request.state,
                "instructions": request.instructions, "options": [list(pair) for pair in options],
                "label": list(request.options).index(label["choice"]),
                "weight": label["weight"], "task": provenance["phase"],
                "provenance": json_copy(provenance)}


EXPORTERS = {"decision-jsonl": DecisionJSONLExporter, "strands": StrandsChoiceExporter}


def load_exporter(name: str) -> TrainingExporter:
    """Built-in name or trusted Python module:attribute (instance or no-arg class)."""
    if name in EXPORTERS:
        return EXPORTERS[name]()
    module, separator, attribute = name.partition(":")
    if not separator or not module or not attribute.isidentifier():
        raise ValueError("Exporter must be a built-in name or module:attribute")
    try:
        exported = getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError):
        raise ValueError("Could not import the requested training exporter") from None
    return exported() if isinstance(exported, type) else exported
