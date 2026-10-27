"""Dataset serializers; provenance, labels, and episode splits live in training.py."""
from __future__ import annotations

import importlib
from typing import Protocol

from .rendering import OPTION_RENDERING, render_options, render_value
from .types import ChoiceRequest, NoulRequest, ScoreRequest, json_copy


class TrainingExporter(Protocol):
    """Implement name and serialize; optional details are copied into the manifest.

    request is a ChoiceRequest, NoulRequest or ScoreRequest (see its question_type),
    and label holds its target under "choice", "answer" (true/false) or "level".
    """
    name: str

    def serialize(self, request: ChoiceRequest | NoulRequest | ScoreRequest, label: dict,
                  provenance: dict) -> dict: ...


class DecisionJSONLExporter:
    """Model-neutral inputs, ordered option IDs, explicit target, and provenance."""
    name = "decision-jsonl-v1"
    details = {"schema_version": 1, "options": "original_json"}

    def serialize(self, request, label, provenance):
        return {"schema_version": 1, "question_type": request.question_type,
                "request": request.to_dict(),
                "candidate_order": list(request.options) if isinstance(request, ChoiceRequest) else None,
                "label": json_copy(label), "provenance": json_copy(provenance)}


class StrandsChoiceExporter:
    """Native Example.from_dict shape, with extra provenance ignored by Strands.

    Writes all three kinds despite the name, which manifests keep. A Noul is the
    options "false" then "true" (label 0 or 1), described by its criteria where
    given; a Score is its levels lowest first, named "0", "1"… (label = level).
    """
    name = "strands-choice-jsonl"
    details = {"option_rendering": OPTION_RENDERING,
               "validated_data_loader": "strands-decider 0.1.0"}

    def serialize(self, request, label, provenance):
        if isinstance(request, NoulRequest):
            criteria = request.criteria or {}
            options = [[name, render_value(criteria[name]) if name in criteria else None]
                       for name in ("false", "true")]
            target = int(label["answer"])
        elif isinstance(request, ScoreRequest):
            options = [[str(level), render_value(text)] for level, text in enumerate(request.levels)]
            target = label["level"]
        else:
            options = [list(pair) for pair in render_options(request.options).items()]
            target = list(request.options).index(label["choice"])
        return {"kind": request.question_type, "state": request.state,
                "instructions": request.instructions, "options": options,
                "label": target, "weight": label["weight"], "task": provenance["phase"],
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
