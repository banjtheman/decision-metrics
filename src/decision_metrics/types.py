"""Provider-neutral requests, results, and strict Choice validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Protocol

VERSION = "0.1.1"
SCHEMA_VERSION = 1


def json_copy(value: Any) -> Any:
    """Snapshot JSON data; reject NaN, infinity, and non-JSON objects."""
    try:
        return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Expected finite JSON data") from None


def probability(value: Any) -> bool:
    return (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1)


@dataclass(frozen=True)
class ChoiceRequest:
    state: Any
    options: dict[str, Any]
    instructions: str

    def __post_init__(self):
        if not isinstance(self.options, dict) or not 1 <= len(self.options) <= 255:
            raise ValueError("Choice requires 1–255 options")
        if any(not isinstance(key, str) or not key for key in self.options):
            raise ValueError("Option IDs must be nonempty strings")
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("Instructions must be nonempty text")
        object.__setattr__(self, "state", json_copy(self.state))
        object.__setattr__(self, "options", json_copy(self.options))

    def to_dict(self) -> dict:
        return json_copy({"state": self.state, "options": self.options,
                          "instructions": self.instructions})

    @property
    def packet_hash(self) -> str:
        # Preserve candidate presentation order while canonicalizing JSON objects.
        packet = {"state": self.state, "options": list(self.options.items()),
                  "instructions": self.instructions}
        encoded = json.dumps(packet, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in (self.input_tokens, self.output_tokens)):
            raise ValueError("Token counts must be nonnegative integers")


def parse_usage(value: Any) -> Usage | None:
    if not isinstance(value, dict):
        return None
    try:
        return Usage(value["input_tokens"], value["output_tokens"])
    except (KeyError, ValueError):
        return None


@dataclass(frozen=True)
class Pricing:
    input_per_million_usd: float
    output_per_million_usd: float = 0.0
    source: str = "user configuration"
    checked_at: str | None = None

    def __post_init__(self):
        for value in (self.input_per_million_usd, self.output_per_million_usd):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Prices must be finite and nonnegative")

    def cost(self, usage: Usage | None) -> float | None:
        if self.input_per_million_usd == self.output_per_million_usd == 0:
            return 0.0  # Explicit zero API price; compute is accounted for separately.
        if usage is None:
            return None
        return (usage.input_tokens * self.input_per_million_usd
                + usage.output_tokens * self.output_per_million_usd) / 1_000_000


@dataclass(frozen=True)
class BackendInfo:
    name: str
    requested_model: str | None
    deployment: str
    endpoint: str | None = None
    pricing: Pricing | None = None
    details: dict[str, Any] = field(default_factory=dict)
    probability_decimals: int | None = None

    def __post_init__(self):
        if not self.name or not self.deployment:
            raise ValueError("Backend name and deployment are required")
        if self.probability_decimals is not None and self.probability_decimals not in range(1, 7):
            raise ValueError("Probability precision must be 1–6 decimal places")
        object.__setattr__(self, "details", json_copy(self.details))

    def to_dict(self) -> dict:
        return json_copy(asdict(self))


@dataclass(frozen=True)
class DecisionResult:
    choice: str
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    model: str | None = None
    usage: Usage | None = None


class DecisionError(RuntimeError):
    """Safe diagnostic codes; never include HTTP bodies, headers, or exception text."""

    def __init__(self, code: str, *, kind: str = "provider", status_code: int | None = None,
                 usage: Usage | None = None, model: str | None = None,
                 diagnostics: dict | None = None, request_attempted: bool | None = None):
        # Codes are programmer-supplied identifiers, not strings from a server.
        if not code or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in code):
            raise ValueError("Error code must be a lowercase identifier")
        super().__init__(code)
        self.code = code
        self.kind = kind
        self.status_code = status_code
        self.usage = usage
        self.model = model
        self.diagnostics = diagnostics
        self.request_attempted = False if kind == "capability" else request_attempted
        self.decision_id: str | None = None


class Backend(Protocol):
    info: BackendInfo

    def choose(self, request: ChoiceRequest) -> DecisionResult: ...
    def close(self) -> None: ...


def answer_diagnostics(answer: Any, request: ChoiceRequest) -> dict:
    """Retain useful invalid-answer evidence without dumping arbitrary server data."""
    if not isinstance(answer, dict):
        return {}
    result = {}
    choice = answer.get("choice")
    if isinstance(choice, str) and choice in request.options:
        result["choice"] = choice
    if probability(answer.get("confidence")):
        result["confidence"] = answer["confidence"]
    distribution = answer.get("probabilities")
    if isinstance(distribution, dict):
        result["probabilities"] = {k: v for k, v in distribution.items()
                                   if k in request.options and probability(v)}
    return result


def validate_result(request: ChoiceRequest, result: DecisionResult,
                    decimals: int | None = None) -> DecisionResult:
    def fail(code):
        raise DecisionError(code, kind="validation",
                            usage=result.usage if isinstance(result.usage, Usage) else None,
                            model=result.model if isinstance(result.model, str) else None,
                            diagnostics=answer_diagnostics(asdict(result), request))

    if not isinstance(result.choice, str) or result.choice not in request.options:
        fail("unknown_choice")
    if result.usage is not None and not isinstance(result.usage, Usage):
        fail("invalid_usage")
    if result.model is not None and (not isinstance(result.model, str) or not result.model):
        fail("invalid_model")
    if result.confidence is not None and not probability(result.confidence):
        fail("invalid_confidence")
    distribution = result.probabilities
    if distribution is not None:
        if not isinstance(distribution, dict) or set(distribution) != set(request.options):
            fail("option_mismatch")
        if not all(probability(v) for v in distribution.values()):
            fail("invalid_probability")
        tolerance = .001
        if decimals is not None:
            unit = 10 ** -decimals
            if all(abs(v - round(v, decimals)) <= 1e-8 for v in distribution.values()):
                tolerance = max(tolerance, min(.05, len(distribution) * unit / 2))
        if abs(sum(distribution.values()) - 1) > tolerance + 1e-8:
            fail("probability_sum")
        if distribution[result.choice] + 1e-6 < max(distribution.values()):
            fail("choice_probability_mismatch")
    return result
