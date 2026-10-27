"""Provider-neutral requests, results, and strict validation for Choice, Noul and Score.

A Choice picks one of the application's options, a Noul is the probability that a
yes/no condition holds, and a Score is the expected level on an ordered rubric.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, ClassVar, NoReturn, Protocol, TypeVar

VERSION = "0.4.0"
SCHEMA_VERSION = 1
QUESTION_TYPES = ("choice", "noul", "score")
MAX_SCORE_LEVELS = 10


def json_copy(value: Any) -> Any:
    """Snapshot JSON data; reject NaN, infinity, and non-JSON objects."""
    try:
        return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Expected finite JSON data") from None


def probability(value: Any) -> bool:
    return (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1)


def _instructions(value: Any):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Instructions must be nonempty text")


def _hash(packet: dict) -> str:
    encoded = json.dumps(packet, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ChoiceRequest:
    question_type: ClassVar[str] = "choice"
    state: Any
    options: dict[str, Any]
    instructions: str

    def __post_init__(self):
        if not isinstance(self.options, dict) or not 1 <= len(self.options) <= 255:
            raise ValueError("Choice requires 1–255 options")
        if any(not isinstance(key, str) or not key for key in self.options):
            raise ValueError("Option IDs must be nonempty strings")
        _instructions(self.instructions)
        object.__setattr__(self, "state", json_copy(self.state))
        object.__setattr__(self, "options", json_copy(self.options))

    def to_dict(self) -> dict:
        return json_copy({"state": self.state, "options": self.options,
                          "instructions": self.instructions})

    @property
    def packet_hash(self) -> str:
        # Preserve candidate presentation order while canonicalizing JSON objects.
        return _hash({"state": self.state, "options": list(self.options.items()),
                      "instructions": self.instructions})


@dataclass(frozen=True)
class NoulRequest:
    """A yes/no question. Optional criteria describe what "true" and "false" mean."""
    question_type: ClassVar[str] = "noul"
    state: Any
    instructions: str
    criteria: dict[str, Any] | None = None

    def __post_init__(self):
        _instructions(self.instructions)
        if self.criteria is not None and (not isinstance(self.criteria, dict) or not self.criteria
                                          or not set(self.criteria) <= {"true", "false"}
                                          or any(value is None for value in self.criteria.values())):
            raise ValueError("Noul criteria must describe true and/or false")
        object.__setattr__(self, "state", json_copy(self.state))
        object.__setattr__(self, "criteria", json_copy(self.criteria))

    def to_dict(self) -> dict:
        return json_copy({"state": self.state, "instructions": self.instructions,
                          "criteria": self.criteria})

    @property
    def packet_hash(self) -> str:
        return _hash({"type": "noul", "state": self.state, "instructions": self.instructions,
                      "criteria": self.criteria})


@dataclass(frozen=True)
class ScoreRequest:
    """Rate the state on ordered levels, lowest first. The answer is the expected level."""
    question_type: ClassVar[str] = "score"
    state: Any
    levels: list[Any]
    instructions: str

    def __post_init__(self):
        if (not isinstance(self.levels, (list, tuple))
                or not 1 <= len(self.levels) <= MAX_SCORE_LEVELS):
            raise ValueError(f"Score requires 1–{MAX_SCORE_LEVELS} levels")
        if any(level is None for level in self.levels):
            raise ValueError("Score levels must be described")
        _instructions(self.instructions)
        object.__setattr__(self, "state", json_copy(self.state))
        object.__setattr__(self, "levels", json_copy(list(self.levels)))

    def to_dict(self) -> dict:
        return json_copy({"state": self.state, "levels": self.levels,
                          "instructions": self.instructions})

    @property
    def packet_hash(self) -> str:
        return _hash({"type": "score", "state": self.state, "levels": self.levels,
                      "instructions": self.instructions})


REQUEST_TYPES = {"choice": ChoiceRequest, "noul": NoulRequest, "score": ScoreRequest}


def request_from_dict(question_type: str, data: dict):
    """Rebuild a recorded request; journals from before Noul and Score are all Choice."""
    if question_type not in REQUEST_TYPES or not isinstance(data, dict):
        raise ValueError("Unknown question type or request")
    return REQUEST_TYPES[question_type](**data)


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
    """A Choice answer."""
    choice: str
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    model: str | None = None
    usage: Usage | None = None


@dataclass(frozen=True)
class NoulResult:
    """The probability that the answer is yes. Thresholds belong to the application."""
    probability: float
    model: str | None = None
    usage: Usage | None = None


@dataclass(frozen=True)
class ScoreResult:
    """The expected level (0 is the lowest) and, when given, one probability per level."""
    score: float
    probabilities: list[float] | None = None
    confidence: float | None = None
    model: str | None = None
    usage: Usage | None = None


RESULT_TYPES = {"choice": DecisionResult, "noul": NoulResult, "score": ScoreResult}


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
    """choose() is required. noul(NoulRequest) -> NoulResult and
    score(ScoreRequest) -> ScoreResult are optional; without them those questions
    are recorded as unsupported_question_type before any request is made."""
    info: BackendInfo

    def choose(self, request: ChoiceRequest) -> DecisionResult: ...
    def close(self) -> None: ...


def answer_diagnostics(answer: Any, request) -> dict:
    """Retain useful invalid-answer evidence without dumping arbitrary server data."""
    if not isinstance(answer, dict):
        return {}
    result = {}
    if probability(answer.get("confidence")):
        result["confidence"] = answer["confidence"]
    if isinstance(request, NoulRequest):
        value = answer.get("probability", answer.get("noul"))
        if probability(value):
            result["probability"] = value
        return result
    distribution = answer.get("probabilities")
    if isinstance(request, ScoreRequest):
        score = answer.get("score")
        if type(score) in (int, float) and math.isfinite(score):
            result["score"] = score
        levels = {str(level) for level in range(len(request.levels))}
        if isinstance(distribution, list):
            distribution = {str(level): value for level, value in enumerate(distribution)}
        if isinstance(distribution, dict):
            result["probabilities"] = {k: v for k, v in distribution.items()
                                       if k in levels and probability(v)}
        return result
    choice = answer.get("choice")
    if isinstance(choice, str) and choice in request.options:
        result["choice"] = choice
    if isinstance(distribution, dict):
        result["probabilities"] = {k: v for k, v in distribution.items()
                                   if k in request.options and probability(v)}
    return result


def _tolerance(values, decimals: int | None) -> float:
    """Allowed distance from a sum of 1, wider when a provider rounds each probability."""
    tolerance = .001
    if decimals is not None:
        unit = 10 ** -decimals
        if all(abs(v - round(v, decimals)) <= 1e-8 for v in values):
            tolerance = max(tolerance, min(.05, len(values) * unit / 2))
    return tolerance


Result = TypeVar("Result", DecisionResult, NoulResult, ScoreResult)


def validate_result(request, result: Result, decimals: int | None = None) -> Result:
    """Check an answer against its request; raise DecisionError with a safe code."""
    expected = RESULT_TYPES[request.question_type]
    if not isinstance(result, expected):
        raise DecisionError("invalid_result_type", kind="validation")

    def fail(code) -> NoReturn:
        raise DecisionError(code, kind="validation",
                            usage=result.usage if isinstance(result.usage, Usage) else None,
                            model=result.model if isinstance(result.model, str) else None,
                            diagnostics=answer_diagnostics(asdict(result), request))

    if isinstance(result, DecisionResult) and (not isinstance(result.choice, str)
                                               or result.choice not in request.options):
        fail("unknown_choice")
    if result.usage is not None and not isinstance(result.usage, Usage):
        fail("invalid_usage")
    if result.model is not None and (not isinstance(result.model, str) or not result.model):
        fail("invalid_model")
    if isinstance(result, NoulResult):
        if not probability(result.probability):
            fail("invalid_probability")
        return result
    if result.confidence is not None and not probability(result.confidence):
        fail("invalid_confidence")
    if isinstance(result, ScoreResult):
        return _validate_score(request, result, decimals, fail)
    distribution = result.probabilities
    if distribution is not None:
        if not isinstance(distribution, dict) or set(distribution) != set(request.options):
            fail("option_mismatch")
        if not all(probability(v) for v in distribution.values()):
            fail("invalid_probability")
        if abs(sum(distribution.values()) - 1) > _tolerance(distribution.values(), decimals) + 1e-8:
            fail("probability_sum")
        if distribution[result.choice] + 1e-6 < max(distribution.values()):
            fail("choice_probability_mismatch")
    return result


def _validate_score(request: ScoreRequest, result: ScoreResult, decimals, fail) -> ScoreResult:
    top = len(request.levels) - 1
    score = result.score
    if type(score) not in (int, float) or not math.isfinite(score):
        fail("invalid_score")
    if not -1e-6 <= score <= top + 1e-6:
        fail("score_out_of_range")
    distribution = result.probabilities
    if distribution is None:
        return result
    if not isinstance(distribution, list) or len(distribution) != top + 1:
        fail("level_mismatch")
    if not all(probability(v) for v in distribution):
        fail("invalid_probability")
    if abs(sum(distribution) - 1) > _tolerance(distribution, decimals) + 1e-8:
        fail("probability_sum")
    # The score is the expected level. Providers round each probability and the
    # score (to two places unless configured), which moves the sum by up to half a
    # unit per level, weighted by the level.
    unit = 10 ** -(decimals or 2)
    expected = sum(level * value for level, value in enumerate(distribution))
    if abs(expected - score) > unit / 2 * (1 + top * (top + 1) / 2) + 1e-8:
        fail("score_probability_mismatch")
    return result
