"""Append-only decision and action events. The caller owns action execution."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Callable, Protocol
from uuid import uuid4

from .types import (Backend, ChoiceRequest, DecisionError, DecisionResult, SCHEMA_VERSION,
                    VERSION, answer_diagnostics, json_copy, validate_result)


class Sink(Protocol):
    def write(self, event: dict) -> None: ...


class JsonlSink:
    """Create a new local file; flush each event before returning control."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8")

    def write(self, event: dict) -> None:
        self._stream.write(json.dumps(event, allow_nan=False, ensure_ascii=False) + "\n")
        self._stream.flush()

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@dataclass(frozen=True)
class Decision:
    decision_id: str
    result: DecisionResult
    latency_ms: float
    source: str
    deadline_missed: bool
    observed_at: float
    completed_at: float
    deadline_ms: float | None
    phase: str

    def expired(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return self.deadline_ms is not None and (now - self.observed_at) * 1000 > self.deadline_ms


class Harness:
    """One sequential control loop. Use separate instances for concurrent workers."""

    def __init__(self, backend: Backend, sink: Sink, *, run_id: str | None = None,
                 record_inputs: bool = True, metadata: dict | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.backend = backend
        self.sink = sink
        self.run_id = run_id or str(uuid4())
        self.record_inputs = record_inputs
        self._clock = clock
        self._origin = clock()
        self._pending: dict[str, Decision] = {}
        self._event("run_start", backend=backend.info.to_dict(),
                    metadata=json_copy(metadata or {}), record_inputs=record_inputs)

    def _event(self, event: str, **fields):
        self.sink.write({"schema_version": SCHEMA_VERSION, "library_version": VERSION,
                         "event": event, "run_id": self.run_id,
                         "timestamp": datetime.now(timezone.utc).isoformat(),
                         "run_elapsed_ms": round((self._clock() - self._origin) * 1000, 3),
                         **fields})

    def choose(self, request: ChoiceRequest, *, phase: str = "decision",
               observation_id: str | int | None = None, observed_at: float | None = None,
               deadline_ms: float | None = None) -> Decision:
        request = ChoiceRequest(**request.to_dict())
        start = self._clock()
        observed_at = start if observed_at is None else observed_at
        if (type(observed_at) not in (int, float) or not math.isfinite(observed_at)
                or observed_at > start):
            raise ValueError("Observation time must be a past monotonic timestamp")
        if deadline_ms is not None and (type(deadline_ms) not in (int, float)
                                       or not math.isfinite(deadline_ms) or deadline_ms <= 0):
            raise ValueError("Deadline must be finite and positive")
        decision_id = str(uuid4())
        fields = {"decision_id": decision_id, "phase": phase,
                  "observation_id": observation_id, "packet_hash": request.packet_hash,
                  "candidate_order": list(request.options), "deadline_ms": deadline_ms,
                  "backend": self.backend.info.to_dict()}
        if self.record_inputs:
            fields["request"] = request.to_dict()
        source = "forced" if len(request.options) == 1 else "provider"
        started_at_unix_ns = time.time_ns()
        try:
            if source == "forced":
                result = DecisionResult(choice=next(iter(request.options)))
            else:
                result = self.backend.choose(ChoiceRequest(**request.to_dict()))
                if not isinstance(result, DecisionResult):
                    raise DecisionError("invalid_result_type", kind="validation")
                validate_result(request, result, self.backend.info.probability_decimals)
        except (Exception, KeyboardInterrupt) as error:
            finish = self._clock()
            interrupted = isinstance(error, KeyboardInterrupt)
            if not isinstance(error, DecisionError):
                error = DecisionError("interrupted" if interrupted else "unexpected_backend_error")
            error.decision_id = decision_id
            error.diagnostics = answer_diagnostics(error.diagnostics, request)
            self._event("decision", **fields, source=source, status="error",
                        started_at_unix_ns=started_at_unix_ns, completed_at_unix_ns=time.time_ns(),
                        latency_ms=round((finish - start) * 1000, 3),
                        observation_age_ms=round((finish - observed_at) * 1000, 3),
                        deadline_missed=deadline_ms is not None and (finish - observed_at) * 1000 > deadline_ms,
                        error={"kind": error.kind, "code": error.code,
                               "status_code": error.status_code},
                        request_attempted=error.request_attempted is not False,
                        usage=asdict(error.usage) if error.usage is not None else None,
                        returned_model=error.model, diagnostics=error.diagnostics,
                        estimated_api_cost_usd=0.0 if error.request_attempted is False else self._cost(error.usage))
            if interrupted:
                raise KeyboardInterrupt from None
            raise error from None
        finish = self._clock()
        latency_ms = (finish - start) * 1000
        missed = deadline_ms is not None and (finish - observed_at) * 1000 > deadline_ms
        self._event("decision", **fields, source=source, status="ok",
                    started_at_unix_ns=started_at_unix_ns, completed_at_unix_ns=time.time_ns(),
                    latency_ms=round(latency_ms, 3), deadline_missed=missed,
                    request_attempted=source == "provider",
                    observation_age_ms=round((finish - observed_at) * 1000, 3),
                    result=asdict(result), returned_model=result.model,
                    usage=asdict(result.usage) if result.usage is not None else None,
                    estimated_api_cost_usd=0.0 if source == "forced" else self._cost(result.usage))
        decision = Decision(decision_id, result, latency_ms, source, missed,
                            observed_at, finish, deadline_ms, phase)
        self._pending[decision_id] = decision
        return decision

    def _cost(self, usage):
        price = self.backend.info.pricing
        return price.cost(usage) if price is not None else None

    def record_action(self, decision: Decision, status: str, *, reason: str | None = None,
                      details: dict | None = None):
        if status not in {"applied", "rejected", "expired", "skipped", "unknown"}:
            raise ValueError("Invalid action status")
        if self._pending.get(decision.decision_id) is not decision:
            raise ValueError("Decision is unknown or already has an action result")
        now = self._clock()
        self._event("action", decision_id=decision.decision_id, status=status,
                    phase=decision.phase, backend=self.backend.info.to_dict(),
                    reason=reason, details=json_copy(details or {}),
                    observation_age_ms=round((now - decision.observed_at) * 1000, 3),
                    decision_to_action_ms=round((now - decision.completed_at) * 1000, 3),
                    deadline_missed=decision.expired(now))
        del self._pending[decision.decision_id]

    def outcome(self, outcome: str, *, metrics: dict | None = None):
        self._event("run_outcome", outcome=outcome, metrics=json_copy(metrics or {}),
                    decisions_without_action_result=len(self._pending))
