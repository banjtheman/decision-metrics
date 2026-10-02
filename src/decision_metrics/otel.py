"""Optional OpenTelemetry sink using the application's tracer/meter providers.

These decision.* names are DecisionMetrics conventions, not official OTel
semantic conventions. Inputs and probability vectors stay in the local journal.
"""
from .types import VERSION


class OpenTelemetrySink:
    def __init__(self, local_sink, *, tracer=None, meter=None):
        try:
            from opentelemetry import metrics, trace
        except ImportError:
            raise ValueError("Install decision-metrics[otel] to enable OpenTelemetry") from None
        self.local_sink = local_sink
        self.tracer = tracer or trace.get_tracer("decision_metrics", VERSION)
        self.meter = meter or metrics.get_meter("decision_metrics", VERSION)
        self.requests = self.meter.create_counter("decision.requests", unit="1")
        self.duration = self.meter.create_histogram("decision.duration", unit="s")
        self.tokens = self.meter.create_counter("decision.tokens", unit="{token}")
        self.api_cost = self.meter.create_counter("decision.api_cost", unit="USD")
        self.actions = self.meter.create_counter("decision.actions", unit="1")
        self.observation_age = self.meter.create_histogram("decision.observation_age", unit="s")
        self.deadline_misses = self.meter.create_counter("decision.deadline_misses", unit="1")
        self.export_failures = 0

    def write(self, event):
        # The local journal is authoritative. A collector failure must not turn
        # a recorded model decision into an unrecorded game action.
        self.local_sink.write(event)
        try:
            self._export(event)
        except Exception:
            self.export_failures += 1

    def _export(self, event):
        from opentelemetry.trace import Status, StatusCode
        if event["event"] not in {"decision", "action"}:
            return
        backend = event["backend"]
        attributes = {"decision.provider": backend["name"],
                      "decision.deployment": backend["deployment"],
                      "decision.model": backend["requested_model"] or "unspecified",
                      "decision.phase": event["phase"], "decision.status": event["status"]}
        if event["event"] == "action":
            self.actions.add(1, attributes)
            self.observation_age.record(event["observation_age_ms"] / 1000, attributes)
            return
        if event["source"] != "provider":
            return
        self.requests.add(1, attributes)
        self.duration.record(event["latency_ms"] / 1000, attributes)
        if event.get("deadline_missed"):
            self.deadline_misses.add(1, attributes)
        usage = event.get("usage")
        if usage is not None:
            for direction in ("input", "output"):
                self.tokens.add(usage[direction + "_tokens"],
                                {**attributes, "decision.token_type": direction})
        if event.get("estimated_api_cost_usd") is not None:
            self.api_cost.add(event["estimated_api_cost_usd"], attributes)
        span_attributes = {**attributes, "decision.id": event["decision_id"],
                           "decision.run_id": event["run_id"],
                           "decision.packet_hash": event["packet_hash"]}
        if event.get("returned_model"):
            span_attributes["decision.returned_model"] = event["returned_model"]
        if event["status"] == "ok":
            span_attributes["decision.choice"] = event["result"]["choice"]
        span = self.tracer.start_span("decision.choose", attributes=span_attributes,
                                      start_time=event["started_at_unix_ns"])
        if event["status"] == "error":
            span.set_status(Status(StatusCode.ERROR, event["error"]["code"]))
        span.end(end_time=event["completed_at_unix_ns"])

