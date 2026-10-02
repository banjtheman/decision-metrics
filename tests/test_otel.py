import json
import unittest

try:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
except ImportError:
    MeterProvider = None

from decision_metrics import ChoiceRequest, DecisionError, DecisionResult, Harness, Usage
from decision_metrics.otel import OpenTelemetrySink
from test_decisions import FakeBackend, MemorySink


@unittest.skipIf(MeterProvider is None, "Optional OpenTelemetry SDK is not installed")
class OpenTelemetryTests(unittest.TestCase):
    def test_real_sdk_receives_spans_metrics_and_safe_error_codes(self):
        exporter = InMemorySpanExporter()
        traces = TracerProvider()
        traces.add_span_processor(SimpleSpanProcessor(exporter))
        reader = InMemoryMetricReader()
        meters = MeterProvider(metric_readers=[reader])
        try:
            local = MemorySink()
            sink = OpenTelemetrySink(local, tracer=traces.get_tracer("test"), meter=meters.get_meter("test"))
            backend = FakeBackend([DecisionResult("a", usage=Usage(100, 5)),
                                   RuntimeError("TEST_ONLY_SECRET")])
            harness = Harness(backend, sink)
            packet = ChoiceRequest({"private_state": "LOCAL_ONLY"}, {"a": {}, "b": {}}, "Choose.")
            ticket = harness.choose(packet, phase="combat")
            harness.record_action(ticket, "applied")
            with self.assertRaises(DecisionError):
                harness.choose(packet, phase="shop")
            spans = exporter.get_finished_spans()
            self.assertEqual(len(spans), 2)
            self.assertLess(spans[0].start_time, spans[0].end_time)
            self.assertEqual(spans[1].status.description, "unexpected_backend_error")
            self.assertNotIn("LOCAL_ONLY", str([dict(span.attributes) for span in spans]))
            self.assertNotIn("TEST_ONLY_SECRET", str(spans))
            data = reader.get_metrics_data()
            metrics = {metric.name: metric for resource in data.resource_metrics
                       for scope in resource.scope_metrics for metric in scope.metrics}
            self.assertEqual(sum(point.value for point in metrics["decision.requests"].data.data_points), 2)
            self.assertEqual(sum(point.value for point in metrics["decision.tokens"].data.data_points), 105)
            self.assertEqual(sum(point.value for point in metrics["decision.actions"].data.data_points), 1)
            for metric in metrics.values():
                for point in metric.data.data_points:
                    self.assertNotIn("decision.id", point.attributes)
                    self.assertNotIn("decision.packet_hash", point.attributes)
            self.assertEqual(sink.export_failures, 0)
            self.assertIn("LOCAL_ONLY", json.dumps(local.events))
        finally:
            traces.shutdown()
            meters.shutdown()

    def test_export_failure_does_not_destroy_local_journal(self):
        class BrokenTracer:
            def start_span(self, *args, **kwargs):
                raise RuntimeError("collector unavailable")
        local = MemorySink()
        meters = MeterProvider()
        try:
            sink = OpenTelemetrySink(local, tracer=BrokenTracer(), meter=meters.get_meter("test"))
            harness = Harness(FakeBackend([DecisionResult("a")]), sink)
            harness.choose(ChoiceRequest({}, {"a": {}, "b": {}}, "Choose."))
            self.assertEqual(sink.export_failures, 1)
            self.assertEqual(local.events[-1]["status"], "ok")
        finally:
            meters.shutdown()


if __name__ == "__main__":
    unittest.main()
