import json
from pathlib import Path
import tempfile
import unittest

from decision_metrics import (BackendInfo, ChoiceRequest, DecisionError, DecisionResult,
                              Harness, JsonlSink, Pricing, Usage)
from decision_metrics.providers import ChoiceHTTPBackend, make_backend
from decision_metrics.report import read_events, summarize
from decision_metrics.types import validate_result


class MemorySink:
    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(json.loads(json.dumps(event, allow_nan=False)))


class FakeBackend:
    def __init__(self, results, info=None, callback=None):
        self.info = info or BackendInfo("test", "test-v1", "local-test", pricing=Pricing(.1))
        self.results = iter(results)
        self.calls = 0
        self.callback = callback

    def choose(self, request):
        self.calls += 1
        if self.callback:
            self.callback(request)
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        return result

    def close(self):
        pass


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.payload = None

    def post(self, payload):
        self.payload = payload
        return self.response

    def close(self):
        pass


def request():
    return ChoiceRequest({"health": 10}, {"a": {"distance": 20}, "b": {"distance": 5}}, "Survive.")


def native_response(choice="a", probabilities=None):
    return {"model": "test-v1", "usage": {"input_tokens": 100, "output_tokens": 5},
            "answers": {"action": {"type": "choice", "choice": choice, "confidence": .8,
                                    "probabilities": probabilities or {"a": .8, "b": .2}}}}


class ChoiceTests(unittest.TestCase):
    def test_request_snapshots_json_and_hash_preserves_option_order(self):
        state = {"nested": [1]}
        packet = ChoiceRequest(state, {"a": {}, "b": {}}, "Choose.")
        state["nested"].append(2)
        self.assertEqual(packet.state, {"nested": [1]})
        reversed_packet = ChoiceRequest(packet.state, {"b": {}, "a": {}}, "Choose.")
        self.assertNotEqual(packet.packet_hash, reversed_packet.packet_hash)
        self.assertEqual(packet.packet_hash, ChoiceRequest(**packet.to_dict()).packet_hash)

    def test_invalid_json_and_option_ids_fail_before_network(self):
        for options in ({}, {1: {}}, {"": {}}, {str(i): {} for i in range(256)}):
            with self.subTest(options_count=len(options)), self.assertRaises(ValueError):
                ChoiceRequest({}, options, "Choose.")
        with self.assertRaises(ValueError):
            ChoiceRequest({"x": float("nan")}, {"a": {}}, "Choose.")

    def test_no_probabilities_are_invented_for_custom_backends(self):
        result = validate_result(request(), DecisionResult("a"))
        self.assertIsNone(result.probabilities)
        self.assertIsNone(result.confidence)
        self.assertIsNone(result.usage)

    def test_distribution_must_cover_exact_options(self):
        for distribution in ({"a": 1}, {"a": .8, "b": .1, "unknown": .1}):
            with self.assertRaises(DecisionError) as caught:
                validate_result(request(), DecisionResult("a", distribution, usage=Usage(25, 0)))
            self.assertEqual(caught.exception.code, "option_mismatch")
            self.assertEqual(caught.exception.usage, Usage(25, 0))

    def test_invalid_numbers_and_confidence_are_rejected(self):
        for value in (True, -1, 1.1, float("nan"), float("inf"), "0.8"):
            with self.subTest(value=value), self.assertRaises(DecisionError):
                validate_result(request(), DecisionResult("a", {"a": value, "b": .2}))
        with self.assertRaises(DecisionError):
            validate_result(request(), DecisionResult("a", confidence=True))
        with self.assertRaises(ValueError):
            Usage(True, 0)

    def test_rounded_probabilities_require_declared_precision(self):
        packet = ChoiceRequest({}, {"a": {}, "b": {}, "c": {}}, "Choose.")
        result = DecisionResult("a", {"a": .34, "b": .33, "c": .34})
        with self.assertRaises(DecisionError):
            validate_result(packet, result)
        self.assertIs(validate_result(packet, result, 2), result)
        self.assertEqual(result.probabilities["c"], .34)  # Never normalize raw evidence.
        with self.assertRaises(DecisionError):
            validate_result(packet, DecisionResult("a", {"a": .4, "b": .4, "c": .4}), 2)

    def test_provider_choice_mismatch_is_not_overridden_with_argmax(self):
        response = native_response("b")
        transport = FakeTransport(response)
        backend = ChoiceHTTPBackend(BackendInfo("test", "test-v1", "test"), transport=transport)
        with self.assertRaises(DecisionError) as caught:
            backend.choose(request())
        self.assertEqual(caught.exception.code, "choice_probability_mismatch")
        self.assertEqual(caught.exception.diagnostics["choice"], "b")
        self.assertEqual(caught.exception.usage, Usage(100, 5))

    def test_cloudflare_envelope_and_structured_candidates(self):
        backend = make_backend({"provider": "clef-flash", "account_id": "account123"},
                               environ={"CLOUDFLARE_AUTH_TOKEN": "TEST_ONLY_SECRET"})
        transport = FakeTransport({"success": True, "result": native_response()})
        backend._transport = transport
        self.assertEqual(backend.choose(request()).choice, "a")
        self.assertEqual(transport.payload["model"], "clef-flash")
        self.assertEqual(json.loads(transport.payload["questions"]["action"]["criteria"]["a"]), {"distance": 20})
        self.assertIn("/@cf/cloudflare/clef-flash", backend.info.endpoint)
        self.assertNotIn("TEST_ONLY_SECRET", json.dumps(backend.info.to_dict()))

    def test_malformed_usage_is_unknown_and_not_zero(self):
        response = native_response()
        response["usage"]["input_tokens"] = True
        backend = ChoiceHTTPBackend(BackendInfo("test", None, "test"), transport=FakeTransport(response))
        with self.assertRaises(DecisionError) as caught:
            backend.choose(request())
        self.assertEqual(caught.exception.code, "invalid_usage")
        self.assertIsNone(caught.exception.usage)

    def test_capability_rejection_never_sends_a_request(self):
        transport = FakeTransport(native_response())
        backend = ChoiceHTTPBackend(BackendInfo("meragpt", "sd-1", "test"),
                                    transport=transport, max_options=10)
        sink = MemorySink()
        harness = Harness(backend, sink)
        packet = ChoiceRequest({}, {str(i): {} for i in range(12)}, "Choose.")
        with self.assertRaises(DecisionError):
            harness.choose(packet)
        self.assertIsNone(transport.payload)
        result = summarize(sink.events)["entries"][0]
        self.assertEqual(result["model_requests"], 0)
        self.assertEqual(result["unknown_usage_requests"], 0)
        self.assertEqual(result["forced_decisions"], 0)
        self.assertEqual(result["errors"], {"unsupported_option_count": 1})


class HarnessTests(unittest.TestCase):
    def test_invalid_answer_keeps_paid_usage_and_safe_diagnostics(self):
        response = native_response("b")
        response["answers"]["action"]["debug"] = "TEST_ONLY_SECRET"
        backend = ChoiceHTTPBackend(BackendInfo("test", "test-v1", "test", pricing=Pricing(.1)),
                                    transport=FakeTransport(response))
        sink = MemorySink()
        harness = Harness(backend, sink)
        with self.assertRaises(DecisionError) as caught:
            harness.choose(request(), phase="shop")
        event = sink.events[-1]
        self.assertEqual(event["decision_id"], caught.exception.decision_id)
        self.assertEqual(event["usage"]["input_tokens"], 100)
        self.assertNotIn("TEST_ONLY_SECRET", json.dumps(sink.events))
        summary = summarize(sink.events)["entries"][0]
        self.assertEqual(summary["input_tokens"], 100)
        self.assertAlmostEqual(summary["known_estimated_api_cost_usd"], .00001)
        self.assertEqual(summary["phases"]["shop"]["errors"], {"choice_probability_mismatch": 1})

    def test_each_decision_retains_its_own_usage_and_model(self):
        backend = FakeBackend([DecisionResult("a", model="v1", usage=Usage(100, 0)),
                               DecisionResult("b", model="v2", usage=Usage(300, 5))])
        sink = MemorySink()
        harness = Harness(backend, sink)
        one = harness.choose(request())
        two = harness.choose(request())
        harness.record_action(two, "rejected")
        harness.record_action(one, "applied")
        values = [event for event in sink.events if event["event"] == "decision"]
        self.assertEqual([event["returned_model"] for event in values], ["v1", "v2"])
        self.assertEqual([event["usage"]["input_tokens"] for event in values], [100, 300])
        summary = summarize(sink.events)["entries"][0]
        self.assertEqual(summary["input_tokens"], 400)
        self.assertEqual(summary["actions"], {"applied": 1, "rejected": 1})

    def test_one_candidate_is_forced_without_model_call_or_fabricated_usage(self):
        backend = FakeBackend([])
        sink = MemorySink()
        harness = Harness(backend, sink)
        ticket = harness.choose(ChoiceRequest({}, {"continue": {}}, "Continue."))
        self.assertEqual(backend.calls, 0)
        self.assertEqual(ticket.source, "forced")
        self.assertIsNone(ticket.result.usage)
        harness.record_action(ticket, "applied")
        summary = summarize(sink.events)["entries"][0]
        self.assertEqual(summary["model_requests"], 0)
        self.assertEqual(summary["forced_decisions"], 1)
        self.assertEqual(summary["unknown_usage_requests"], 0)

    def test_late_valid_decision_and_action_are_separate(self):
        clock = [10.0]
        backend = FakeBackend([DecisionResult("a")], callback=lambda _: clock.__setitem__(0, 10.9))
        sink = MemorySink()
        harness = Harness(backend, sink, clock=lambda: clock[0])
        ticket = harness.choose(request(), observed_at=10, deadline_ms=850)
        self.assertTrue(ticket.deadline_missed)
        harness.record_action(ticket, "expired", reason="combat_deadline")
        with self.assertRaises(ValueError):
            harness.record_action(ticket, "applied")
        summary = summarize(sink.events)["entries"][0]
        self.assertEqual(summary["valid_responses"], 1)
        self.assertEqual(summary["deadline_misses"], 1)
        self.assertEqual(summary["actions"], {"expired": 1})
        self.assertAlmostEqual(summary["median_latency_ms"], 900)

    def test_backend_cannot_mutate_frozen_packet_to_add_an_action(self):
        def mutate(packet):
            packet.options["invented"] = {}
        backend = FakeBackend([DecisionResult("invented")], callback=mutate)
        sink = MemorySink()
        with self.assertRaises(DecisionError):
            Harness(backend, sink).choose(request())
        self.assertEqual(sink.events[-1]["candidate_order"], ["a", "b"])
        self.assertNotIn("invented", sink.events[-1]["request"]["options"])

    def test_unexpected_exception_text_never_enters_journal(self):
        sink = MemorySink()
        with self.assertRaises(DecisionError) as caught:
            Harness(FakeBackend([RuntimeError("TEST_ONLY_SECRET")]), sink).choose(request())
        self.assertEqual(caught.exception.code, "unexpected_backend_error")
        self.assertNotIn("TEST_ONLY_SECRET", json.dumps(sink.events))
        self.assertEqual(summarize(sink.events)["entries"][0]["unknown_usage_requests"], 1)

    def test_record_inputs_can_be_disabled_and_missing_cost_stays_explicit(self):
        sink = MemorySink()
        harness = Harness(FakeBackend([DecisionResult("a")]), sink, record_inputs=False)
        harness.choose(request())
        self.assertNotIn("request", sink.events[-1])
        summary = summarize(sink.events)["entries"][0]
        self.assertEqual(summary["unknown_cost_requests"], 1)
        self.assertEqual(summary["decisions_without_action_result"], 1)

    def test_jsonl_is_exclusive_and_duplicate_events_cannot_double_bill(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with JsonlSink(path) as sink:
                Harness(FakeBackend([DecisionResult("a", usage=Usage(100, 0))]), sink).choose(request())
            with self.assertRaises(FileExistsError):
                JsonlSink(path)
            events = list(read_events(path))
            with self.assertRaises(ValueError):
                summarize(events + [events[-1]])

    def test_local_and_remote_deployments_have_separate_entries(self):
        sink = MemorySink()
        for deployment in ("local-mps", "remote-gpu"):
            backend = FakeBackend([DecisionResult("a")], info=BackendInfo("strands", "v19", deployment))
            Harness(backend, sink).choose(request())
        summary = summarize(sink.events)
        self.assertEqual({entry["deployment"] for entry in summary["entries"]}, {"local-mps", "remote-gpu"})

    def test_keyboard_interrupt_is_recorded_with_unknown_usage_and_still_stops(self):
        sink = MemorySink()
        with self.assertRaises(KeyboardInterrupt):
            Harness(FakeBackend([KeyboardInterrupt()]), sink).choose(request())
        self.assertEqual(sink.events[-1]["error"]["code"], "interrupted")
        self.assertEqual(summarize(sink.events)["entries"][0]["unknown_usage_requests"], 1)


if __name__ == "__main__":
    unittest.main()
