"""Google Decisions REST integration; synthetic fixtures, no vendor calls."""
from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from decision_metrics import ChoiceRequest, DecisionError, Harness, NoulRequest, ScoreRequest, Usage
from decision_metrics.providers import make_backend
from test_decisions import MemorySink


KEY = "TEST_ONLY_GOOGLE_KEY"
OPTIONS = {"technical": {"handles": ["software errors"]}, "billing": "Payments and refunds"}
CHOICE = {"type": "choice", "choice": "billing", "confidence": .9,
          "probabilities": {"billing": .9, "technical": .1}}
USAGE = {"total_input_tokens": 144, "total_output_tokens": 0, "total_tokens": 144,
         "input_tokens_by_modality": [{"modality": "text", "tokens": 144}]}


def response(answer=CHOICE):
    return {"answers": {"action": deepcopy(answer)}, "usage": deepcopy(USAGE)}


def backend(**settings):
    return make_backend({"provider": "google-decisions", **settings}, environ={"GEMINI_API_KEY": KEY})


def request(state="I was charged twice."):
    return ChoiceRequest(state, OPTIONS, "Which queue should handle this ticket?")


@contextmanager
def server(responses):
    captured = []
    replies = iter(responses)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            captured.append({"path": self.path,
                "body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                "key": self.headers.get("x-goog-api-key"),
                "authorization": self.headers.get("Authorization"),
                "connection": self.client_address})
            status, document = next(replies)
            content = json.dumps(document).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            if status == 302:
                self.send_header("Location", "https://redirect.example/secret")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1beta/decisions", captured
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=2)


class GoogleDecisionsTests(unittest.TestCase):
    def test_defaults_and_explicit_price_override(self):
        model = backend()
        try:
            self.assertEqual(model.info.endpoint, "https://generativelanguage.googleapis.com/v1beta/decisions")
            self.assertEqual(model.info.requested_model, "gemini-decision-1")
            self.assertEqual(model.question_types, ("choice", "noul", "score"))
            self.assertEqual(model.info.probability_decimals, 2)
            self.assertEqual(model.info.details["provider_option_order"], "utf8-byte-sorted")
            self.assertEqual(model.info.details["confidence_semantics"], "top-outcome-probability")
            self.assertEqual(model.info.pricing.input_per_million_usd, .1)
            self.assertEqual(model.info.pricing.output_per_million_usd, 0)
            self.assertEqual(model.info.pricing.checked_at, "2026-11-11")
            self.assertNotIn(KEY, json.dumps(model.info.to_dict()))
        finally:
            model.close()
        free_tier = backend(input_per_million_usd=0, price_source="Explicit free tier")
        try:
            self.assertEqual(free_tier.info.pricing.cost(Usage(144, 0)), 0)
        finally:
            free_tier.close()

    def test_credential_precedence_and_explicit_environment_name(self):
        cases = [
            ({}, {"GEMINI_API_KEY": "GEMINI", "GOOGLE_API_KEY": "GOOGLE"}, "GEMINI"),
            ({}, {"GOOGLE_API_KEY": "GOOGLE"}, "GOOGLE"),
            ({"api_key_env": "CUSTOM"}, {"GEMINI_API_KEY": "GEMINI", "CUSTOM": "CUSTOM_VALUE"}, "CUSTOM_VALUE"),
        ]
        for settings, environ, expected in cases:
            with self.subTest(settings=settings, environ=environ), patch("decision_metrics.providers.JsonTransport") as transport:
                model = make_backend({"provider": "google-decisions", **settings}, environ=environ)
                self.assertEqual(transport.call_args.kwargs["api_key"], expected)
                model.close()
        for settings, environ in (({}, {}), ({"api_key_env": "CUSTOM"}, {"GEMINI_API_KEY": "GEMINI"})):
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, "credential"):
                make_backend({"provider": "google-decisions", **settings}, environ=environ)

    def test_native_three_types_over_http_with_key_header_and_usage(self):
        # The published Score example rounds each component separately.
        score_answer = {"type": "score", "score": 1.12, "confidence": .83,
                        "probabilities": {"2": .15, "0": .03, "1": .83},
                        "legend": {"0": "Cosmetic", "1": "Workaround", "2": "Blocked"}}
        with server([(200, response()), (200, response({"type": "noul", "noul": .97, "confidence": .97})),
                     (200, response(score_answer))]) as (endpoint, captured):
            model = backend(endpoint=endpoint)
            try:
                choice = model.choose(request({"ticket": "Charged twice", "currency": "€"}))
                noul = model.noul(NoulRequest("Refund requested.", "Does the customer request a refund?",
                                             {"false": "No refund requested", "true": {"meaning": "Refund requested"}}))
                score = model.score(ScoreRequest("Export fails in Safari.", ["Cosmetic", "Workaround", "Blocked"],
                                                "How severe is this?"))
            finally:
                model.close()
        self.assertEqual((choice.choice, choice.probabilities, choice.confidence),
                         ("billing", CHOICE["probabilities"], .9))
        self.assertEqual(noul.probability, .97)
        self.assertEqual((score.score, score.probabilities, score.confidence), (1.12, [.03, .83, .15], .83))
        for result in (choice, noul, score):
            self.assertEqual(result.usage, Usage(144, 0))
            self.assertIsNone(result.model)  # The documented response does not echo a model.
        for call in captured:
            self.assertEqual(call["path"], "/v1beta/decisions")
            self.assertEqual(call["key"], KEY)
            self.assertIsNone(call["authorization"])
            self.assertEqual(set(call["body"]), {"model", "input", "questions"})
            self.assertEqual(call["body"]["model"], "gemini-decision-1")
        self.assertEqual(captured[0]["connection"], captured[2]["connection"])
        self.assertEqual(captured[0]["body"]["input"], '{"ticket":"Charged twice","currency":"€"}')
        questions = [call["body"]["questions"]["action"] for call in captured]
        self.assertEqual(list(questions[0]["criteria"]), list(OPTIONS))
        self.assertEqual(questions[0]["criteria"]["technical"], '{"handles":["software errors"]}')
        self.assertEqual(questions[1]["criteria"], {"false_description": "No refund requested",
                                                  "true_description": '{"meaning":"Refund requested"}'})
        self.assertEqual(questions[2]["criteria"], ["Cosmetic", "Workaround", "Blocked"])

    def test_plain_text_and_media_shaped_objects_remain_text_input(self):
        states = ["A plain text ticket.", {"type": "image", "uri": "https://example.com/private.png"},
                  [{"type": "text", "text": "untrusted data"}], [1, True, None]]
        model = backend()
        model._transport = MagicMock()
        model._transport.post.return_value = response()
        try:
            for state in states:
                with self.subTest(state=state):
                    model.choose(request(state))
                    expected = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, separators=(",", ":"))
                    self.assertEqual(model._transport.post.call_args.args[0]["input"], expected)
        finally:
            model.close()

    def test_noul_criteria_are_optional_and_partial_descriptions_are_not_invented(self):
        model = backend()
        model._transport = MagicMock()
        model._transport.post.return_value = response({"type": "noul", "noul": .9})
        try:
            for criteria, expected in ((None, None), ({"true": "Requested"}, {"true_description": "Requested"}),
                                        ({"false": "Not requested"}, {"false_description": "Not requested"})):
                model.noul(NoulRequest("Ticket", "Refund requested?", criteria))
                question = model._transport.post.call_args.args[0]["questions"]["action"]
                if expected is None:
                    self.assertNotIn("criteria", question)
                else:
                    self.assertEqual(question["criteria"], expected)
        finally:
            model.close()

    def test_usage_is_unknown_when_absent_and_reported_values_are_preserved(self):
        model = backend()
        model._transport = MagicMock()
        try:
            document = response()
            del document["usage"]
            model._transport.post.return_value = document
            self.assertIsNone(model.choose(request()).usage)
            self.assertIsNone(model.info.pricing.cost(None))
            document = response()
            document["usage"]["total_output_tokens"] = 3
            document["model"] = "gemini-decision-1-revision"
            model._transport.post.return_value = document
            result = model.choose(request())
            self.assertEqual(result.usage, Usage(144, 3))
            self.assertEqual(result.model, "gemini-decision-1-revision")
        finally:
            model.close()

    def test_invalid_native_usage_is_not_fabricated_or_silently_accepted(self):
        bad_usage = [{"total_input_tokens": 144}, {"total_output_tokens": 0},
                     {"total_input_tokens": True, "total_output_tokens": 0},
                     {"total_input_tokens": -1, "total_output_tokens": 0},
                     {"total_input_tokens": 144, "total_output_tokens": "0"},
                     {"input_tokens": 144, "output_tokens": 0}, [], "invalid", None]
        model = backend()
        model._transport = MagicMock()
        try:
            for usage in bad_usage:
                with self.subTest(usage=usage):
                    model._transport.post.return_value = {**response(), "usage": usage}
                    with self.assertRaises(DecisionError) as caught:
                        model.choose(request())
                    self.assertEqual(caught.exception.code, "invalid_usage")
                    self.assertIsNone(caught.exception.usage)
        finally:
            model.close()

    def test_malformed_answers_preserve_usage_without_a_fallback_action(self):
        bad_answers = [
            (request(), {"type": "choice", "choice": "billing", "probabilities": {"billing": .9}}, "option_mismatch"),
            (request(), {**CHOICE, "choice": "technical"}, "choice_probability_mismatch"),
            (request(), {**CHOICE, "probabilities": {"billing": .5, "technical": .1}}, "probability_sum"),
            (NoulRequest("Ticket", "Refund?"), {"type": "noul", "noul": True}, "invalid_probability"),
            (ScoreRequest("Ticket", ["Low", "High"], "Urgency?"),
             {"type": "score", "score": .8, "probabilities": {"0": .8, "1": .2}}, "score_probability_mismatch"),
        ]
        model = backend()
        model._transport = MagicMock()
        sink = MemorySink()
        harness = Harness(model, sink)
        try:
            for packet, answer, code in bad_answers:
                with self.subTest(code=code):
                    model._transport.post.return_value = response(answer)
                    with self.assertRaises(DecisionError) as caught:
                        harness.ask(packet)
                    self.assertEqual(caught.exception.code, code)
                    self.assertEqual(caught.exception.usage, Usage(144, 0))
                    self.assertEqual(sink.events[-1]["usage"], {"input_tokens": 144, "output_tokens": 0})
                    self.assertEqual(sink.events[-1]["status"], "error")
                    self.assertNotIn(KEY, json.dumps(sink.events))
            self.assertFalse(any(event["event"] == "action" for event in sink.events))
        finally:
            model.close()

    def test_http_errors_retain_native_usage_without_body_or_retries(self):
        for status in (400, 429, 302):
            document = {"error": {"code": "safety", "message": KEY}, "usage": USAGE}
            with self.subTest(status=status), server([(status, document)]) as (endpoint, captured):
                model = backend(endpoint=endpoint)
                sink = MemorySink()
                try:
                    with self.assertRaises(DecisionError) as caught:
                        Harness(model, sink).choose(request())
                finally:
                    model.close()
                self.assertEqual(caught.exception.status_code, status)
                self.assertEqual(caught.exception.code, "http_error")
                self.assertEqual(caught.exception.usage, Usage(144, 0))
                self.assertEqual(len(captured), 1)
                self.assertNotIn(KEY, json.dumps(sink.events))
                self.assertNotIn(KEY, str(caught.exception))

    def test_google_key_cannot_be_sent_to_remote_plain_http_or_embedded_in_url(self):
        for endpoint in ("http://remote.example.org/v1beta/decisions",
                         "https://example.org/v1beta/decisions?key=secret",
                         "https://secret@example.org/v1beta/decisions"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                backend(endpoint=endpoint)

    def test_forced_choices_and_scores_do_not_call_google(self):
        model = backend()
        model._transport = MagicMock()
        harness = Harness(model, MemorySink())
        try:
            choice = harness.choose(ChoiceRequest("Ticket", {"only": "Only choice"}, "Choose."))
            score = harness.score(ScoreRequest("Ticket", ["Only level"], "Rate."))
            self.assertEqual(choice.source, "forced")
            self.assertEqual(score.source, "forced")
            model._transport.post.assert_not_called()
        finally:
            model.close()

    def test_maximum_choice_count_and_score_scale_use_common_request_limits(self):
        options = {str(index): str(index) for index in range(255)}
        model = backend()
        model._transport = MagicMock()
        model._transport.post.return_value = response({"type": "choice", "choice": "0",
            "probabilities": {key: float(key == "0") for key in options}})
        try:
            self.assertEqual(model.choose(ChoiceRequest("Ticket", options, "Choose.")).choice, "0")
            self.assertEqual(len(model._transport.post.call_args.args[0]["questions"]["action"]["criteria"]), 255)
            with self.assertRaises(ValueError):
                ChoiceRequest("Ticket", {str(i): str(i) for i in range(256)}, "Choose.")
            with self.assertRaises(ValueError):
                ScoreRequest("Ticket", [str(i) for i in range(11)], "Rate.")
        finally:
            model.close()


if __name__ == "__main__":
    unittest.main()
