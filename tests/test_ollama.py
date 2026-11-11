"""Ollama Decision API integration, with local HTTP fixtures and no vendor calls."""
import json
import unittest
from unittest.mock import patch

from decision_metrics import (ChoiceRequest, DecisionError, Harness, NoulRequest,
                              ScoreRequest, Usage)
from decision_metrics.providers import make_backend
from test_decisions import MemorySink
from test_transport import server


def response(answer, model="nimble"):
    return {"model": model, "answers": {"action": answer},
            "usage": {"input_tokens": 174, "output_tokens": 0}}


CHOICE = {"type": "choice", "choice": "bug", "confidence": .8906,
          "probabilities": {"billing": .0125, "bug": .9781, "account": .0093}}
OPTIONS = {"billing": "Payments and refunds", "bug": {"category": "Software errors"},
           "account": "Login and account access"}


class OllamaTests(unittest.TestCase):
    def test_defaults_support_all_questions_without_credentials_or_assumed_cost(self):
        backend = make_backend({"provider": "ollama"}, environ={})
        try:
            self.assertEqual(backend.info.endpoint, "http://localhost:11434/v1/systemone")
            self.assertEqual(backend.info.requested_model, "nimble")
            self.assertEqual(backend.question_types, ("choice", "noul", "score"))
            self.assertEqual(backend.info.probability_decimals, 4)
            self.assertIsNone(backend.info.pricing)
        finally:
            backend.close()

    def test_cold_start_timeout_can_be_overridden(self):
        for settings, expected in (({"provider": "ollama"}, 120.0),
                                   ({"provider": "ollama", "timeout": 7.5}, 7.5),
                                   ({"provider": "strands"}, 2.0)):
            with self.subTest(settings=settings), patch("decision_metrics.providers.JsonTransport") as transport:
                backend = make_backend(settings, environ={})
                self.assertEqual(transport.call_args.kwargs["timeout"], expected)
                backend.close()

    def test_three_native_question_types_over_http_keep_results_usage_and_order(self):
        scored = {"type": "score", "score": 1.1, "confidence": .4,
                  "probabilities": {"2": .2, "0": .1, "1": .7}}
        responses = [(200, response(CHOICE)),
                     (200, response({"type": "noul", "noul": .9989})),
                     (200, response(scored))]
        state = {"ticket": "Checkout has returned 500 errors since 9am."}
        with server(responses) as (endpoint, captured):
            backend = make_backend({"provider": "ollama", "endpoint": endpoint},
                                   environ={"OLLAMA_API_KEY": "TEST_ONLY_UNUSED"})
            try:
                choice = backend.choose(ChoiceRequest(state, OPTIONS, "Which team should handle this?"))
                noul = backend.noul(NoulRequest(state, "Is a refund requested?",
                                               {"false": "No refund", "true": "Refund requested"}))
                score = backend.score(ScoreRequest(state, ["Routine", "Soon", "Immediate"],
                                                   "How urgent is this?"))
            finally:
                backend.close()
        self.assertEqual((choice.choice, choice.probabilities, choice.confidence),
                         ("bug", CHOICE["probabilities"], .8906))
        self.assertEqual(noul.probability, .9989)
        self.assertEqual((score.score, score.probabilities, score.confidence), (1.1, [.1, .7, .2], .4))
        for result in (choice, noul, score):
            self.assertEqual(result.usage, Usage(174, 0))
            self.assertEqual(result.model, "nimble")
        for request in captured:
            self.assertEqual(request["path"], "/v1/systemone")
            self.assertIsNone(request["authorization"])
            self.assertEqual(set(request["body"]), {"model", "state", "questions"})
            self.assertEqual(request["body"]["state"], state)
            self.assertEqual(request["body"]["model"], "nimble")
        questions = [request["body"]["questions"]["action"] for request in captured]
        self.assertEqual(list(questions[0]["criteria"]), list(OPTIONS))
        self.assertEqual(questions[0]["criteria"]["bug"], '{"category":"Software errors"}')
        self.assertEqual(questions[1], {"type": "noul", "instructions": "Is a refund requested?",
                                       "criteria": {"false": "No refund", "true": "Refund requested"}})
        self.assertEqual(questions[2], {"type": "score", "instructions": "How urgent is this?",
                                       "criteria": ["Routine", "Soon", "Immediate"]})

    def test_cloud_model_name_passes_through_local_server_without_auth(self):
        with server([(200, response({"type": "noul", "noul": .9}, "clef:cloud"))]) as (endpoint, captured):
            backend = make_backend({"provider": "ollama", "endpoint": endpoint, "model": "clef:cloud"},
                                   environ={})
            try:
                result = backend.noul(NoulRequest("Checkout is unavailable.", "Is this urgent?"))
            finally:
                backend.close()
        self.assertEqual(result.model, "clef:cloud")
        self.assertEqual(captured[0]["body"]["model"], "clef:cloud")
        self.assertEqual(captured[0]["body"]["state"], "Checkout is unavailable.")
        self.assertNotIn("criteria", captured[0]["body"]["questions"]["action"])
        self.assertIsNone(captured[0]["authorization"])
        self.assertIsNone(backend.info.pricing)

    def test_direct_cloud_resolves_default_or_explicit_key_without_exposing_it(self):
        settings = {"provider": "ollama", "endpoint": "https://ollama.com/v1/systemone", "model": "clef"}
        for extra, environ, expected in (
            ({}, {"OLLAMA_API_KEY": "TEST_ONLY_DEFAULT"}, "TEST_ONLY_DEFAULT"),
            ({"api_key_env": "CUSTOM_KEY"},
             {"OLLAMA_API_KEY": "TEST_ONLY_DEFAULT", "CUSTOM_KEY": "TEST_ONLY_CUSTOM"}, "TEST_ONLY_CUSTOM"),
        ):
            with self.subTest(extra=extra), patch("decision_metrics.providers.JsonTransport") as transport:
                backend = make_backend({**settings, **extra}, environ=environ)
                self.assertEqual(transport.call_args.kwargs["api_key"], expected)
                self.assertNotIn(expected, json.dumps(backend.info.to_dict()))
                self.assertIsNone(backend.info.pricing)
                backend.close()
        for extra, environ in (({}, {}), ({"api_key_env": "CUSTOM_KEY"}, {"OLLAMA_API_KEY": "unused"})):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "credential"):
                make_backend({**settings, **extra}, environ=environ)

    def test_explicit_auth_and_cost_are_honored_for_custom_server(self):
        with server([(200, response(CHOICE))]) as (endpoint, captured):
            backend = make_backend({"provider": "ollama", "endpoint": endpoint,
                                    "api_key_env": "CUSTOM_KEY", "input_per_million_usd": 0},
                                   environ={"CUSTOM_KEY": "TEST_ONLY_SECRET"})
            try:
                result = backend.choose(ChoiceRequest("Ticket", OPTIONS, "Choose a queue."))
            finally:
                backend.close()
        self.assertEqual(captured[0]["authorization"], "Bearer TEST_ONLY_SECRET")
        self.assertEqual(backend.info.pricing.cost(result.usage), 0)
        self.assertNotIn("TEST_ONLY_SECRET", json.dumps(backend.info.to_dict()))

    def test_option_limit_rejects_before_network_and_one_option_is_forced(self):
        with server([]) as (endpoint, captured):
            backend = make_backend({"provider": "ollama", "endpoint": endpoint}, environ={})
            sink = MemorySink()
            harness = Harness(backend, sink)
            try:
                with self.assertRaises(DecisionError) as caught:
                    harness.choose(ChoiceRequest("Ticket", {str(i): str(i) for i in range(27)}, "Choose."))
                forced = harness.choose(ChoiceRequest("Ticket", {"only": "Only"}, "Choose."))
            finally:
                backend.close()
        self.assertEqual(caught.exception.code, "unsupported_option_count")
        self.assertFalse(caught.exception.request_attempted)
        self.assertEqual(forced.source, "forced")
        self.assertEqual(forced.result.choice, "only")
        self.assertEqual(captured, [])

    def test_twenty_six_options_are_supported(self):
        options = {str(i): f"Label {i}" for i in range(26)}
        answer = {"type": "choice", "choice": "0",
                  "probabilities": {key: float(key == "0") for key in options}}
        with server([(200, response(answer))]) as (endpoint, captured):
            backend = make_backend({"provider": "ollama", "endpoint": endpoint}, environ={})
            try:
                self.assertEqual(backend.choose(ChoiceRequest("Ticket", options, "Choose.")).choice, "0")
            finally:
                backend.close()
        self.assertEqual(len(captured[0]["body"]["questions"]["action"]["criteria"]), 26)

    def test_keep_alive_duration_or_seconds_is_sent(self):
        values = ["5m", 0, -1, 2.5]
        with server([(200, response(CHOICE))] * len(values)) as (endpoint, captured):
            for value in values:
                backend = make_backend({"provider": "ollama", "endpoint": endpoint, "keep_alive": value},
                                       environ={})
                try:
                    backend.choose(ChoiceRequest("Ticket", OPTIONS, "Choose."))
                finally:
                    backend.close()
        self.assertEqual([request["body"]["keep_alive"] for request in captured], values)
        for value in (True, float("inf"), float("nan"), [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                make_backend({"provider": "ollama", "keep_alive": value}, environ={})

    def test_invalid_answer_preserves_usage_and_records_no_action(self):
        invalid = {**CHOICE, "probabilities": {"bug": .9781}}
        with server([(200, response(invalid))]) as (endpoint, captured):
            backend = make_backend({"provider": "ollama", "endpoint": endpoint}, environ={})
            sink = MemorySink()
            harness = Harness(backend, sink)
            try:
                with self.assertRaises(DecisionError) as caught:
                    harness.choose(ChoiceRequest("Ticket", OPTIONS, "Choose."))
            finally:
                backend.close()
        self.assertEqual(caught.exception.code, "option_mismatch")
        self.assertEqual(caught.exception.usage, Usage(174, 0))
        self.assertEqual(caught.exception.model, "nimble")
        self.assertEqual(sink.events[-1]["error"]["code"], "option_mismatch")
        self.assertEqual(sink.events[-1]["usage"], {"input_tokens": 174, "output_tokens": 0})
        self.assertFalse(any(event["event"] == "action" for event in sink.events))
        self.assertEqual(len(captured), 1)


if __name__ == "__main__":
    unittest.main()
