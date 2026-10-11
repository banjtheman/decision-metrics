from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest
from unittest.mock import MagicMock

from decision_metrics import ChoiceRequest, DecisionError, Harness
from decision_metrics.providers import (ChoiceHTTPBackend, JsonTransport,
                                       OpenAIDecisionsBackend, OpenRouterDecisionsBackend, make_backend)
from decision_metrics.types import BackendInfo


@contextmanager
def server(responses):
    captured = []
    replies = iter(responses)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            captured.append({"path": self.path, "body": json.loads(body),
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
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1/systemone", captured
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


class TransportTests(unittest.TestCase):
    def test_systemone_payload_bearer_and_connection_reuse_over_real_local_http(self):
        response = {"model": "test-v1", "usage": {"input_tokens": 50, "output_tokens": 0},
                    "answers": {"action": {"type": "choice", "choice": "left",
                                            "probabilities": {"left": .9, "right": .1}}}}
        with server([(200, response), (200, response)]) as (endpoint, captured):
            transport = JsonTransport(endpoint, api_key="TEST_ONLY_SECRET")
            backend = ChoiceHTTPBackend(BackendInfo("test", "test-v1", "local", endpoint), transport=transport)
            packet = ChoiceRequest({"health": 5}, {"left": {}, "right": {}}, "Survive.")
            try:
                for _ in range(2):
                    self.assertEqual(backend.choose(packet).choice, "left")
            finally:
                backend.close()
            self.assertEqual(len(captured), 2)
            self.assertEqual(captured[0]["connection"], captured[1]["connection"])
            self.assertEqual(captured[0]["authorization"], "Bearer TEST_ONLY_SECRET")
            self.assertEqual(captured[0]["path"], "/v1/systemone")
            self.assertEqual(captured[0]["body"]["questions"]["action"]["instructions"], "Survive.")
            self.assertEqual(captured[0]["body"]["questions"]["action"]["criteria"], {"left": "{}", "right": "{}"})

    def test_structured_descriptions_are_text_without_losing_values_or_candidate_order(self):
        response = {"answers": {"action": {"type": "choice", "choice": "north",
                                            "probabilities": {"east": .2, "north": .8}}}}
        with server([(200, response)]) as (endpoint, captured):
            backend = ChoiceHTTPBackend(BackendInfo("test", None, "local"), transport=JsonTransport(endpoint))
            try:
                backend.choose(ChoiceRequest({}, {"east": {"danger": True, "distance": 12},
                                                  "north": "clear path"}, "Survive."))
            finally:
                backend.close()
            criteria = captured[0]["body"]["questions"]["action"]["criteria"]
            self.assertEqual(list(criteria), ["east", "north"])
            self.assertEqual(json.loads(criteria["east"]), {"danger": True, "distance": 12})
            self.assertEqual(criteria["north"], "clear path")

    def test_redirect_is_not_followed_or_retried(self):
        with server([(302, {"debug": "TEST_ONLY_SECRET"})]) as (endpoint, captured):
            transport = JsonTransport(endpoint, api_key="TEST_ONLY_SECRET")
            try:
                with self.assertRaises(DecisionError) as caught:
                    transport.post({"state": {}})
            finally:
                transport.close()
            self.assertEqual(caught.exception.status_code, 302)
            self.assertEqual(str(caught.exception), "http_error")
            self.assertEqual(len(captured), 1)

    def test_http_error_keeps_valid_usage_without_response_body(self):
        with server([(422, {"error": "TEST_ONLY_SECRET", "usage": {"input_tokens": 5, "output_tokens": 0}})]) as (endpoint, _):
            transport = JsonTransport(endpoint)
            try:
                with self.assertRaises(DecisionError) as caught:
                    transport.post({})
            finally:
                transport.close()
            self.assertEqual(caught.exception.usage.input_tokens, 5)
            self.assertNotIn("TEST_ONLY_SECRET", str(caught.exception))

    def test_urls_cannot_embed_credentials_or_redirect_bearer_over_plain_remote_http(self):
        for endpoint in ("https://token@example.org/api", "https://example.org/api?key=secret",
                         "https://example.org/api#key", "file:///tmp/api"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                JsonTransport(endpoint)
        with self.assertRaises(ValueError):
            JsonTransport("http://remote.example.org/api", api_key="secret")

    def test_remote_template_requires_configuration(self):
        with self.assertRaisesRegex(ValueError, "Replace the example endpoint"):
            make_backend({"provider": "strands", "endpoint": "https://YOUR-SERVER.example/v1/systemone"})

    def test_perplexity_uses_documented_model_endpoint_and_input_price(self):
        backend = make_backend({"provider": "perplexity"}, environ={"PERPLEXITY_API_KEY": "TEST_ONLY_SECRET"})
        self.assertEqual(backend.info.endpoint, "https://api.perplexity.ai/v1/decisions")
        self.assertEqual(backend.info.requested_model, "pplx-decider-v1-27b")
        self.assertEqual(backend.info.pricing.input_per_million_usd, .04)
        self.assertNotIn("TEST_ONLY_SECRET", json.dumps(backend.info.to_dict()))


class OpenRouterDecisionsTests(unittest.TestCase):
    def test_native_choice_request_routing_and_response_over_real_http(self):
        response = {"model": "microsoft/microsoft-decision-1", "provider": "Azure", "id": "gen-dec-test",
            "usage": {"input_tokens": 50, "output_tokens": 17, "cost": .0000021},
            "answers": {"action": {"type": "choice", "choice": "west", "confidence": .8,
                                    "probabilities": {"east": .1, "west": .9}}}}
        with server([(200, response)]) as (base, captured):
            endpoint = base.removesuffix("/v1/systemone") + "/api/alpha/decisions"
            backend = make_backend({"provider": "openrouter-decisions", "endpoint": endpoint,
                "model": "microsoft/microsoft-decision-1", "api_key_env": "OPEN_ROUTER_KEY",
                "input_per_million_usd": .042}, environ={"OPEN_ROUTER_KEY": "TEST_ONLY_SECRET"})
            request = ChoiceRequest({"hp": 5, "build": ["SMG"]},
                {"west": {"clearance": 120}, "east": {"clearance": 20}}, "Avoid enemies.")
            result = backend.choose(request)
            backend.close()
        self.assertEqual(captured[0]["path"], "/api/alpha/decisions")
        self.assertEqual(captured[0]["authorization"], "Bearer TEST_ONLY_SECRET")
        body = captured[0]["body"]
        self.assertEqual(body["state"], request.state)
        self.assertEqual(body["model"], "microsoft/microsoft-decision-1")
        self.assertEqual(body["provider"], {"allow_fallbacks": False})
        self.assertEqual(list(body["questions"]["action"]["criteria"]), ["west", "east"])
        self.assertEqual(json.loads(body["questions"]["action"]["criteria"]["west"]), {"clearance": 120})
        self.assertEqual(result.choice, "west")
        self.assertEqual(result.probabilities, {"east": .1, "west": .9})
        self.assertEqual(result.usage.output_tokens, 17)
        self.assertAlmostEqual(backend.info.pricing.cost(result.usage), .0000021)
        self.assertNotIn("TEST_ONLY_SECRET", json.dumps(backend.info.to_dict()))

    def test_model_is_explicit_and_price_is_not_assumed_for_other_models(self):
        backend = make_backend({"provider": "openrouter-decisions", "model": "typesafe/jev-1.13"},
                               environ={"OPENROUTER_API_KEY": "TEST_ONLY_SECRET"})
        self.assertIsInstance(backend, OpenRouterDecisionsBackend)
        self.assertEqual(backend.info.endpoint, "https://openrouter.ai/api/alpha/decisions")
        self.assertEqual(backend.info.requested_model, "typesafe/jev-1.13")
        self.assertIsNone(backend.info.pricing)
        self.assertEqual(backend.info.details["api"], "openrouter-decisions")
        with self.assertRaisesRegex(ValueError, "explicit model"):
            make_backend({"provider": "openrouter-decisions"},
                         environ={"OPENROUTER_API_KEY": "TEST_ONLY_SECRET"})
        with self.assertRaisesRegex(ValueError, "credential"):
            make_backend({"provider": "openrouter-decisions", "model": "microsoft/microsoft-decision-1"}, environ={})

    def test_invalid_distribution_keeps_usage_without_an_action(self):
        transport = MagicMock()
        transport.post.return_value = {"model": "microsoft/microsoft-decision-1",
            "usage": {"input_tokens": 50, "output_tokens": 0},
            "answers": {"action": {"type": "choice", "choice": "left",
                                    "probabilities": {"left": .5}}}}
        backend = OpenRouterDecisionsBackend(BackendInfo("microsoft", "microsoft/microsoft-decision-1", "test"),
                                            transport=transport)
        with self.assertRaises(DecisionError) as raised:
            backend.choose(ChoiceRequest({}, {"left": "left", "right": "right"}, "Choose."))
        self.assertEqual(raised.exception.code, "option_mismatch")
        self.assertEqual(raised.exception.usage.input_tokens, 50)


class OpenAIDecisionsTests(unittest.TestCase):
    def response(self):
        return {"model": "gpt-6-luna", "usage": {"input_tokens": 50, "output_tokens": 0},
                "answers": [{"name": "action", "type": "choice", "choice": "west",
                             "confidence": .75, "probabilities": [
                                 {"value": "east", "probability": .1},
                                 {"value": "west", "probability": .9}]}]}

    def request(self):
        return ChoiceRequest({"health": 5, "threat": "east", "build": {"weapons": ["SMG"]}},
                             {"east": {"danger": True, "distance": 12}, "west": "clear path"},
                             "Choose the movement that helps the player survive.")

    def test_native_payload_and_array_response_over_real_http(self):
        with server([(200, self.response()), (200, self.response())]) as (endpoint, captured):
            endpoint = endpoint.replace("/v1/systemone", "/v1/decisions")
            backend = make_backend({"provider": "openai-decisions", "endpoint": endpoint},
                                   environ={"OPENAI_API_KEY": "TEST_ONLY_SECRET"})
            try:
                for _ in range(2):
                    result = backend.choose(self.request())
                    self.assertEqual(result.choice, "west")
                    self.assertEqual(result.probabilities, {"east": .1, "west": .9})
                    self.assertEqual(result.confidence, .75)
                    self.assertEqual(result.usage.input_tokens, 50)
                    self.assertEqual(result.model, "gpt-6-luna")
            finally:
                backend.close()
            self.assertEqual(captured[0]["connection"], captured[1]["connection"])
            self.assertEqual(captured[0]["authorization"], "Bearer TEST_ONLY_SECRET")
            self.assertEqual(captured[0]["path"], "/v1/decisions")
            payload = captured[0]["body"]
            self.assertEqual(set(payload), {"model", "input", "questions"})
            self.assertEqual(json.loads(payload["input"]), self.request().state)
            question = payload["questions"][0]
            self.assertEqual(question["name"], "action")
            self.assertEqual(question["type"], "choice")
            self.assertEqual(question["instructions"], self.request().instructions)
            self.assertEqual([c["value"] for c in question["choices"]], ["east", "west"])
            self.assertEqual(json.loads(question["choices"][0]["description"]), self.request().options["east"])
            self.assertEqual(question["choices"][1]["description"], "clear path")

    def test_defaults_and_metadata_have_the_endpoint_specific_price_and_no_key(self):
        backend = make_backend({"provider": "openai-decisions"},
                               environ={"OPENAI_API_KEY": "TEST_ONLY_SECRET"})
        self.assertIsInstance(backend, OpenAIDecisionsBackend)
        self.assertEqual(backend.info.endpoint, "https://api.openai.com/v1/decisions")
        self.assertEqual(backend.info.requested_model, "gpt-6-luna")
        self.assertEqual(backend.info.pricing.input_per_million_usd, .10)
        self.assertEqual(backend.info.pricing.output_per_million_usd, 0)
        self.assertEqual(backend.info.pricing.checked_at, "2026-10-06")
        self.assertNotIn("TEST_ONLY_SECRET", json.dumps(backend.info.to_dict()))
        with self.assertRaisesRegex(ValueError, "credential"):
            make_backend({"provider": "openai-decisions"}, environ={})

    def test_plain_text_state_stays_plain_text(self):
        transport = MagicMock()
        transport.post.return_value = self.response()
        backend = OpenAIDecisionsBackend(BackendInfo("openai", "gpt-6-luna", "test"), transport=transport)
        backend.choose(ChoiceRequest("A threat is to the east.", self.request().options, "Survive."))
        self.assertEqual(transport.post.call_args.args[0]["input"], "A threat is to the east.")

    def test_invalid_native_answers_never_become_choices_and_keep_usage(self):
        original = self.response()
        mutations = [
            ("missing_choice_answer", {"answers": []}),
            ("missing_choice_answer", {"answers": original["answers"] * 2}),
            ("missing_choice_answer", {"answers": {"action": original["answers"][0]}}),
        ]
        answer_mutations = [
            ("missing_choice_answer", {"name": "unrelated"}),
            ("missing_choice_answer", {"type": "predicate"}),
            ("model_refusal", {"type": "refusal"}),
            ("missing_probabilities", {"probabilities": {"east": .1, "west": .9}}),
            ("duplicate_probability_value", {"probabilities": [
                {"value": "west", "probability": .5}, {"value": "west", "probability": .5}]}),
            ("option_mismatch", {"probabilities": [{"value": "west", "probability": 1}]}),
            ("option_mismatch", {"probabilities": [{"value": True, "probability": 1}]}),
            ("option_mismatch", {"probabilities": [{"value": "invented", "probability": 1}]}),
            ("invalid_probability", {"probabilities": [
                {"value": "east", "probability": .1}, {"value": "west", "probability": float("nan")}]}),
            ("probability_sum", {"probabilities": [
                {"value": "east", "probability": .1}, {"value": "west", "probability": .6}]}),
            ("choice_probability_mismatch", {"choice": "east"}),
            ("unknown_choice", {"choice": "invented"}),
            ("unknown_choice", {"choice": True}),
        ]
        for code, change in answer_mutations:
            document = deepcopy(original)
            document["answers"][0].update(change)
            mutations.append((code, document))
        for code, change in mutations:
            with self.subTest(code=code, change=change):
                document = {**deepcopy(original), **change}
                transport = MagicMock()
                transport.post.return_value = document
                backend = OpenAIDecisionsBackend(BackendInfo("openai", "gpt-6-luna", "test"), transport=transport)
                with self.assertRaises(DecisionError) as caught:
                    backend.choose(self.request())
                self.assertEqual(caught.exception.code, code)
                self.assertEqual(caught.exception.usage.input_tokens, 50)
                self.assertEqual(caught.exception.model, "gpt-6-luna")
                transport.post.assert_called_once()

    def test_a_single_legal_action_is_forced_without_openai_request(self):
        transport = MagicMock()
        backend = OpenAIDecisionsBackend(BackendInfo("openai", "gpt-6-luna", "test"), transport=transport)
        sink = MagicMock()
        ticket = Harness(backend, sink).choose(ChoiceRequest({}, {"continue": "Continue"}, "Continue."))
        self.assertEqual(ticket.source, "forced")
        self.assertEqual(ticket.result.choice, "continue")
        self.assertIsNone(ticket.result.usage)
        transport.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
