from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest

from decision_metrics import ChoiceRequest, DecisionError
from decision_metrics.providers import ChoiceHTTPBackend, JsonTransport, make_backend
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


if __name__ == "__main__":
    unittest.main()
