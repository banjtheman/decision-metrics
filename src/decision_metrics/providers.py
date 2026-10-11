"""Documented System One, Workers AI, Perplexity, OpenAI and OpenRouter transports.

No SDK dependency, redirects, background calls, or automatic retries.
"""
from __future__ import annotations

from dataclasses import replace
from http.client import HTTPConnection, HTTPSConnection, HTTPException
import json
import math
import os
from pathlib import Path
import re
import tomllib
from urllib.parse import urlsplit

from .types import (BackendInfo, ChoiceRequest, DecisionError, DecisionResult, Pricing,
                    answer_diagnostics, parse_usage, validate_result)
from .rendering import OPTION_RENDERING, render_options

_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
PROVIDERS = {
    "jev": {"model": "jev-1.13.0", "endpoint": "https://api.typesafe.ai/v1/systemone",
            "key_env": "JEV_KEY", "price": .042, "source": "https://docs.typesafe.ai/models",
            "probability_decimals": 2},
    "clef": {"model": "clef", "key_env": "CLOUDFLARE_AUTH_TOKEN", "price": .24,
             "source": "https://developers.cloudflare.com/workers-ai/models/clef/"},
    "clef-flash": {"model": "clef-flash", "key_env": "CLOUDFLARE_AUTH_TOKEN", "price": .09,
                   "source": "https://developers.cloudflare.com/workers-ai/models/clef-flash/"},
    "perplexity": {"model": "pplx-decider-v1-27b",
                   "endpoint": "https://api.perplexity.ai/v1/decisions",
                   "key_env": "PERPLEXITY_API_KEY", "price": .04,
                   "source": "https://docs.perplexity.ai/docs/decisions/quickstart"},
    "openai-decisions": {"model": "gpt-6-luna",
                         "endpoint": "https://api.openai.com/v1/decisions",
                         "key_env": "OPENAI_API_KEY", "price": .10,
                         "source": "https://developers.openai.com/api/docs/guides/decisions",
                         "price_checked_at": "2026-10-06"},
    "openrouter-decisions": {"endpoint": "https://openrouter.ai/api/alpha/decisions",
                             "key_env": "OPENROUTER_API_KEY"},
    "strands": {"model": "strands-decider-2B-hobson-v19",
                "endpoint": "http://127.0.0.1:8000/v1/systemone"},
    "kev": {"model": "kev-latest", "endpoint": "http://127.0.0.1:8008/v1/systemone"},
    "meragpt": {"model": "state-decider-1", "endpoint": "https://meragpt.com/v1/systemone",
                "key_env": "MERAGPT_API_KEY", "price": .04,
                "source": "https://meragpt.com/docs/systemone", "max_options": 10},
    "systemone": {},
}


def _safe_model(value):
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_./:@+\-]{1,512}", value):
        return value
    return None


class JsonTransport:
    """One persistent HTTP connection. Failed calls are never resent."""

    def __init__(self, endpoint: str, *, api_key: str | None = None, timeout: float = 2.0):
        url = urlsplit(endpoint)
        if (url.scheme not in {"https", "http"} or not url.hostname or url.username is not None
                or url.password is not None or url.query or url.fragment):
            raise ValueError("Endpoint requires HTTP(S), a host, and no credentials/query/fragment")
        if url.scheme == "http" and api_key and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Bearer authentication on a remote endpoint requires HTTPS")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be finite and positive")
        # Accessing url.port validates its numeric range before any network activity.
        self._port = url.port
        self._scheme = url.scheme
        self._host = url.hostname
        self._path = url.path or "/"
        self._timeout = timeout
        self._api_key = api_key
        self._connection = None

    def post(self, payload: dict) -> dict:
        body = json.dumps(payload, allow_nan=False, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key
        try:
            if self._connection is None:
                connection_type = HTTPSConnection if self._scheme == "https" else HTTPConnection
                self._connection = connection_type(self._host, self._port, timeout=self._timeout)
            self._connection.request("POST", self._path, body=body, headers=headers)
            response = self._connection.getresponse()
            content = response.read(_MAX_RESPONSE_BYTES + 1)
            status = response.status
        except (OSError, HTTPException, ValueError):
            self.close()
            raise DecisionError("transport_failure", kind="transport") from None
        if len(content) > _MAX_RESPONSE_BYTES:
            self.close()
            raise DecisionError("response_too_large", kind="transport", status_code=status)
        try:
            document = json.loads(content)
        except (ValueError, UnicodeError):
            document = None
        if not 200 <= status < 300:
            self.close()
            usage = parse_usage(document.get("usage")) if isinstance(document, dict) else None
            raise DecisionError("http_error", kind="http", status_code=status, usage=usage)
        if not isinstance(document, dict):
            raise DecisionError("invalid_json_response", kind="validation")
        return document

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None


class ChoiceHTTPBackend:
    def __init__(self, info: BackendInfo, *, transport: JsonTransport,
                 envelope: str = "direct", max_options: int = 255):
        if envelope not in {"direct", "cloudflare"}:
            raise ValueError("Unknown response envelope")
        self.info = replace(info, details={**info.details, "option_rendering": OPTION_RENDERING})
        self._transport = transport
        self._envelope = envelope
        self._max_options = max_options

    def choose(self, request: ChoiceRequest) -> DecisionResult:
        if not 2 <= len(request.options) <= self._max_options:
            raise DecisionError("unsupported_option_count", kind="capability")
        document = self._transport.post(self._payload(request))
        if self._envelope == "cloudflare":
            if document.get("success") is not True or not isinstance(document.get("result"), dict):
                raise DecisionError("cloudflare_envelope", kind="validation")
            document = document["result"]
        answers = document.get("answers")
        answer = answers.get("action") if isinstance(answers, dict) else None
        return self._parse_answer(document, answer, request)

    def _payload(self, request: ChoiceRequest) -> dict:
        payload = {"state": request.state, "questions": {"action": {
            "type": "choice", "instructions": request.instructions,
            "criteria": render_options(request.options)}}}
        if self.info.requested_model is not None:
            payload["model"] = self.info.requested_model
        return payload

    def _parse_answer(self, document: dict, answer, request: ChoiceRequest) -> DecisionResult:
        usage = parse_usage(document.get("usage"))
        model = _safe_model(document.get("model"))

        def fail(code):
            raise DecisionError(code, kind="validation", usage=usage, model=model,
                                diagnostics=answer_diagnostics(answer, request))

        if not isinstance(answer, dict) or answer.get("type") != "choice":
            fail("missing_choice_answer")
        if document.get("model") is not None and model is None:
            fail("invalid_model")
        if document.get("usage") is not None and usage is None:
            fail("invalid_usage")
        # Native decision models promise an option distribution; a missing one
        # is different from a custom generative backend that never provides it.
        if not isinstance(answer.get("probabilities"), dict):
            fail("missing_probabilities")
        result = DecisionResult(choice=answer.get("choice"),
                                probabilities=answer["probabilities"],
                                confidence=answer.get("confidence"), model=model, usage=usage)
        return validate_result(request, result, self.info.probability_decimals)

    def close(self):
        self._transport.close()


class OpenAIDecisionsBackend(ChoiceHTTPBackend):
    """Map a Choice request to OpenAI's named-question and typed-value arrays."""

    def __init__(self, info: BackendInfo, *, transport: JsonTransport):
        super().__init__(info, transport=transport)
        self.info = replace(self.info, details={**self.info.details,
            "api": "openai-decisions", "question_name": "action",
            "state_rendering": "text-or-compact-json-v1"})

    def choose(self, request: ChoiceRequest) -> DecisionResult:
        if not 2 <= len(request.options) <= self._max_options:
            raise DecisionError("unsupported_option_count", kind="capability")
        shared_input = request.state if isinstance(request.state, str) else json.dumps(
            request.state, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        payload = {"model": self.info.requested_model, "input": shared_input,
            "questions": [{"type": "choice", "name": "action",
                "instructions": request.instructions,
                "choices": [{"value": key, "description": description}
                            for key, description in render_options(request.options).items()]}]}
        document = self._transport.post(payload)
        usage = parse_usage(document.get("usage"))
        model = _safe_model(document.get("model"))
        answers = document.get("answers")
        answer = answers[0] if isinstance(answers, list) and len(answers) == 1 else None

        def fail(code, *, kind="validation"):
            raise DecisionError(code, kind=kind, usage=usage, model=model,
                                diagnostics=answer_diagnostics(answer, request))

        if not isinstance(answer, dict) or answer.get("name") != "action":
            fail("missing_choice_answer")
        if answer.get("type") == "refusal":
            fail("model_refusal", kind="provider")
        if answer.get("type") != "choice":
            fail("missing_choice_answer")
        entries = answer.get("probabilities")
        if not isinstance(entries, list):
            fail("missing_probabilities")
        distribution = {}
        for entry in entries:
            if (not isinstance(entry, dict) or not isinstance(entry.get("value"), str)
                    or entry["value"] not in request.options):
                fail("option_mismatch")
            key = entry["value"]
            if key in distribution:
                fail("duplicate_probability_value")
            distribution[key] = entry.get("probability")
        normalized = {**answer, "probabilities": distribution}
        return self._parse_answer(document, normalized, request)


class OpenRouterDecisionsBackend(ChoiceHTTPBackend):
    """Native Decisions API for any configured model, without provider fallbacks."""

    def __init__(self, info: BackendInfo, *, transport: JsonTransport):
        super().__init__(info, transport=transport)
        self.info = replace(self.info, details={**self.info.details,
            "api": "openrouter-decisions", "allow_provider_fallbacks": False})

    def _payload(self, request: ChoiceRequest) -> dict:
        return {**super()._payload(request), "provider": {"allow_fallbacks": False}}


def make_backend(settings: dict, *, name: str | None = None,
                 environ: dict | None = None) -> ChoiceHTTPBackend:
    """Resolve credentials at runtime; public metadata contains no credential fields."""
    environ = os.environ if environ is None else environ
    provider = settings.get("provider")
    if provider not in PROVIDERS:
        raise ValueError("Unknown provider; choose one from PROVIDERS")
    defaults = PROVIDERS[provider]
    endpoint = settings.get("endpoint", defaults.get("endpoint"))
    if provider in {"clef", "clef-flash"}:
        account = settings.get("account_id") or environ.get(
            settings.get("account_id_env", "CLOUDFLARE_ACCOUNT_ID"))
        if not isinstance(account, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", account):
            raise ValueError("Cloudflare requires a valid account ID")
        if endpoint is None:
            endpoint = (f"https://api.cloudflare.com/client/v4/accounts/{account}"
                        f"/ai/run/@cf/cloudflare/{provider}")
    if not isinstance(endpoint, str):
        raise ValueError("Provider requires an endpoint")
    if (urlsplit(endpoint).hostname or "").endswith(".example"):
        raise ValueError("Replace the example endpoint with a real server before use")
    key_env = settings.get("api_key_env", defaults.get("key_env"))
    api_key = environ.get(key_env) if key_env else None
    if key_env and not api_key:
        raise ValueError("Required API credential environment variable is not set")
    model = settings.get("model", defaults.get("model"))
    if model is not None and _safe_model(model) is None:
        raise ValueError("Model must be a valid nonempty identifier")
    if provider == "openrouter-decisions" and model is None:
        raise ValueError("OpenRouter Decisions requires an explicit model")
    price_value = settings.get("input_per_million_usd", defaults.get("price"))
    price = None if price_value is None else Pricing(
        price_value, settings.get("output_per_million_usd", 0.0),
        settings.get("price_source", defaults.get("source", "user configuration")),
        settings.get("price_checked_at", defaults.get("price_checked_at", "2026-10-02")))
    transport = JsonTransport(endpoint, api_key=api_key, timeout=settings.get("timeout", 2.0))
    info = BackendInfo(name or settings.get("name", provider), model,
                       settings.get("deployment", provider), endpoint, price,
                       settings.get("details", {}),
                       settings.get("probability_decimals", defaults.get("probability_decimals")))
    if provider == "openai-decisions":
        if model is None:
            raise ValueError("OpenAI Decisions requires a model")
        return OpenAIDecisionsBackend(info, transport=transport)
    if provider == "openrouter-decisions":
        return OpenRouterDecisionsBackend(info, transport=transport)
    return ChoiceHTTPBackend(info, transport=transport,
                             envelope="cloudflare" if provider in {"clef", "clef-flash"} else "direct",
                             max_options=defaults.get("max_options", 255))


def load_backend(path: str | Path, name: str) -> ChoiceHTTPBackend:
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    settings = config.get("backends", {}).get(name)
    if not isinstance(settings, dict):
        raise ValueError("Named backend is missing from the configuration")
    return make_backend(settings, name=name)
