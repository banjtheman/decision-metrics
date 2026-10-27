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
from typing import NoReturn
from urllib.parse import urlsplit

from .types import (BackendInfo, ChoiceRequest, DecisionError, DecisionResult, MAX_SCORE_LEVELS,
                    NoulRequest, NoulResult, Pricing, QUESTION_TYPES, ScoreRequest, ScoreResult,
                    answer_diagnostics, parse_usage, validate_result)
from .rendering import OPTION_RENDERING, render_options, render_value

_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
# question_types: what each provider documents beyond Choice. A provider without
# Noul or Score answers those with unsupported_question_type before any request;
# a configuration's question_types overrides the default.
ALL_TYPES = QUESTION_TYPES
PROVIDERS = {
    # TypeSafe's documented key name; JEV_KEY, the name before it, still works.
    "jev": {"model": "jev-1.14.0", "endpoint": "https://api.typesafe.ai/v1/systemone",
            "key_env": "TYPESAFE_API_KEY", "old_key_env": "JEV_KEY", "price": .042,
            "source": "https://docs.typesafe.ai/models",
            "probability_decimals": 2, "question_types": ALL_TYPES},
    "clef": {"model": "clef", "key_env": "CLOUDFLARE_AUTH_TOKEN", "price": .24,
             "source": "https://developers.cloudflare.com/workers-ai/models/clef/",
             "question_types": ALL_TYPES},
    "clef-flash": {"model": "clef-flash", "key_env": "CLOUDFLARE_AUTH_TOKEN", "price": .09,
                   "source": "https://developers.cloudflare.com/workers-ai/models/clef-flash/",
                   "question_types": ALL_TYPES},
    "perplexity": {"model": "pplx-decider-v1-27b",
                   "endpoint": "https://api.perplexity.ai/v1/decisions",
                   "key_env": "PERPLEXITY_API_KEY", "price": .04,
                   "source": "https://docs.perplexity.ai/docs/decisions/quickstart",
                   "question_types": ALL_TYPES},
    "openai-decisions": {"model": "gpt-6-luna",
                         "endpoint": "https://api.openai.com/v1/decisions",
                         "key_env": "OPENAI_API_KEY", "price": .10,
                         "source": "https://developers.openai.com/api/docs/guides/decisions",
                         "price_checked_at": "2026-10-06", "question_types": ALL_TYPES},
    "openrouter-decisions": {"endpoint": "https://openrouter.ai/api/alpha/decisions",
                             "key_env": "OPENROUTER_API_KEY"},
    "strands": {"model": "strands-decider-2B-hobson-v19",
                "endpoint": "http://127.0.0.1:8000/v1/systemone", "question_types": ALL_TYPES},
    "kev": {"model": "kev-latest", "endpoint": "http://127.0.0.1:8008/v1/systemone"},
    "meragpt": {"model": "state-decider-1", "endpoint": "https://meragpt.com/v1/systemone",
                "key_env": "MERAGPT_API_KEY", "price": .04,
                "source": "https://meragpt.com/docs/systemone", "max_options": 10,
                "question_types": ALL_TYPES},
    "systemone": {"question_types": ALL_TYPES},
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
    """The System One request shape: shared state and one named question.

    Despite the name it also asks Noul and Score questions, where the provider's
    question_types allow them.
    """

    def __init__(self, info: BackendInfo, *, transport: JsonTransport,
                 envelope: str = "direct", max_options: int = 255,
                 question_types=("choice",)):
        if envelope not in {"direct", "cloudflare"}:
            raise ValueError("Unknown response envelope")
        if (not isinstance(question_types, (list, tuple)) or "choice" not in question_types
                or not set(question_types) <= set(QUESTION_TYPES)):
            raise ValueError(f"Question types must include choice and come from {QUESTION_TYPES}")
        self.question_types = tuple(t for t in QUESTION_TYPES if t in question_types)
        self.info = replace(info, details={**info.details, "option_rendering": OPTION_RENDERING,
                                           "question_types": list(self.question_types)})
        self._transport = transport
        self._envelope = envelope
        self._max_options = max_options

    def choose(self, request: ChoiceRequest) -> DecisionResult:
        if not 2 <= len(request.options) <= self._max_options:
            raise DecisionError("unsupported_option_count", kind="capability")
        document, answer = self._ask(request)
        return self._parse_answer(document, answer, request)

    def noul(self, request: NoulRequest) -> NoulResult:
        self._supports(request)
        document, answer = self._ask(request)
        return self._parse_noul(document, answer, request)

    def score(self, request: ScoreRequest) -> ScoreResult:
        self._supports(request)
        document, answer = self._ask(request)
        return self._parse_score(document, answer, request)

    def _supports(self, request):
        if request.question_type not in self.question_types:
            raise DecisionError("unsupported_question_type", kind="capability")
        if isinstance(request, ScoreRequest) and not 2 <= len(request.levels) <= MAX_SCORE_LEVELS:
            raise DecisionError("unsupported_level_count", kind="capability")

    def _ask(self, request) -> tuple[dict, object]:
        document = self._transport.post(self._payload(request))
        if self._envelope == "cloudflare":
            if document.get("success") is not True or not isinstance(document.get("result"), dict):
                raise DecisionError("cloudflare_envelope", kind="validation")
            document = document["result"]
        answers = document.get("answers")
        return document, answers.get("action") if isinstance(answers, dict) else None

    def _payload(self, request) -> dict:
        payload = {"state": request.state, "questions": {"action": _question(request)}}
        if self.info.requested_model is not None:
            payload["model"] = self.info.requested_model
        return payload

    def _checked(self, document: dict, answer, request):
        """The answer's usage and model, and a fail() that keeps them with the error."""
        usage = parse_usage(document.get("usage"))
        model = _safe_model(document.get("model"))

        def fail(code, *, kind="validation") -> NoReturn:
            raise DecisionError(code, kind=kind, usage=usage, model=model,
                                diagnostics=answer_diagnostics(answer, request))

        if not isinstance(answer, dict) or answer.get("type") != request.question_type:
            fail(f"missing_{request.question_type}_answer")
        if document.get("model") is not None and model is None:
            fail("invalid_model")
        if document.get("usage") is not None and usage is None:
            fail("invalid_usage")
        return usage, model, fail

    def _parse_answer(self, document: dict, answer, request: ChoiceRequest) -> DecisionResult:
        usage, model, fail = self._checked(document, answer, request)
        # Native decision models promise an option distribution; a missing one
        # is different from a custom generative backend that never provides it.
        if not isinstance(answer.get("probabilities"), dict):
            fail("missing_probabilities")
        result = DecisionResult(choice=answer.get("choice"),
                                probabilities=answer["probabilities"],
                                confidence=answer.get("confidence"), model=model, usage=usage)
        return validate_result(request, result, self.info.probability_decimals)

    def _parse_noul(self, document: dict, answer, request: NoulRequest) -> NoulResult:
        usage, model, _ = self._checked(document, answer, request)
        return validate_result(request, NoulResult(answer.get("noul"), model, usage),
                               self.info.probability_decimals)

    def _parse_score(self, document: dict, answer, request: ScoreRequest) -> ScoreResult:
        usage, model, fail = self._checked(document, answer, request)
        # Probabilities come keyed by level number as text; results list them lowest first.
        distribution = answer.get("probabilities")
        if not isinstance(distribution, dict):
            fail("missing_probabilities")
        levels = [str(level) for level in range(len(request.levels))]
        if set(distribution) != set(levels):
            fail("level_mismatch")
        result = ScoreResult(answer.get("score"), [distribution[level] for level in levels],
                             answer.get("confidence"), model, usage)
        return validate_result(request, result, self.info.probability_decimals)

    def close(self):
        self._transport.close()


def _question(request) -> dict:
    """One System One question. Descriptions are rendered as text, as for options."""
    question = {"type": request.question_type, "instructions": request.instructions}
    if isinstance(request, ChoiceRequest):
        question["criteria"] = render_options(request.options)
    elif isinstance(request, ScoreRequest):
        question["criteria"] = [render_value(level) for level in request.levels]
    elif request.criteria is not None:
        question["criteria"] = render_options(request.criteria)
    return question


class OpenAIDecisionsBackend(ChoiceHTTPBackend):
    """Map requests to OpenAI's named-question and typed-value arrays.

    OpenAI calls a Noul a predicate, which takes no criteria, and gives Score
    levels a label and a description: the label is the level number, as in a
    System One legend, and the description is the level's text.
    """

    def __init__(self, info: BackendInfo, *, transport: JsonTransport,
                 question_types=ALL_TYPES):
        super().__init__(info, transport=transport, question_types=question_types)
        self.info = replace(self.info, details={**self.info.details,
            "api": "openai-decisions", "question_name": "action",
            "state_rendering": "text-or-compact-json-v1"})

    def choose(self, request: ChoiceRequest) -> DecisionResult:
        if not 2 <= len(request.options) <= self._max_options:
            raise DecisionError("unsupported_option_count", kind="capability")
        document, answer, fail = self._post(request, {"type": "choice", "name": "action",
            "instructions": request.instructions,
            "choices": [{"value": key, "description": description}
                        for key, description in render_options(request.options).items()]})
        distribution = {}
        for entry in self._entries(answer, fail):
            if (not isinstance(entry, dict) or not isinstance(entry.get("value"), str)
                    or entry["value"] not in request.options):
                fail("option_mismatch")
            key = entry["value"]
            if key in distribution:
                fail("duplicate_probability_value")
            distribution[key] = entry.get("probability")
        normalized = {**answer, "probabilities": distribution}
        return self._parse_answer(document, normalized, request)

    def noul(self, request: NoulRequest) -> NoulResult:
        self._supports(request)
        if request.criteria is not None:
            raise DecisionError("unsupported_noul_criteria", kind="capability")
        document, answer, _ = self._post(request, {"type": "predicate", "name": "action",
                                                   "instructions": request.instructions})
        return self._parse_noul(document, {"type": "noul", "noul": answer.get("probability")},
                                request)

    def score(self, request: ScoreRequest) -> ScoreResult:
        self._supports(request)
        document, answer, fail = self._post(request, {"type": "score", "name": "action",
            "instructions": request.instructions,
            "levels": [{"label": str(level), "description": render_value(text)}
                       for level, text in enumerate(request.levels)]})
        distribution = {}
        for entry in self._entries(answer, fail):
            value = entry.get("value") if isinstance(entry, dict) else None
            if type(value) is not int or not 0 <= value < len(request.levels):
                fail("level_mismatch")
            if str(value) in distribution:
                fail("duplicate_probability_value")
            distribution[str(value)] = entry.get("probability")
        return self._parse_score(document, {**answer, "probabilities": distribution}, request)

    def _post(self, request, question: dict):
        shared_input = request.state if isinstance(request.state, str) else json.dumps(
            request.state, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        document = self._transport.post({"model": self.info.requested_model,
                                         "input": shared_input, "questions": [question]})
        usage = parse_usage(document.get("usage"))
        model = _safe_model(document.get("model"))
        answers = document.get("answers")
        answer = answers[0] if isinstance(answers, list) and len(answers) == 1 else None
        missing = f"missing_{request.question_type}_answer"

        def fail(code, *, kind="validation") -> NoReturn:
            raise DecisionError(code, kind=kind, usage=usage, model=model,
                                diagnostics=answer_diagnostics(answer, request))

        if not isinstance(answer, dict) or answer.get("name") != "action":
            fail(missing)
        if answer.get("type") == "refusal":
            fail("model_refusal", kind="provider")
        if answer.get("type") != question["type"]:
            fail(missing)
        return document, answer, fail

    @staticmethod
    def _entries(answer: dict, fail) -> list:
        entries = answer.get("probabilities")
        if not isinstance(entries, list):
            fail("missing_probabilities")
        return entries


class OpenRouterDecisionsBackend(ChoiceHTTPBackend):
    """Native Decisions API for any configured model, without provider fallbacks."""

    def __init__(self, info: BackendInfo, *, transport: JsonTransport,
                 question_types=("choice",)):
        super().__init__(info, transport=transport, question_types=question_types)
        self.info = replace(self.info, details={**self.info.details,
            "api": "openrouter-decisions", "allow_provider_fallbacks": False})

    def _payload(self, request) -> dict:
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
    if not api_key and "api_key_env" not in settings and defaults.get("old_key_env"):
        api_key = environ.get(defaults["old_key_env"])
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
    question_types = settings.get("question_types", defaults.get("question_types", ("choice",)))
    if provider == "openai-decisions":
        if model is None:
            raise ValueError("OpenAI Decisions requires a model")
        return OpenAIDecisionsBackend(info, transport=transport, question_types=question_types)
    if provider == "openrouter-decisions":
        return OpenRouterDecisionsBackend(info, transport=transport, question_types=question_types)
    return ChoiceHTTPBackend(info, transport=transport,
                             envelope="cloudflare" if provider in {"clef", "clef-flash"} else "direct",
                             max_options=defaults.get("max_options", 255),
                             question_types=question_types)


def load_backend(path: str | Path, name: str) -> ChoiceHTTPBackend:
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    settings = config.get("backends", {}).get(name)
    if not isinstance(settings, dict):
        raise ValueError("Named backend is missing from the configuration")
    return make_backend(settings, name=name)
