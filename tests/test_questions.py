"""Noul and Score: requests, validation, the harness, transports, replay and training export."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from decision_metrics import (BackendInfo, ChoiceRequest, DecisionError, DecisionResult, Harness,
                              NoulRequest, NoulResult, Pricing, ScoreRequest, ScoreResult,
                              StrandsChoiceExporter, Usage, export_training)
from decision_metrics.__main__ import main
from decision_metrics.providers import make_backend
from decision_metrics.report import read_events, summarize
from decision_metrics.types import request_from_dict, validate_result
from test_decisions import MemorySink
from test_transport import server

LEVELS = ["Calm", "Annoyed", "Angry", "Furious"]


def noul():
    return NoulRequest({"review": "Runs at 10 FPS on my 3070."}, "Does the reviewer complain about performance?",
                       {"true": "They say it runs badly.", "false": {"note": "Other complaints"}})


def score():
    return ScoreRequest({"review": "Refunded after an hour."}, LEVELS, "How angry is the reviewer?")


class Judge:
    """A backend that answers every question type from queued results."""

    def __init__(self, *results, info=None):
        self.info = info or BackendInfo("judge", "judge-v1", "local-test", pricing=Pricing(.1),
                                        probability_decimals=2)
        self.results = iter(results)
        self.calls = []

    def _next(self, request):
        self.calls.append(request)
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        return result

    choose = noul = score = _next

    def close(self):
        pass


class RequestTests(unittest.TestCase):
    def test_requests_validate_snapshot_and_round_trip(self):
        for bad in ({}, {"maybe": "x"}, {"true": None}, ["true"]):
            with self.subTest(criteria=bad), self.assertRaises(ValueError):
                NoulRequest({}, "Is it?", bad)
        for levels in ([], ["a"] * 11, ["a", None], "abc"):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                ScoreRequest({}, levels, "How much?")
        with self.assertRaises(ValueError):
            NoulRequest({}, "  ")
        state = {"nested": [1]}
        packet = NoulRequest(state, "Is it?")
        state["nested"].append(2)
        self.assertEqual(packet.state, {"nested": [1]})
        for request in (noul(), score()):
            copy = request_from_dict(request.question_type, request.to_dict())
            self.assertEqual(copy, request)
            self.assertEqual(copy.packet_hash, request.packet_hash)
        with self.assertRaises(ValueError):
            request_from_dict("box", {})

    def test_packet_hashes_differ_by_question_type_and_level_order(self):
        hashes = {NoulRequest({"x": 1}, "Q?").packet_hash,
                  ScoreRequest({"x": 1}, ["a", "b"], "Q?").packet_hash,
                  ScoreRequest({"x": 1}, ["b", "a"], "Q?").packet_hash,
                  ChoiceRequest({"x": 1}, {"a": {}, "b": {}}, "Q?").packet_hash}
        self.assertEqual(len(hashes), 4)


class ValidationTests(unittest.TestCase):
    def code(self, request, result, decimals=None):
        with self.assertRaises(DecisionError) as raised:
            validate_result(request, result, decimals)
        return raised.exception.code

    def test_noul_probability_must_be_a_probability(self):
        self.assertEqual(validate_result(noul(), NoulResult(.93)).probability, .93)
        for value in (1.2, -.1, float("nan"), None, True, "0.5"):
            with self.subTest(value=value):
                self.assertEqual(self.code(noul(), NoulResult(value)), "invalid_probability")
        self.assertEqual(self.code(noul(), DecisionResult("a")), "invalid_result_type")

    def test_score_checks_range_levels_sum_and_expected_level(self):
        # Jev's own answer: probabilities and the score rounded to two places.
        ok = ScoreResult(1.14, [0.0, .86, .14, 0.0], .86, "jev-1.13.0", Usage(395, 62))
        self.assertIs(validate_result(score(), ok, 2), ok)
        self.assertEqual(self.code(score(), ScoreResult(3.5)), "score_out_of_range")
        self.assertEqual(self.code(score(), ScoreResult(float("inf"))), "invalid_score")
        self.assertEqual(self.code(score(), ScoreResult(1, [.5, .5])), "level_mismatch")
        self.assertEqual(self.code(score(), ScoreResult(1, [.5, .5, .5, 0])), "probability_sum")
        self.assertEqual(self.code(score(), ScoreResult(1, [.5, .5, 0, 0], 1.5)), "invalid_confidence")
        # Levels read in the wrong order give the wrong expected level.
        error = None
        try:
            validate_result(score(), ScoreResult(1.14, [0.0, 0.0, .14, .86]), 2)
        except DecisionError as caught:
            error = caught
        self.assertEqual(error.code, "score_probability_mismatch")
        self.assertEqual(error.diagnostics["score"], 1.14)


class HarnessTests(unittest.TestCase):
    def test_noul_and_score_record_their_type_answer_usage_and_actions(self):
        sink = MemorySink()
        backend = Judge(NoulResult(.93, "judge-v1", Usage(100, 0)),
                        ScoreResult(2.1, [0.0, .1, .7, .2], .7, "judge-v1", Usage(200, 0)))
        harness = Harness(backend, sink)
        yes = harness.noul(noul(), phase="reviews", observation_id="r1")
        level = harness.score(score(), phase="reviews")
        harness.record_action(yes, "applied", reason="flagged")
        harness.record_action(level, "skipped")
        self.assertEqual((yes.question_type, yes.result.probability), ("noul", .93))
        self.assertEqual((level.question_type, level.result.score), ("score", 2.1))
        decisions = [e for e in sink.events if e["event"] == "decision"]
        self.assertEqual([e["question_type"] for e in decisions], ["noul", "score"])
        self.assertIsNone(decisions[0]["candidate_order"])
        self.assertEqual(decisions[0]["request"]["criteria"]["false"], {"note": "Other complaints"})
        self.assertEqual(decisions[0]["result"]["probability"], .93)
        self.assertEqual(decisions[1]["level_count"], 4)
        self.assertEqual(decisions[1]["result"]["probabilities"], [0.0, .1, .7, .2])
        self.assertAlmostEqual(decisions[1]["estimated_api_cost_usd"], .00002)
        summary = summarize(sink.events)["entries"][0]
        self.assertEqual(summary["question_types"], {"noul": 1, "score": 1})
        self.assertEqual(summary["input_tokens"], 300)
        self.assertEqual(summary["actions"], {"applied": 1, "skipped": 1})

    def test_one_level_is_forced_and_missing_capability_is_never_sent(self):
        class ChoiceOnly:
            info = BackendInfo("choice-only", None, "local-test", pricing=Pricing(.1))

            def choose(self, request):
                raise AssertionError("not called")

            def close(self):
                pass

        sink = MemorySink()
        harness = Harness(ChoiceOnly(), sink)
        forced = harness.score(ScoreRequest({}, ["Only"], "How much?"))
        self.assertEqual((forced.source, forced.result.score, forced.result.probabilities), ("forced", 0.0, [1.0]))
        with self.assertRaises(DecisionError) as raised:
            harness.noul(noul())
        self.assertEqual((raised.exception.code, raised.exception.kind), ("unsupported_question_type", "capability"))
        event = sink.events[-1]
        self.assertEqual((event["request_attempted"], event["estimated_api_cost_usd"]), (False, 0.0))
        self.assertEqual(summarize(sink.events)["entries"][0]["model_requests"], 0)

    def test_methods_take_only_their_own_request_type(self):
        harness = Harness(Judge(), MemorySink())
        with self.assertRaises(ValueError):
            harness.choose(noul())
        with self.assertRaises(ValueError):
            harness.noul(score())
        with self.assertRaises(ValueError):
            harness.ask({"state": {}})

    def test_invalid_answer_is_recorded_with_safe_diagnostics(self):
        sink = MemorySink()
        harness = Harness(Judge(ScoreResult(1.0, [.5, .5], usage=Usage(10, 0))), sink)
        with self.assertRaises(DecisionError):
            harness.score(score())
        event = sink.events[-1]
        self.assertEqual(event["error"]["code"], "level_mismatch")
        self.assertEqual(event["usage"], {"input_tokens": 10, "output_tokens": 0})
        self.assertEqual(event["diagnostics"], {"score": 1.0, "probabilities": {"0": .5, "1": .5}})


def systemone_response(answer, usage=None):
    return {"model": "jev-1.14.0", "usage": usage or {"input_tokens": 395, "output_tokens": 62},
            "answers": {"action": answer}}


NOUL_ANSWER = {"type": "noul", "noul": .03}
SCORE_ANSWER = {"type": "score", "score": 1.14, "confidence": .86,
                "legend": dict(enumerate(LEVELS)),
                "probabilities": {"0": 0.0, "1": .86, "2": .14, "3": 0.0}}


class SystemOneTransportTests(unittest.TestCase):
    def test_noul_and_score_payloads_and_answers_over_real_local_http(self):
        responses = [(200, systemone_response(NOUL_ANSWER)), (200, systemone_response(SCORE_ANSWER))]
        with server(responses) as (endpoint, captured):
            backend = make_backend({"provider": "jev", "endpoint": endpoint},
                                   environ={"TYPESAFE_API_KEY": "TEST_ONLY_SECRET"})
            try:
                yes = backend.noul(noul())
                level = backend.score(score())
            finally:
                backend.close()
        self.assertEqual((yes.probability, yes.model, yes.usage), (.03, "jev-1.14.0", Usage(395, 62)))
        self.assertEqual((level.score, level.probabilities, level.confidence), (1.14, [0.0, .86, .14, 0.0], .86))
        first, second = (c["body"]["questions"]["action"] for c in captured)
        self.assertEqual(first, {"type": "noul", "instructions": noul().instructions,
                                 "criteria": {"true": "They say it runs badly.",
                                              "false": '{"note":"Other complaints"}'}})
        self.assertEqual(second, {"type": "score", "instructions": score().instructions, "criteria": LEVELS})
        self.assertEqual(captured[0]["body"]["model"], "jev-1.14.0")
        self.assertEqual(backend.info.details["question_types"], ["choice", "noul", "score"])

    def test_jev_reads_typesafe_api_key_then_the_older_jev_key(self):
        with server([(200, systemone_response(NOUL_ANSWER))] * 2) as (endpoint, captured):
            for environ in ({"TYPESAFE_API_KEY": "TEST_ONLY_NEW", "JEV_KEY": "TEST_ONLY_OLD"},
                            {"JEV_KEY": "TEST_ONLY_OLD"}):
                backend = make_backend({"provider": "jev", "endpoint": endpoint}, environ=environ)
                try:
                    backend.noul(noul())
                finally:
                    backend.close()
            for settings, environ in (({}, {}), ({"api_key_env": "MY_KEY"}, {"JEV_KEY": "TEST_ONLY_OLD"})):
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    make_backend({"provider": "jev", "endpoint": endpoint, **settings}, environ=environ)
        self.assertEqual([c["authorization"] for c in captured], ["Bearer TEST_ONLY_NEW", "Bearer TEST_ONLY_OLD"])

    def test_noul_without_criteria_sends_none_and_wrong_levels_keep_usage(self):
        bad = {**SCORE_ANSWER, "probabilities": {"0": .5, "1": .5}}
        with server([(200, systemone_response(NOUL_ANSWER)),
                     (200, systemone_response(bad))]) as (endpoint, captured):
            backend = make_backend({"provider": "systemone", "endpoint": endpoint})
            try:
                backend.noul(NoulRequest("text", "Is it?"))
                with self.assertRaises(DecisionError) as raised:
                    backend.score(score())
            finally:
                backend.close()
        self.assertNotIn("criteria", captured[0]["body"]["questions"]["action"])
        self.assertEqual(raised.exception.code, "level_mismatch")
        self.assertEqual(raised.exception.usage, Usage(395, 62))

    def test_answer_of_another_type_is_rejected(self):
        with server([(200, systemone_response(SCORE_ANSWER))]) as (endpoint, _):
            backend = make_backend({"provider": "systemone", "endpoint": endpoint})
            try:
                with self.assertRaises(DecisionError) as raised:
                    backend.noul(noul())
            finally:
                backend.close()
        self.assertEqual(raised.exception.code, "missing_noul_answer")

    def test_providers_without_noul_or_score_refuse_before_any_request(self):
        with server([]) as (endpoint, captured):
            for settings in ({"provider": "kev", "endpoint": endpoint},
                             {"provider": "openrouter-decisions", "endpoint": endpoint, "model": "m",
                              "api_key_env": "KEY"}):
                backend = make_backend(settings, environ={"KEY": "TEST_ONLY_SECRET"})
                for request, call in ((noul(), backend.noul), (score(), backend.score)):
                    with self.subTest(provider=settings["provider"], type=request.question_type):
                        with self.assertRaises(DecisionError) as raised:
                            call(request)
                        self.assertEqual(raised.exception.kind, "capability")
                        self.assertFalse(raised.exception.request_attempted)
                backend.close()
            enabled = make_backend({"provider": "kev", "endpoint": endpoint,
                                    "question_types": ["choice", "noul"]})
            self.assertEqual(enabled.question_types, ("choice", "noul"))
            with self.assertRaises(DecisionError):
                enabled.score(score())
        self.assertEqual(captured, [])
        for bad in (["noul"], ["choice", "box"], "choice"):
            with self.subTest(question_types=bad), self.assertRaises(ValueError):
                make_backend({"provider": "kev", "question_types": bad})

    def test_cloudflare_envelope_carries_noul_answers(self):
        with server([(200, {"success": True, "result": systemone_response(NOUL_ANSWER)})]) as (endpoint, _):
            backend = make_backend({"provider": "clef", "endpoint": endpoint, "account_id": "abc"},
                                   environ={"CLOUDFLARE_AUTH_TOKEN": "TEST_ONLY_SECRET"})
            try:
                self.assertEqual(backend.noul(noul()).probability, .03)
            finally:
                backend.close()


class OpenAIQuestionTests(unittest.TestCase):
    def backend(self, endpoint):
        return make_backend({"provider": "openai-decisions",
                             "endpoint": endpoint.replace("/v1/systemone", "/v1/decisions")},
                            environ={"OPENAI_API_KEY": "TEST_ONLY_SECRET"})

    def test_predicate_and_score_map_to_noul_and_levels(self):
        predicate = {"model": "gpt-6-luna", "usage": {"input_tokens": 40, "output_tokens": 0},
                     "answers": [{"type": "predicate", "name": "action", "probability": .92}]}
        rated = {"model": "gpt-6-luna", "usage": {"input_tokens": 60, "output_tokens": 0},
                 "answers": [{"type": "score", "name": "action", "score": 1.1, "confidence": .55,
                              "probabilities": [{"value": 2, "label": "2", "probability": .2},
                                                {"value": 0, "label": "0", "probability": .1},
                                                {"value": 1, "label": "1", "probability": .7}]}]}
        request = ScoreRequest({"ticket": "Export fails in Safari."}, ["Cosmetic", "Workaround", "Blocked"],
                               "How severe is this issue?")
        with server([(200, predicate), (200, rated)]) as (endpoint, captured):
            backend = self.backend(endpoint)
            try:
                yes = backend.noul(NoulRequest("A cracked screen.", "Does the product have visible damage?"))
                level = backend.score(request)
            finally:
                backend.close()
        self.assertEqual((yes.probability, yes.usage.input_tokens), (.92, 40))
        self.assertEqual((level.score, level.probabilities, level.confidence), (1.1, [.1, .7, .2], .55))
        self.assertEqual(captured[0]["body"]["questions"],
                         [{"type": "predicate", "name": "action", "instructions": "Does the product have visible damage?"}])
        self.assertEqual(captured[0]["body"]["input"], "A cracked screen.")
        self.assertEqual(captured[1]["body"]["questions"][0]["levels"],
                         [{"label": "0", "description": "Cosmetic"}, {"label": "1", "description": "Workaround"},
                          {"label": "2", "description": "Blocked"}])

    def test_criteria_refusals_and_unknown_levels_never_become_answers(self):
        refusal = {"answers": [{"type": "refusal", "name": "action"}]}
        unknown = {"answers": [{"type": "score", "name": "action", "score": 1,
                                "probabilities": [{"value": 7, "probability": 1}]}]}
        with server([(200, refusal), (200, unknown)]) as (endpoint, captured):
            backend = self.backend(endpoint)
            try:
                with self.assertRaises(DecisionError) as criteria:
                    backend.noul(noul())
                with self.assertRaises(DecisionError) as refused:
                    backend.noul(NoulRequest({}, "Is it?"))
                with self.assertRaises(DecisionError) as levels:
                    backend.score(score())
            finally:
                backend.close()
        self.assertEqual((criteria.exception.code, criteria.exception.kind),
                         ("unsupported_noul_criteria", "capability"))
        self.assertEqual(refused.exception.code, "model_refusal")
        self.assertEqual(levels.exception.code, "level_mismatch")
        self.assertEqual(len(captured), 2)


def judged_journal(*answers, teacher="judge"):
    """One run asking each request in turn; the judge answers from `answers`."""
    sink = MemorySink()
    judge = Judge(*(result for _, result in answers),
                  info=BackendInfo(teacher, "judge-v1", "offline", probability_decimals=2))
    harness = Harness(judge, sink, run_id="run-1", metadata={"episode_id": "ep-1"})
    decisions = []
    for request, _ in answers:
        decision = harness.ask(request, phase="reviews")
        harness.record_action(decision, "applied")
        decisions.append(decision)
    harness.outcome("done")
    return sink.events, decisions


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class TrainingQuestionTests(unittest.TestCase):
    def test_human_labels_and_teacher_answers_for_noul_and_score(self):
        events, decisions = judged_journal(
            (noul(), NoulResult(.93)),
            (NoulRequest("text", "Is it?"), NoulResult(.2)),
            (score(), ScoreResult(2.1, [0.0, .1, .7, .2], .7)))
        human = {"run_id": "run-1", "decision_id": decisions[0].decision_id,
                 "label": {"answer": False, "source": "human", "labeler": "reviewer"}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generic"
            manifest = export_training(events, output, annotations=[human], teacher_backend="judge")
            train = rows(output / "train.jsonl")
        self.assertEqual(manifest["question_types"], {"noul": 2, "score": 1})
        self.assertEqual(manifest["label_sources"], {"human": 1, "teacher": 2})
        self.assertEqual([row["question_type"] for row in train], ["noul", "noul", "score"])
        self.assertEqual(train[0]["label"]["answer"], False)
        self.assertEqual(train[0]["provenance"]["model_result"]["probability"], .93)
        self.assertEqual((train[1]["label"]["answer"], train[1]["label"]["details"]["method"]),
                         (False, "recorded_provider_probability"))
        self.assertEqual((train[2]["label"]["level"], train[2]["label"]["details"]["method"]),
                         (2, "recorded_provider_top_level"))
        self.assertIsNone(train[2]["candidate_order"])

    def test_strands_rows_use_false_true_and_level_indexes(self):
        events, decisions = judged_journal((noul(), NoulResult(.93)), (NoulRequest("x", "Is it?"), NoulResult(.9)),
                                           (score(), ScoreResult(1.0, [0.0, 1.0, 0.0, 0.0])))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "strands"
            export_training(events, output, teacher_backend="judge", exporter=StrandsChoiceExporter())
            first, bare, level = rows(output / "train.jsonl")
        self.assertEqual((first["kind"], first["label"]), ("noul", 1))
        self.assertEqual(first["options"], [["false", '{"note":"Other complaints"}'],
                                            ["true", "They say it runs badly."]])
        self.assertEqual(bare["options"], [["false", None], ["true", None]])
        self.assertEqual((level["kind"], level["label"], level["task"]), ("score", 1, "reviews"))
        self.assertEqual(level["options"], [[str(i), text] for i, text in enumerate(LEVELS)])

    def test_labels_must_fit_their_question(self):
        events, decisions = judged_journal((noul(), NoulResult(.93)), (score(), ScoreResult(1.0, [0, 1.0, 0, 0])))
        bad = [(decisions[0], {"answer": "yes"}), (decisions[0], {"choice": "true"}),
               (decisions[1], {"level": 4}), (decisions[1], {"level": True})]
        for decision, target in bad:
            annotation = {"run_id": "run-1", "decision_id": decision.decision_id,
                          "label": {**target, "source": "human", "labeler": "reviewer"}}
            with self.subTest(label=target), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    export_training(events, Path(directory) / "out", annotations=[annotation])

    def test_conflicting_answers_for_one_noul_packet_fail(self):
        events, decisions = judged_journal((noul(), NoulResult(.9)), (noul(), NoulResult(.9)))
        annotations = [{"run_id": "run-1", "decision_id": d.decision_id,
                        "label": {"answer": answer, "source": "human", "labeler": "reviewer"}}
                       for d, answer in zip(decisions, (True, False))]
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            export_training(events, Path(directory) / "out", annotations=annotations)

    def test_forced_scores_are_not_training_examples(self):
        events, _ = judged_journal((ScoreRequest({}, ["Only"], "How much?"), None))
        with tempfile.TemporaryDirectory() as directory:
            manifest = export_training(events, Path(directory) / "out", teacher_backend="judge")
        self.assertEqual(manifest["excluded"], {"forced_or_nonprovider": 1})


class ReplayTests(unittest.TestCase):
    def test_replay_sends_noul_and_score_packets(self):
        with server([(200, systemone_response(NOUL_ANSWER)),
                     (200, systemone_response(SCORE_ANSWER))]) as (endpoint, captured):
            with (tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()),
                  redirect_stderr(io.StringIO())):
                path = Path(directory)
                config = path / "providers.toml"
                config.write_text(f'[backends.test]\nprovider = "systemone"\nendpoint = "{endpoint}"\n'
                                  'input_per_million_usd = 0.042\n')
                cases = path / "packets.jsonl"
                cases.write_text(json.dumps({"type": "noul", "phase": "reviews", **noul().to_dict()}) + "\n"
                                 + json.dumps({"type": "score", **score().to_dict()}) + "\n")
                events_path = path / "events.jsonl"
                code = main(["replay", str(cases), "--config", str(config), "--backend", "test",
                             "--output", str(events_path)])
                events = list(read_events(events_path))
                bad = path / "bad.jsonl"
                bad.write_text(json.dumps({"type": "box", "state": {}, "instructions": "Where?"}) + "\n")
                self.assertEqual(main(["replay", str(bad), "--config", str(config), "--backend", "test",
                                       "--output", str(path / "bad-events.jsonl")]), 1)
        self.assertEqual(code, 0)
        self.assertEqual([c["body"]["questions"]["action"]["type"] for c in captured], ["noul", "score"])
        decisions = [e for e in events if e["event"] == "decision"]
        self.assertEqual([(e["question_type"], e["phase"]) for e in decisions],
                         [("noul", "reviews"), ("score", "decision")])


if __name__ == "__main__":
    unittest.main()
