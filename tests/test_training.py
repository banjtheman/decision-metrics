import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from decision_metrics import (BackendInfo, ChoiceRequest, DecisionResult, Harness,
                              StrandsChoiceExporter, export_training, read_annotations)
from decision_metrics.exporters import load_exporter
from decision_metrics.rendering import OPTION_RENDERING


class MemorySink:
    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)


class Teacher:
    info = BackendInfo("teacher", "teacher-v1", "offline",
                       details={"option_rendering": OPTION_RENDERING})

    def choose(self, request):
        return DecisionResult(next(iter(request.options)), model="teacher-v1")


def journal(run_id="run-1", episode_id=None, status="applied", state=None):
    sink = MemorySink()
    harness = Harness(Teacher(), sink, run_id=run_id,
                      metadata={"episode_id": episode_id or run_id})
    request = ChoiceRequest({"health": 5} if state is None else state,
                            {"east": {"danger": True, "distance": 12}, "north": "clear path"}, "Survive.")
    decision = harness.choose(request, phase="combat", observation_id="tick-1")
    harness.record_action(decision, status)
    harness.outcome("won", metrics={"waves": 20})
    return sink.events, {"run_id": run_id, "decision_id": decision.decision_id,
                         "packet_hash": request.packet_hash,
                         "label": {"choice": "north", "source": "human", "labeler": "reviewer-v1"}}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class TrainingTests(unittest.TestCase):
    def test_default_creates_review_queue_without_guessing_labels(self):
        events, _ = journal()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review"
            manifest = export_training(events, output)
            self.assertEqual(manifest["examples"], dict(train=0, validation=0, test=0))
            self.assertFalse(manifest["evaluation_ready"])
            self.assertEqual(manifest["format"], "decision-jsonl-v1")
            self.assertIsNone(rows(output / "review.jsonl")[0]["label"])
            self.assertEqual(manifest["unlabeled_eligible_decisions"], 1)

    def test_human_correction_preserves_original_model_choice_action_and_outcome(self):
        events, annotation = journal()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generic"
            manifest = export_training(events, output, annotations=[annotation])
            row = rows(output / "train.jsonl")[0]
            self.assertEqual(row["label"]["choice"], "north")
            self.assertEqual(row["request"]["options"]["east"], {"danger": True, "distance": 12})
            self.assertEqual(row["candidate_order"], ["east", "north"])
            self.assertEqual(row["provenance"]["model_result"]["choice"], "east")
            self.assertEqual(row["provenance"]["action"]["status"], "applied")
            self.assertEqual(row["provenance"]["run_outcome"]["outcome"], "won")
            self.assertEqual(manifest["label_sources"], {"human": 1})
            self.assertEqual(hashlib.sha256((output / "train.jsonl").read_bytes()).hexdigest(),
                             manifest["files"]["train.jsonl"]["sha256"])

    def test_strands_preserves_option_order_and_maps_corrected_label_to_index(self):
        events, annotation = journal()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "strands"
            manifest = export_training(events, output, annotations=[annotation], exporter=StrandsChoiceExporter())
            row = rows(output / "train.jsonl")[0]
            self.assertEqual(manifest["format"], "strands-choice-jsonl")
            self.assertEqual(row["label"], 1)
            self.assertEqual(row["options"], [["east", '{"danger":true,"distance":12}'], ["north", "clear path"]])
            self.assertEqual(row["state"], events[1]["request"]["state"])

    def test_teacher_imitation_is_explicit_and_human_labels_override_it(self):
        events, annotation = journal()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = export_training(events, root / "teacher", teacher_backend="teacher")
            label = rows(root / "teacher" / "train.jsonl")[0]["label"]
            self.assertEqual((label["source"], label["labeler"], label["choice"]), ("teacher", "teacher-v1", "east"))
            self.assertEqual(manifest["label_sources"], {"teacher": 1})
            export_training(events, root / "human", teacher_backend="teacher", annotations=[annotation])
            self.assertEqual(rows(root / "human" / "train.jsonl")[0]["label"]["source"], "human")

    def test_self_labels_require_explicit_opt_in(self):
        events, annotation = journal()
        annotation["label"]["source"] = "self"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "self"
            with self.assertRaisesRegex(ValueError, "Self-labels"):
                export_training(events, output, annotations=[annotation])
            self.assertFalse(output.exists())
            manifest = export_training(events, output, annotations=[annotation], allow_self_labels=True)
            self.assertEqual(manifest["label_sources"], {"self": 1})

    def test_expired_rejected_unknown_and_missing_actions_do_not_become_training_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            for status in ("expired", "rejected", "unknown"):
                events, _ = journal(status=status)
                manifest = export_training(events, Path(directory) / status, teacher_backend="teacher")
                self.assertEqual(manifest["excluded"], {"action_" + status: 1})
                self.assertEqual(manifest["examples"]["train"], 0)
            events, _ = journal()
            events = [e for e in events if e["event"] != "action"]
            manifest = export_training(events, Path(directory) / "missing", teacher_backend="teacher")
            self.assertEqual(manifest["excluded"], {"action_missing": 1})
            events, _ = journal(status="skipped")
            manifest = export_training(events, Path(directory) / "replay", teacher_backend="teacher", require_applied=True)
            self.assertEqual(manifest["excluded"], {"action_skipped": 1})

    def test_separate_episodes_can_be_held_out_but_the_same_episode_stays_together(self):
        first, a = journal("run-1", "episode-1")
        second, b = journal("run-2", "episode-2", state={"health": 6})
        with tempfile.TemporaryDirectory() as directory:
            manifest = export_training(first + second, Path(directory) / "held-out", annotations=[a, b],
                                       splits={"episode-1": "train", "episode-2": "test"})
            self.assertTrue(manifest["evaluation_ready"])
            self.assertEqual(manifest["examples"], dict(train=1, validation=0, test=1))
            second[0]["metadata"]["episode_id"] = "episode-1"
            with self.assertRaisesRegex(ValueError, "episode groups"):
                export_training(first + second, Path(directory) / "bad", splits={"run-1": "train", "run-2": "test"})

    def test_identical_packets_cannot_cross_splits_even_with_different_run_ids(self):
        first, _ = journal("run-1")
        second, _ = journal("run-2")
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "crosses dataset splits"):
            export_training(first + second, Path(directory) / "bad", splits={"run-1": "train", "run-2": "test"})

    def test_invalid_annotations_tampered_inputs_and_conflicting_labels_fail_before_writing(self):
        events, annotation = journal()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for field, value in (("choice", "illegal"), ("source", "guessed"), ("labeler", ""), ("weight", -1)):
                invalid = copy.deepcopy(annotation)
                invalid["label"][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    export_training(events, root / field, annotations=[invalid])
                self.assertFalse((root / field).exists())
            invalid = copy.deepcopy(annotation)
            invalid["packet_hash"] = "wrong"
            with self.assertRaisesRegex(ValueError, "packet hash"):
                export_training(events, root / "wrong-hash", annotations=[invalid])
            altered = copy.deepcopy(events)
            altered[1]["request"]["state"]["health"] = 999
            with self.assertRaisesRegex(ValueError, "packet hash"):
                export_training(altered, root / "tampered")
            second, b = journal("run-2")
            b["label"]["choice"] = "east"
            with self.assertRaisesRegex(ValueError, "Conflicting labels"):
                export_training(events + second, root / "conflict", annotations=[annotation, b])

    def test_custom_exporter_receives_complete_inputs_without_mutating_review_data(self):
        class CustomExporter:
            name = "custom-choice-v1"

            def serialize(self, request, label, provenance):
                result = {"input": request.to_dict(), "target": label["choice"],
                          "source": provenance["decision_id"]}
                provenance.clear()
                label.clear()
                return result

        events, annotation = journal()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "custom"
            manifest = export_training(events, output, annotations=[annotation], exporter=CustomExporter())
            row = rows(output / "train.jsonl")[0]
            self.assertEqual((manifest["format"], row["target"]), ("custom-choice-v1", "north"))
            self.assertEqual(row["input"], events[1]["request"])
            self.assertEqual(rows(output / "review.jsonl")[0]["label"]["choice"], "north")

    def test_edited_review_rows_and_plugin_loader_are_supported(self):
        events, annotation = journal()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "queue"
            export_training(events, output)
            row = rows(output / "review.jsonl")[0]
            row["label"] = annotation["label"]
            labels_path = Path(directory) / "annotations.jsonl"
            labels_path.write_text(json.dumps(row) + "\n")
            manifest = export_training(events, Path(directory) / "labeled", annotations=read_annotations(labels_path))
            self.assertEqual(manifest["examples"]["train"], 1)
        self.assertEqual(load_exporter("decision_metrics.exporters:StrandsChoiceExporter").name, "strands-choice-jsonl")


if __name__ == "__main__":
    unittest.main()
