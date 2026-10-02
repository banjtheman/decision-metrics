from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from decision_metrics.__main__ import main
from decision_metrics.report import read_events
from test_transport import server


class CliTests(unittest.TestCase):
    def test_offline_demo_and_summary_are_runnable(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as output:
            path = Path(directory) / "demo.jsonl"
            self.assertEqual(main(["demo", "--output", str(path)]), 0)
            self.assertEqual(main(["summary", str(path)]), 0)
            events = list(read_events(path))
        self.assertEqual(events[-1]["outcome"], "replay_complete")
        self.assertEqual(next(e for e in events if e["event"] == "action")["status"], "skipped")

    def test_sequential_replay_uses_frozen_packets_without_live_game_actions(self):
        response = {"model": "test", "usage": {"input_tokens": 100, "output_tokens": 0},
                    "answers": {"action": {"type": "choice", "choice": "a",
                                            "probabilities": {"a": .9, "b": .1}}}}
        with server([(200, response), (200, response)]) as (endpoint, captured):
            with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                path = Path(directory)
                config = path / "providers.toml"
                config.write_text(f'[backends.test]\nprovider = "systemone"\nendpoint = "{endpoint}"\nmodel = "test"\ndeployment = "fixture"\ninput_per_million_usd = 0.1\n')
                cases = path / "packets.jsonl"
                packet = {"phase": "shop", "state": {"gold": 20}, "options": {"a": {}, "b": {}},
                          "instructions": "CURRENT_FROZEN_PROMPT"}
                cases.write_text((json.dumps(packet) + "\n") * 3)
                events_path = path / "events.jsonl"
                code = main(["replay", str(cases), "--config", str(config), "--backend", "test",
                             "--output", str(events_path), "--limit", "2"])
                events = list(read_events(events_path))
            self.assertEqual(code, 0)
            self.assertEqual(len(captured), 2)
            self.assertEqual(captured[0]["body"]["questions"]["action"]["instructions"], "CURRENT_FROZEN_PROMPT")
            self.assertEqual([e["status"] for e in events if e["event"] == "action"], ["skipped", "skipped"])


if __name__ == "__main__":
    unittest.main()
