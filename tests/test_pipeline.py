"""Exercise the public entry points with configurable, synthetic hardware."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from est2exp.config import build_topology, load_config, load_scenario
from est2exp.pipeline import run_scenario


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class PipelineTests(unittest.TestCase):
    def test_four_schedules_replay(self):
        with TemporaryDirectory() as tmp:
            for name in ("gpipe", "1f1b", "vpp", "dual_pipe"):
                with self.subTest(name=name):
                    case = load_scenario(EXAMPLES / (name + ".py"))
                    folder = Path(tmp) / name
                    dag, report = run_scenario(case, folder)
                    self.assertEqual(report["timing_mode"], "compute_and_communication")
                    self.assertGreater(report["latency_s"], 0)
                    self.assertTrue(report["critical_path"])
                    self.assertEqual(dag, json.loads((folder / "dag.json").read_text()))
                    replay = load_scenario(folder / "scenario.json")
                    self.assertEqual((dag, report), run_scenario(replay))
                    self.assertEqual(set(dag_node["id"] for dag_node in dag["nodes"]), set(report["nodes"]))
                    self.assertTrue((folder / "dag.html").is_file())

    def test_link_speed_and_placement_change_time_without_changing_sends(self):
        hardware = load_config(EXAMPLES / "hardware.json")
        case = load_scenario(EXAMPLES / "gpipe.py")
        for layer in case["layers"]:
            for name in ("forward_s", "backward_s", "weight_gradient_s"):
                layer.pop(name)
        case["strategy"].update(dp=1, microbatches=1, rank_map=[0, 2])
        dag, initial = run_scenario(case)
        self.assertEqual(initial["timing_mode"], "communication_only")
        self.assertEqual(len([n for n in dag["nodes"] if n["kind"] == "pp_send"]), 2)
        hardware["topology"]["link_overrides"][0]["bandwidth_gbps"] = .04
        case["hardware"] = build_topology(hardware)
        _, slow = run_scenario(case)
        self.assertGreater(slow["latency_s"], initial["latency_s"])
        self.assertEqual(slow["sent_bytes"], initial["sent_bytes"])
        case["strategy"]["rank_map"] = [0, 1]
        _, local = run_scenario(case)
        self.assertLess(local["latency_s"], slow["latency_s"])
        self.assertEqual(local["sent_bytes"], slow["sent_bytes"])

    def test_pp_message_count_and_gradient_accumulation(self):
        hardware = load_config(EXAMPLES / "hardware.json")
        hardware["topology"]["affinity_groups"][1]["servers"][0]["ranks"] += [4, 5]
        hardware["topology"]["total_devices"] = 6
        case = load_scenario(EXAMPLES / "gpipe.py")
        case["hardware"] = build_topology(hardware)
        case["strategy"].update(pp=3, microbatches=2, rank_map=list(range(6)))
        dag, _ = run_scenario(case)
        pp = [n for n in dag["nodes"] if n["kind"] == "pp_send"]
        dp = [n for n in dag["nodes"] if n["kind"] == "all_reduce"]
        self.assertEqual(len(pp), 16)
        self.assertEqual(len(dp), 3)
        self.assertEqual(sum(n["bytes"] for n in dp), sum(l["gradient_bytes"] for l in case["layers"]))
