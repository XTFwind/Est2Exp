import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from est2exp.config import build_topology, load_config, load_scenario, resolve_topology


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(EXAMPLES / "hardware.json")

    def test_build_and_override(self):
        top = build_topology(self.config)
        self.assertEqual(top["summary"]["total_compute_tflops"], 380)
        self.assertEqual(len(top["links"]), 10)
        nodes = {n["id"]: n for n in top["nodes"]}
        self.assertEqual(nodes["server:1"]["accelerator_compute_tflops"], 180)
        self.assertEqual(nodes["device:3"]["memory_capacity_bytes"], 16_000_000_000)
        self.assertEqual(nodes["device:3"]["hbm_bandwidth_bytes_s"], 400_000_000_000)
        self.assertEqual(len(top["containment"]), 6)
        links = {(e["source"], e["target"]): e for e in top["links"]}
        self.assertEqual(links["server:1", "leaf:1"]["effective_bandwidth_bytes_s"], 40e9 * .85 / 8)
        self.assertEqual(links["server:0", "leaf:0"]["effective_bandwidth_bytes_s"], 40e9 * .9 / 8)
        self.assertNotIn(("affinity:1", "server:1"), links)
        self.config["network"]["bandwidth_unit_bits"] = 2**30
        self.assertEqual(build_topology(self.config)["links"][0]["bandwidth_bytes_s"], 100 * 2**30 / 8)

    def test_relative_snapshot_and_python(self):
        scenario = load_scenario(EXAMPLES / "1f1b.py")
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "hw.json").write_text(json.dumps(scenario["hardware"]), encoding="utf8")
            public = {"layers": scenario["layers"], "strategy": scenario["strategy"], "topology_file": "hw.json"}
            (base / "case.json").write_text(json.dumps(public), encoding="utf8")
            self.assertEqual(load_scenario(base / "case.json"), scenario)
            (base / "case.py").write_text("def get_config():\n    return " + repr(public), encoding="utf8")
            self.assertEqual(load_scenario(base / "case.py"), scenario)
            snapshot = copy.deepcopy(scenario["hardware"])
            snapshot["nodes"][0]["note"] = "internal"
            (base / "hw.json").write_text(json.dumps(snapshot))
            with self.assertRaises(ValueError):
                load_scenario(base / "case.json")

    def test_physical_scope_and_ports(self):
        top = self.config["topology"]
        top["clos"]["leaf_ports"] = 1
        with self.assertRaisesRegex(ValueError, "ports"):
            build_topology(self.config)
        top["clos"]["leaf_ports"] = 3
        top["link_overrides"][0].update(source="device:0", target="device:2")
        with self.assertRaisesRegex(ValueError, "Link override"):
            build_topology(self.config)

    def test_custom_group_membership_and_access(self):
        groups = self.config["topology"]["affinity_groups"]
        groups[0]["servers"].extend(groups.pop()["servers"])
        top = build_topology(self.config)
        self.assertEqual(sum(e["type"] == "intra_group" for e in top["links"]), 4)
        groups[0]["servers"][1]["ranks"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "rank"):
            build_topology(self.config)

    def test_hardware_only_does_not_copy_other_sections(self):
        self.config["unrelated"] = {"private": "ignored"}
        self.assertNotIn("private", json.dumps(build_topology(self.config)))
        with self.assertRaises(ValueError):
            resolve_topology({**self.config, "topology_file": "hw.json"}, "case.json")


if __name__ == "__main__":
    unittest.main()
