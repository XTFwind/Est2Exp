from pathlib import Path
import unittest

from est2exp.config import build_topology, load_config
from est2exp.estimator.routing import Router


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).resolve().parents[1] / "examples" / "hardware.json")

    def test_direct_and_cross_group(self):
        router = Router(build_topology(self.config))
        self.assertEqual(router.route(0, 0), [])
        direct, = router.route(0, 1)
        self.assertEqual(direct["type"], "intra_server")
        self.assertEqual(set(direct["resources"]), {"device:0>device:1", "fabric_send:device:0", "fabric_receive:device:1"})
        path = router.route(0, 3)
        self.assertEqual([e["target"] for e in path], ["server:0", "leaf:0", "spine:0", "leaf:1", "server:1", "device:3"])
        self.assertEqual(path[-2]["bandwidth_bytes_s"], 40e9*.85/8)

    def test_shared_leaf_and_group_membership(self):
        groups = self.config["topology"]["affinity_groups"]
        groups[1]["servers"][0]["leaf_ids"] = [0, 1]
        router = Router(build_topology(self.config))
        self.assertEqual([e["type"] for e in router.route(0, 3)], ["device_server", "server_leaf", "server_leaf", "device_server"])
        groups[0]["servers"].extend(groups.pop()["servers"])
        router = Router(build_topology(self.config))
        self.assertEqual([e["type"] for e in router.route(0, 3)], ["intra_group"])
