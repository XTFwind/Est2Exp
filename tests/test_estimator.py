import unittest

from est2exp.dag.algorithms.dp import ALGORITHMS, collective_rounds
from est2exp.config import build_topology
from est2exp.estimator import estimate


def machine(groups, delay=0):
    """A 100 B/s link in each direction, with numbered servers and ranks."""
    affinity = []
    server_id = 0
    for gid, servers in enumerate(groups):
        entries = []
        for ranks in servers:
            entries.append({"server_id": server_id, "ranks": list(ranks)})
            server_id += 1
        affinity.append({"affinity_group_id": gid, "servers": entries})
    profile = {"bandwidth_gbps": 100, "latency_s": delay}
    return build_topology({"topology": {"affinity_groups": affinity,
                                        "total_devices": sum(len(r) for group in groups for r in group),
                                        "default_compute_tflops": 1, "default_memory_capacity_gb": 1,
                                        "default_hbm_bandwidth_bytes_s": 1},
                           "network": {"bandwidth_unit_bits": 8, "intra_server": profile,
                                       "intra_group": profile, "server_leaf": profile}})


def pp(key, a, b, size):
    return {"id": key, "kind": "pp_send", "ranks": [a, b], "bytes": size, "stage": a, "layer": a}


def simulate(nodes, topology, edges=(), strategy=None):
    return estimate({"nodes": nodes, "edges": [{"source": a, "target": b, "kind": "data"} for a,b in edges],
                     "timing_mode": "compute_and_communication", "assumptions": []}, topology, strategy or {})


class CollectiveTests(unittest.TestCase):
    def test_four_rank_oracles(self):
        placement = {r: (r // 2, r // 2) for r in range(4)}
        expected_rounds = {"ring": 6, "double_ring": 6, "hd": 4, "rhd": 4, "recursive_doubling": 2,
                           "nb": 4, "nhr": 4, "nhr_dual_band_theory": 4, "nhr_dual_band": 4,
                           "hierarchical": 4}
        self.assertEqual(set(expected_rounds), set(ALGORITHMS))
        for alg, expected in expected_rounds.items():
            with self.subTest(algorithm=alg):
                rounds = collective_rounds("all_reduce", list(range(4)), 400, alg, placement=placement)
                self.assertEqual(len(rounds), expected)
                self.assertEqual(sum(b for step in rounds for a, d, b in step),
                                 3200 if alg == "recursive_doubling" else 2400)
                rs = collective_rounds("reduce_scatter", list(range(4)), 400, alg, placement=placement)
                ag = collective_rounds("all_gather", list(range(4)), 400, alg, placement=placement)
                self.assertEqual(sum(b for step in rs for a, d, b in step), 1200)
                self.assertEqual(ag, [[(d, a, b) for a, d, b in step] for step in reversed(rs)])
        mesh = collective_rounds("all_reduce", list(range(4)), 400, "hierarchical",
                                 placement=dict.fromkeys(range(4), (0, 0)), level_algorithms={"within_server": "mesh"})
        self.assertEqual(len(mesh), 2)
        self.assertEqual(len(mesh[0]), 12)
        self.assertEqual(set(b for step in mesh for a, d, b in step), {100})

    def test_ragged_collectives_and_fold(self):
        for n in (2, 3, 5, 6, 7, 13):
            ranks = list(reversed(range(n)))
            placement = {r: (r // 5, r // 2) for r in ranks}
            for alg in ALGORITHMS:
                if alg == "hd" and n & (n - 1):
                    with self.assertRaisesRegex(ValueError, "power-of-two"):
                        collective_rounds("all_reduce", ranks, n*12, alg)
                    continue
                with self.subTest(n=n, alg=alg):
                    rounds = collective_rounds("all_reduce", ranks, n * 12, alg, placement=placement)
                    self.assertTrue(all(step for step in rounds))
                    self.assertTrue(all(a != d and a in ranks and d in ranks and b > 0
                                        for step in rounds for a, d, b in step))
                    if alg not in ("rhd", "recursive_doubling"):
                        self.assertAlmostEqual(sum(b for step in rounds for a, d, b in step),
                                               2*(n-1)*n*12)
        folded = collective_rounds("all_reduce", list(range(6)), 120, "rhd")
        self.assertEqual(len(folded), 7)
        self.assertEqual(folded[1], [(0, 1, 60), (2, 3, 60)])
        self.assertEqual(folded[-1], [(1, 0, 120), (3, 2, 120)])
        self.assertEqual(sum(b for s in folded for _, _, b in s), 1320)

    def test_zero_traffic_and_validation(self):
        for alg in ALGORITHMS:
            self.assertEqual(collective_rounds("all_gather", [0], 200, alg), [])
            self.assertEqual(collective_rounds("all_gather", [0, 1], 0, alg), [])
        with self.assertRaisesRegex(ValueError, "Unknown DP algorithm"):
            simulate([], machine([[[0]]]), strategy={"dp_algorithm":"none"})
        with self.assertRaisesRegex(ValueError, "per-level"):
            collective_rounds("all_gather", [0], 0, level_algorithms={"within_server":"random"})


class SimulationTests(unittest.TestCase):
    def test_zero_duration_path_ends_at_sink(self):
        events = [{"id": n,"kind":"forward","ranks":[0],"duration_s":0.0} for n in "abc"]
        report = simulate(events, machine([[[0]]]), [("a","c"),("b","c"),("a","b")])
        self.assertEqual(report["critical_path"], ["a","b","c"])
        self.assertEqual(report["latency_s"], 0.0)

    def test_link_latency_and_ledger(self):
        topology = machine([[[0, 1]], [[2, 3]]], .125)
        report = simulate([pp("a", 0, 1, 100), pp("b", 0, 2, 100)], topology, [("a", "b")])
        self.assertEqual(report["nodes"]["a"]["end_s"], 1.125)
        self.assertEqual(report["latency_s"], 2 + 7*.125)
        self.assertEqual(report["link_bytes"], 700)
        self.assertEqual(report["sent_bytes"], 200)
        self.assertEqual(report["rounds_by_domain"], {"dp": 0, "pp": 2})
        self.assertEqual(report["critical_path"], ["a", "b"])

    def test_contention_and_duplex(self):
        topology = machine([[[0, 1, 2, 3]]])
        def run(other):
            return simulate([pp("a",0,1,100), other], topology)["latency_s"]
        self.assertEqual(run(pp("shared send",0,2,100)), 2)
        self.assertEqual(run(pp("shared receive",2,1,100)), 2)
        self.assertEqual(run(pp("reverse",1,0,100)), 1)
        self.assertEqual(run(pp("separate",2,3,100)), 1)
        topology = machine([[[0, 1]], [[2, 3]]])
        self.assertEqual(simulate([pp("a",0,2,100),pp("b",1,3,100)], topology)["latency_s"], 2)

    def test_capacity_redistribution(self):
        topology = machine([[[0, 1, 2]]])
        nodes = [pp("long",0,1,200), {"id":"work", "kind":"forward","ranks":[2],"duration_s":.5},
                 pp("short",0,2,50)]
        report = simulate(nodes, topology, [("work","short")])
        self.assertEqual(report["nodes"]["short"]["end_s"], 1.5)
        self.assertEqual(report["nodes"]["long"]["end_s"], 2.5)
        self.assertEqual(report["latency_s"], 2.5)

    def test_stage_and_layer_attribution(self):
        topology = machine([[[0, 1, 2, 3]]])
        node = {"kind": "all_reduce", "id": "grad", "bytes":400, "stage":0,
                "ranks": [0, 1, 2, 3], "layer_bytes":{"7":100,"12":300}}
        report = simulate([node], topology, strategy={"rank_map":[0,1,2,3],"dp":2})
        self.assertEqual(report["sent_bytes_by_layer"], {"7":600,"12":1800})
        self.assertEqual(report["sent_bytes_by_stage"], {"0":1200,"1":1200})
        self.assertEqual(report["sent_bytes_by_domain"], {"dp":2400,"pp":0})
        self.assertEqual(report["critical_path_rounds"], 6)
        self.assertEqual(report["latency_s"], 6)
        self.assertEqual(simulate([node], topology, strategy={"dp_algorithm":"double_ring"})["latency_s"], 6)
