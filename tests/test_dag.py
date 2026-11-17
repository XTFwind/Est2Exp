from graphlib import TopologicalSorter
import unittest

from est2exp.dag.algorithms.pp import stage_slots
from est2exp.dag import build_dag


def example(schedule, pp=2, dp=2, microbatches=4, layer_count=5, **options):
    layers = [{"id": 10 + i, "gradient_bytes": 100 + i, "forward_bytes": 200 + i,
               "backward_bytes": 300 + i, "forward_s": 1, "backward_s": 3,
               "weight_gradient_s": 1} for i in range(layer_count)]
    strategy = {"schedule": schedule, "pp": pp, "dp": dp, "microbatches": microbatches,
                "dp_sync": "all_reduce", "rank_map": list(reversed(range(pp * dp))), **options}
    return {"layers": layers, "strategy": strategy}


class DagTests(unittest.TestCase):
    def test_physical_stage_orders(self):
        strategy = example("1f1b")["strategy"]
        labels = [[(slot[0].kind[0], slot[0].microbatch) for slot in row] for row in stage_slots(strategy)]
        self.assertEqual(labels[0], list(zip("fffbfbbb", [0, 1, 2, 0, 3, 1, 2, 3])))
        self.assertEqual(labels[1], list(zip("fbfbfbfb", [0, 0, 1, 1, 2, 2, 3, 3])))
        for row in stage_slots({**strategy, "schedule": "gpipe"}):
            self.assertEqual([slot[0].kind[0] for slot in row], list("ffffbbbb"))

    def test_vpp_group_order_and_wrap(self):
        case = example("vpp", dp=1)
        row = stage_slots(case["strategy"])[1]
        self.assertEqual([(slot[0].kind[0], slot[0].chunk, slot[0].microbatch) for slot in row[:6]],
                         [("f", 1, 0), ("f", 1, 1), ("f", 3, 0), ("b", 3, 0),
                          ("f", 3, 1), ("b", 3, 1)])
        dag = build_dag(case)
        sends = [n for n in dag["nodes"] if n["kind"] == "pp_send" and n["microbatch"] == 0]
        self.assertEqual([(n["ranks"], n["bytes"]) for n in sends if n["direction"] == "forward"],
                         [([1, 0], 201), ([0, 1], 202), ([1, 0], 203)])
        self.assertEqual([(n["ranks"], n["bytes"]) for n in sends if n["direction"] == "backward"],
                         [([0, 1], 301), ([1, 0], 302), ([0, 1], 303)])

    def test_dualpipe_two_rank_slots_and_mirrored_sync(self):
        case = example("dual_pipe", dp=1, dp_sync="rs_ag")
        row = stage_slots(case["strategy"])[0]
        sequence = [[(w.kind[0], w.microbatch, w.deferred) for w in slot] for slot in row]
        self.assertEqual(sequence, [
            [("f", 0, False)], [("f", 2, False)], [("f", 1, False)], [("b", 2, False)],
            [("f", 3, False), ("b", 0, False)], [("b", 3, False)], [("b", 1, True)], [("w", 1, False)],
        ])
        dag = build_dag(case)
        by_id = {n["id"]: n for n in dag["nodes"]}
        parents = {}
        for edge in dag["edges"]:
            parents.setdefault(edge["target"], set()).add(edge["source"])
        for s, expected in ((0, {"10": 100, "11": 101, "12": 102}), (1, {"13": 103, "14": 104})):
            rs = by_id[f"dp:reduce_scatter:s{s}"]
            self.assertEqual(set(rs["ranks"]), {0, 1})
            self.assertEqual(rs["layer_bytes"], expected)
            self.assertEqual(parents[f"dp:all_gather:s{s}"], {rs["id"]})
            self.assertEqual(len(parents[rs["id"]]), 4)
            weights = {k for k in parents[rs["id"]] if by_id[k]["kind"] == "weight_gradient"}
            self.assertEqual(weights, {"weight_gradient:r0:m1:l10", "weight_gradient:r0:m3:l10"} if s == 0 else set())
        self.assertEqual(by_id["backward:r0:m1:l10"]["duration_s"], 2)
        self.assertEqual(by_id["weight_gradient:r0:m1:l10"]["duration_s"], 1)
        self.assertEqual(by_id["backward:r0:m0:l10"]["duration_s"], 3)
        self.assertEqual(by_id["pp:forward:r0:m0:boundary0"]["ranks"], [1, 0])
        self.assertEqual(by_id["pp:forward:r0:m2:boundary0"]["ranks"], [0, 1])

    def test_whole_iteration_for_all_schedules(self):
        for name in ("gpipe", "1f1b", "vpp", "dual_pipe"):
            for pp, mb in ((2, 4), (4, 8), (4, 12)):
                with self.subTest(name=name, pp=pp, mb=mb):
                    case = example(name, pp=pp, microbatches=mb, layer_count=2 * pp + 1)
                    dag = build_dag(case)
                    by_id = {n["id"]: n for n in dag["nodes"]}
                    self.assertEqual(len(by_id), len(dag["nodes"]))
                    order = TopologicalSorter({node: [] for node in by_id})
                    for e in dag["edges"]:
                        self.assertIn(e["source"], by_id)
                        self.assertIn(e["target"], by_id)
                        order.add(e["target"], e["source"])
                    self.assertEqual(set(order.static_order()), set(by_id))
                    chunks = 2 * pp if name == "vpp" else pp
                    self.assertEqual(sum(n["kind"] == "pp_send" for n in by_id.values()), 4 * mb * (chunks - 1))
                    for kind in ("forward", "backward"):
                        self.assertEqual(sum(n["kind"] == kind for n in by_id.values()), 2 * mb * len(case["layers"]))
                    syncs = [n for n in by_id.values() if n["kind"] == "all_reduce"]
                    self.assertEqual(len(syncs), pp)
                    self.assertEqual(sum(n["bytes"] for n in syncs), sum(l["gradient_bytes"] for l in case["layers"]))
                    self.assertEqual({len(n["ranks"]) for n in syncs}, {4 if name == "dual_pipe" else 2})
                    self.assertEqual(sum(n.get("duration_s", 0) for n in by_id.values()), 2 * mb * 4 * len(case["layers"]))

    def test_degenerate_and_incompatible_schedules(self):
        dag = build_dag(example("1f1b", dp=1, pp=1, microbatches=1))
        self.assertEqual({n["kind"] for n in dag["nodes"]}, {"forward", "backward"})
        for case in (example("dual_pipe", pp=3), example("dual_pipe", microbatches=3),
                     example("dual_pipe", pp=4, microbatches=4), example("vpp", microbatches=3),
                     example("vpp", virtual_stages=3), example("gpipe", virtual_stages=2)):
            with self.subTest(strategy=case["strategy"]), self.assertRaises(ValueError):
                build_dag(case)

    def test_typed_dependencies_and_lane_metadata(self):
        basic = build_dag(example("gpipe", dp=1, pp=1, microbatches=1, layer_count=2))
        edges = {(e["source"], e["target"], e["kind"]) for e in basic["edges"]}
        pair = ("forward:r0:m0:l11", "backward:r0:m0:l11")
        for kind in ("activation_ready", "stage_order"):
            self.assertIn((*pair, kind), edges)
        dag = build_dag(example("dual_pipe", dp=1))
        edges = {(e["source"], e["target"], e["kind"]) for e in dag["edges"]}
        for n in dag["nodes"]:
            if n["kind"] in ("forward", "backward", "weight_gradient", "pp_send"):
                self.assertEqual(n["pipeline_direction"], "increasing_stage" if n["microbatch"] < 2 else "decreasing_stage")
            if n["kind"] == "backward":
                self.assertIn((n["id"].replace("backward", "forward"), n["id"], "activation_ready"), edges)
            if n["kind"] == "weight_gradient":
                self.assertIn((n["id"].replace("weight_gradient", "backward"), n["id"], "weight_input_ready"), edges)
        vpp = build_dag(example("vpp", dp=1))
        wraps = [n for n in vpp["nodes"] if n["kind"] == "pp_send" and n["source_chunk"] == 1 and n["target_chunk"] == 2]
        self.assertEqual(len(wraps), 4)
