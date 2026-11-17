import json
import os
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from est2exp.config import load_scenario
from est2exp.evolve import run_evolution
from est2exp.evolve.islands import ISLANDS
from est2exp.evolve.scoring import ScoringProgram


@contextmanager
def model_service(respond):
    calls = []

    class Endpoint(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, self.headers.get("Authorization"), request))
            evidence = json.loads(request["messages"][1]["content"])
            source = respond(evidence)
            body = json.dumps({"choices": [{"message": {"content": json.dumps({"source": source})}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with patch.dict(os.environ, {
            "EST2EXP_LLM_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
            "EST2EXP_LLM_MODEL": "test-ranker",
            "EST2EXP_LLM_API_KEY": "synthetic-test-token",
        }):
            yield calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def fixture(directory, timed=True):
    profile = {"bandwidth_gbps": 40, "latency_s": 1e-6}
    config = {
        "topology": {
            "total_devices": 4, "default_compute_tflops": 1, "default_memory_capacity_gb": 1,
            "default_hbm_bandwidth_bytes_s": 1e8, "affinity_groups": [
                {"affinity_group_id": 0, "servers": [{"server_id": 0, "ranks": [0, 1]}]},
                {"affinity_group_id": 1, "servers": [{"server_id": 1, "ranks": [2, 3]}]},
            ],
        },
        "network": {"intra_server": profile, "intra_group": profile, "server_leaf": profile},
        "strategy": {"pp": 2, "dp": 2, "microbatches": 2},
        "layers": [{"id": i, "gradient_bytes": 600 + 10 * i, "forward_bytes": 12, "backward_bytes": 16,
                    **({"forward_s": 0.01, "backward_s": 0.02} if timed else {})} for i in range(4)],
    }
    path = directory / "input.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return load_scenario(path)


class EvolutionTests(unittest.TestCase):
    def test_model_generations_migrate_and_replay(self):
        from est2exp.pipeline import run_scenario

        def respond(evidence):
            # Mutate the selected parent's final arithmetic expression.
            return evidence["parent"]["source"].rstrip() + f" + {evidence['generation']} * (strategy['dp_sync'] == 'rs_ag')"

        with tempfile.TemporaryDirectory() as tmp, model_service(respond) as calls:
            directory = Path(tmp)
            scenario = fixture(directory)
            result = run_evolution(scenario, directory / "run", rounds=4)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["timing_mode"], "compute_and_communication")
            self.assertEqual(result["rejections"], [])
            self.assertEqual(len(calls), 16)
            self.assertEqual(len(result["migrations"]), 8)
            records = {item["id"]: item for item in result["programs"]}
            self.assertEqual(len(records), 4 + 16 + 8)
            for record in records.values():
                for parent in record["parents"] + record["inspirations"]:
                    self.assertIn(parent, records)
            for path, auth, request in calls:
                self.assertEqual((path, auth, request["model"]),
                                 ("/v1/chat/completions", "Bearer synthetic-test-token", "test-ranker"))
                evidence = json.loads(request["messages"][1]["content"])
                self.assertTrue(request["messages"][0]["content"].startswith(ISLANDS[evidence["island"]].PROMPT))
                self.assertEqual(evidence["parent"]["island"], evidence["island"])
                self.assertTrue(evidence["parent_feedback"])
                self.assertTrue(all("rounds_by_domain" in r and "sent_bytes_by_domain" in r
                                    for r in evidence["parent_feedback"].values()))
                views = evidence["candidate_features"].values()
                self.assertTrue(all((v["pp"], v["dp"], v["microbatches"]) == (2, 2, 2) for v in views))
                self.assertTrue(all(v["cross_group_pp_boundaries"] == 2 and v["max_dp_servers"] == 1 for v in views))
                for local_field in ('"layers"', '"hardware"', '"rank_map"'):
                    self.assertNotIn(local_field, request["messages"][1]["content"])
            names = list(ISLANDS)
            for migration in result["migrations"]:
                self.assertIn(migration["generation"], (2, 4))
                self.assertEqual(migration["to"], names[(names.index(migration["from"]) + 1) % 4])
                self.assertEqual(records[migration["parent"]]["island"], migration["from"])
                self.assertLess(records[migration["parent"]]["generation"], migration["generation"] + 1)
                self.assertFalse(migration["parent"].startswith(f"{migration['generation']}/")
                                 and migration["parent"].endswith("/migration"))
            self.assertTrue(any(m["retained"] for m in result["migrations"]))
            best = result["best"]
            self.assertEqual(best["latency_s"], min(r["latency_s"] for r in result["measurements"].values()
                                                   if "latency_s" in r))
            saved = load_scenario(directory / "run" / "best" / "scenario.json")
            self.assertEqual(saved["layers"], scenario["layers"])
            self.assertEqual(saved["hardware"], scenario["hardware"])
            self.assertEqual(run_scenario(saved)[1]["latency_s"], best["latency_s"])
            persisted = (directory / "run" / "evolution.json").read_text(encoding="utf-8")
            self.assertEqual(json.loads(persisted), result)
            self.assertNotIn("synthetic-test-token", persisted)

    def test_rejected_model_proposals_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp, model_service(lambda _: "import os") as calls:
            directory = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "All model proposals failed"):
                run_evolution(fixture(directory, timed=False), directory / "run", 1)
            summary = json.loads((directory / "run" / "evolution.json").read_text(encoding="utf-8"))
            self.assertEqual(len(calls), 4)
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["timing_mode"], "communication_only")
            self.assertEqual(len(summary["rejections"]), 4)
            self.assertIsNone(summary["best"])

    def test_one_bad_proposal_keeps_other_islands_running(self):
        def propose(evidence):
            if evidence["island"] == "memory_safe":
                return "def score_strategy(strategy):\n    return 'untrusted-provider-text'"
            return "def score_strategy(strategy):\n    return 1 + (strategy['schedule'] == 'gpipe') # free-form-provider-note"

        with tempfile.TemporaryDirectory() as tmp, model_service(propose) as calls:
            directory = Path(tmp)
            scenario = fixture(directory, timed=False)
            scenario["search"] = {"schedules": ["1f1b"], "dp_algorithms": ["ring"],
                                  "dp_syncs": ["all_reduce"], "island_capacity": 1}
            summary = run_evolution(scenario, directory / "run", rounds=2)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["candidate_count"], 1)
            self.assertEqual(len(summary["rejections"]), 2)
            self.assertEqual(summary["rejections"][0]["reason"], "Invalid model proposal or scoring expression")
            self.assertEqual(summary["best"]["strategy"], scenario["strategy"])
            exported = json.dumps(summary) + json.dumps([request for _, _, request in calls])
            for private in ("untrusted-provider-text", "free-form-provider-note", "synthetic-test-token", "127.0.0.1"):
                self.assertNotIn(private, exported)

    def test_invalid_search_names_rejected_before_a_request(self):
        with tempfile.TemporaryDirectory() as tmp, model_service(lambda _: "") as calls:
            directory = Path(tmp)
            scenario = fixture(directory)
            scenario["search"] = {"dp_algorithms": ["unrecognized-algorithm"]}
            with self.assertRaisesRegex(ValueError, "dp_algorithms"):
                run_evolution(scenario, directory / "run")
            self.assertFalse(calls)

    def test_ranking_program_is_arithmetic_only(self):
        for source in (
            "def score_strategy(strategy):\n    return open('sample')",
            "def score_strategy(strategy):\n    return strategy['unshared']",
            "def score_strategy(strategy):\n    while 1: pass\n    return 2",
            "def score_strategy(strategy):\n    return ().__class__",
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                ScoringProgram(source)
        scorer = ScoringProgram("def score_strategy(strategy):\n    return 1e309")
        with self.assertRaisesRegex(ValueError, "finite"):
            scorer({})
