"""Run the same simulation for CLI scenarios and evolved candidates."""
import json
from pathlib import Path

from .dag.builder import build_dag
from .dag.render import render_html
from .estimator import estimate


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def run_scenario(scenario, output_dir=None):
    dag = build_dag(scenario)
    report = estimate(dag, scenario["hardware"], scenario["strategy"])
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "topology.json", scenario["hardware"])
        write_json(destination / "scenario.json", {"topology_file": "topology.json", "layers": scenario["layers"],
                                                     "strategy": scenario["strategy"]})
        write_json(destination / "dag.json", dag)
        write_json(destination / "report.json", report)
        render_html(dag, report, scenario["strategy"]["schedule"], destination / "dag.html")
    return dag, report
