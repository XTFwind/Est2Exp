"""Write a self-contained, full-iteration DAG viewer."""
import json
from importlib.resources import files
from pathlib import Path


def render_html(dag, report, schedule, output):
    data = {"dag": dag, "schedule": schedule, "timing": report["timing_mode"],
            "latency": report["latency_s"], "critical": report["critical_path"], "details": report["nodes"]}
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    template = files(__package__).joinpath("viewer.html").read_text(encoding="utf-8")
    Path(output).write_text(template.replace("/*GRAPH_DATA*/", payload.replace("<", "\\u003c")), encoding="utf-8")
