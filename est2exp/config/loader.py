from __future__ import annotations

import json
import math
from pathlib import Path
import re
import runpy
import sys

from .topology import build_topology, fields, integer, number
from ..dag.algorithms.dp import ALIASES


def load_config(path):
    """Read JSON or execute a trusted Python configuration (serial entry point)."""
    path = Path(path).resolve()
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    elif path.suffix.lower() == ".py":
        root = path.parent
        while (root / "__init__.py").is_file():
            root = root.parent
        if root == path.parent:
            root = path.parent.parent
        prefix = path.relative_to(root).parts[0].removesuffix(".py")
        old_path, old_bytecode = sys.path[:], sys.dont_write_bytecode
        local_names = [p.stem for p in path.parent.glob("*.py")] + [prefix]
        saved = {k: v for k, v in sys.modules.copy().items() if k.split(".")[0] in local_names}
        try:
            for key in saved:
                del sys.modules[key]
            sys.dont_write_bytecode = True
            sys.path[:0] = [str(path.parent), str(root)]
            namespace = runpy.run_path(str(path))
            value = namespace["get_config"]() if "get_config" in namespace else namespace["CONFIG"]
        finally:
            sys.path[:] = old_path
            sys.dont_write_bytecode = old_bytecode
            for key in list(sys.modules):
                if key.split(".")[0] in local_names:
                    del sys.modules[key]
            sys.modules.update(saved)
    else:
        raise ValueError("Use a JSON or trusted Python configuration")
    if not isinstance(value, dict):
        raise ValueError("Configuration must be an object")
    return value


def _read_snapshot(path):
    snapshot = json.loads(path.read_text(encoding="utf-8-sig"))
    fields(snapshot, ("format", "bandwidth_unit_bits", "nodes", "links", "containment", "summary"), "topology snapshot")
    if snapshot["format"] != "est2exp-topology-1":
        raise ValueError("Expected an exported Est2Exp topology")
    unit = number(snapshot["bandwidth_unit_bits"], "bandwidth unit")
    node_keys = {"switch": {"role", "ports", "used_ports"}, "affinity_group": {"accelerator_compute_tflops"},
                 "server": {"affinity_group_id", "accelerator_compute_tflops", "compute_tflops"},
                 "accelerator": {"rank", "affinity_group_id", "server_id", "compute_tflops",
                                 "memory_capacity_bytes", "hbm_bandwidth_bytes_s"}}
    ids, ranks = set(), set()
    for node in snapshot["nodes"]:
        kind, nid = node["type"], node["id"]
        prefix, _, index = nid.partition(":")
        if (kind not in node_keys or not re.fullmatch(r"0|[1-9][0-9]*", index)
                or prefix not in {"switch": ("leaf", "spine"), "accelerator": ("device",),
                                  "affinity_group": ("affinity",), "server": ("server",)}[kind] or nid in ids):
            raise ValueError("Invalid topology node")
        ids.add(nid)
        fields(node, node_keys[kind] | {"id", "type"}, "topology node")
        for key in node.keys() - {"id", "type", "role"}:
            number(node[key], key, zero=True)
        if kind == "switch" and node["role"] != prefix:
            raise ValueError("Invalid switch role")
        if kind == "accelerator":
            if integer(node["rank"], "rank") != int(index):
                raise ValueError("Invalid device rank")
            ranks.add(node["rank"])
    for edge in snapshot["links"]:
        fields(edge, ("source", "target", "type", "bandwidth_gbps", "efficiency", "latency_s",
                      "bandwidth_bytes_s", "effective_bandwidth_bytes_s"), "topology link")
        if (edge["source"] not in ids or edge["target"] not in ids
                or edge["type"] not in ("intra_server", "intra_group", "device_server", "server_leaf", "leaf_spine")):
            raise ValueError("Invalid topology link")
        for key in edge.keys() - {"source", "target", "type"}:
            number(edge[key], key, zero=(key == "latency_s"))
        nominal = edge["bandwidth_gbps"] * unit / 8
        if (edge["efficiency"] > 1 or not math.isclose(edge["bandwidth_bytes_s"], nominal)
                or not math.isclose(edge["effective_bandwidth_bytes_s"], nominal * edge["efficiency"])):
            raise ValueError("Inconsistent link rates in snapshot; rebuild it from the hardware configuration")
    for relation in snapshot["containment"]:
        fields(relation, ("parent", "child"), "containment")
        if relation["parent"] not in ids or relation["child"] not in ids:
            raise ValueError("Unknown containment endpoint")
    summary = snapshot["summary"]
    fields(summary, ("total_devices", "total_affinity_groups", "total_servers", "total_compute_tflops"), "summary")
    for key, value in summary.items():
        number(value, key)
    if ranks != set(range(integer(summary["total_devices"], "total_devices", 1))):
        raise ValueError("Snapshot rank coverage does not match its summary")
    cards = [n for n in snapshot["nodes"] if n["type"] == "accelerator"]
    if not math.isclose(sum(n["compute_tflops"] for n in cards), summary["total_compute_tflops"]):
        raise ValueError("Inconsistent device compute total in snapshot")
    return snapshot


def resolve_topology(config, config_path):
    if "topology_file" in config:
        if "topology" in config or "network" in config:
            raise ValueError("Choose inline hardware or topology_file")
        return _read_snapshot(Path(config_path).resolve().parent / config["topology_file"])
    return build_topology(config)


def load_scenario(path):
    config = load_config(path)
    fields(config, ("layers", "strategy", "topology", "network", "topology_file", "search"), "scenario")
    hardware = resolve_topology(config, path)
    layers = config["layers"]
    if not isinstance(layers, list) or not layers:
        raise ValueError("Provide at least one anonymous layer")
    ids, timed = set(), set()
    for layer in layers:
        fields(layer, ("id", "gradient_bytes", "forward_bytes", "backward_bytes", "forward_s", "backward_s",
                       "weight_gradient_s"), "layer")
        lid = integer(layer["id"], "layer ID")
        if lid in ids:
            raise ValueError("Anonymous layer IDs must be unique")
        ids.add(lid)
        for name in ("gradient_bytes", "forward_bytes", "backward_bytes"):
            integer(layer[name], name)
        if ("forward_s" in layer) != ("backward_s" in layer):
            raise ValueError("Supply both forward_s and backward_s")
        timed.add("forward_s" in layer)
        for name in ("forward_s", "backward_s", "weight_gradient_s"):
            if name in layer:
                number(layer[name], name, zero=True)
        if "weight_gradient_s" in layer and layer["weight_gradient_s"] > layer.get("backward_s", -1):
            raise ValueError("weight_gradient_s is a portion of backward_s")
    if len(timed) != 1:
        raise ValueError("Supply timings for every layer or omit all timings")
    strategy = dict(config["strategy"])
    fields(strategy, ("pp", "dp", "microbatches", "schedule", "virtual_stages", "dp_sync",
                      "dp_algorithm", "rank_map", "dp_level_algorithms"), "strategy")
    for name in ("pp", "dp", "microbatches"):
        integer(strategy[name], name, 1)
    size = strategy["pp"] * strategy["dp"]
    strategy.setdefault("rank_map", list(range(size)))
    mapping = strategy["rank_map"]
    if (not isinstance(mapping, list) or len(mapping) != size or len(set(mapping)) != size
            or any(type(r) is not int or not 0 <= r < hardware["summary"]["total_devices"] for r in mapping)):
        raise ValueError("rank_map must assign distinct hardware devices to all strategy ranks")
    strategy.setdefault("schedule", "1f1b")
    strategy.setdefault("dp_sync", "all_reduce")
    algorithm = strategy.get("dp_algorithm", "ring").lower()
    strategy["dp_algorithm"] = ALIASES.get(algorithm, algorithm)
    result = {"hardware": hardware, "layers": layers, "strategy": strategy}
    if "search" in config:
        result["search"] = config["search"]
    return result
