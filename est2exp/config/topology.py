"""Build a two-tier folded Clos and a logical fabric within each affinity group."""
from collections import Counter
from itertools import combinations
import math


def fields(obj, allowed, where):
    if not isinstance(obj, dict) or set(obj) - set(allowed):
        raise ValueError(f"Unexpected fields or object type in {where}")


def number(value, where, *, zero=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (not zero and value == 0)):
        raise ValueError(f"Invalid numeric value for {where}")
    return value


def integer(value, where, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{where} must be an integer >= {minimum}")
    return value


def _profile(settings, unit):
    fields(settings, ("bandwidth_gbps", "efficiency", "latency_s"), "link profile")
    speed = number(settings["bandwidth_gbps"], "bandwidth_gbps")
    efficiency = number(settings.get("efficiency", 1), "efficiency")
    if efficiency > 1:
        raise ValueError("Link efficiency cannot exceed 1")
    nominal = speed * unit / 8
    return {"bandwidth_gbps": speed, "efficiency": efficiency,
            "latency_s": number(settings.get("latency_s", 0), "latency_s", zero=True),
            "bandwidth_bytes_s": nominal, "effective_bandwidth_bytes_s": nominal * efficiency}


def build_topology(config):
    """Materialize public hardware data; creates no files."""
    top = config["topology"]
    fields(top, ("total_devices", "default_compute_tflops", "default_memory_capacity_gb",
                 "default_hbm_bandwidth_bytes_s", "affinity_groups", "clos",
                 "device_overrides", "device_compute_tflops", "link_overrides"), "topology")
    total = integer(top["total_devices"], "total_devices", 1)
    default_compute = number(top["default_compute_tflops"], "default_compute_tflops")
    memory = number(top["default_memory_capacity_gb"], "default_memory_capacity_gb")
    hbm = number(top["default_hbm_bandwidth_bytes_s"], "default_hbm_bandwidth_bytes_s")
    net = config["network"]
    fields(net, ("bandwidth_unit_bits", "intra_server", "intra_group", "server_leaf",
                 "device_server", "leaf_spine"), "network")
    unit = number(net.get("bandwidth_unit_bits", 1e9), "bandwidth_unit_bits")
    profiles = {name: _profile(net[name], unit) for name in ("intra_server", "intra_group", "server_leaf")}
    for name in ("device_server", "leaf_spine"):
        profiles[name] = _profile(net[name], unit) if name in net else profiles["server_leaf"]
    clos = top.get("clos", {})
    fields(clos, ("spine_count", "leaf_count", "server_leaf_count", "spine_ports", "leaf_ports"), "clos")
    spines = integer(clos.get("spine_count", 2), "spine_count", 1)
    leaves = integer(clos.get("leaf_count", 2), "leaf_count", 1)
    uplinks = integer(clos.get("server_leaf_count", 1), "server_leaf_count", 1)
    if uplinks > leaves:
        raise ValueError("Default server uplinks exceed the number of leaves")
    overrides = {}
    for key, shorthand in (("device_compute_tflops", True), ("device_overrides", False)):
        seen = set()
        for rank_key, value in top.get(key, {}).items():
            rank = int(rank_key)
            if str(rank) != str(rank_key) or not 0 <= rank < total or rank in seen:
                raise ValueError("Invalid or duplicate device override rank")
            seen.add(rank)
            values = {"compute_tflops": value} if shorthand else value
            fields(values, ("compute_tflops", "memory_capacity_gb", "hbm_bandwidth_bytes_s"), key)
            overrides.setdefault(rank, {}).update({k: number(v, k) for k, v in values.items()})

    nodes, links, containment = [], [], []
    def connect(source, target, kind):
        links.append({"source": source, "target": target, "type": kind, **profiles[kind]})

    for role, count in (("spine", spines), ("leaf", leaves)):
        for i in range(count):
            switch = {"id": f"{role}:{i}", "type": "switch", "role": role}
            if role + "_ports" in clos:
                switch["ports"] = integer(clos[role + "_ports"], role + "_ports", 1)
            nodes.append(switch)
    for leaf in range(leaves):
        for spine in range(spines):
            connect(f"leaf:{leaf}", f"spine:{spine}", "leaf_spine")
    group_ids, server_ids, ranks = set(), set(), set()
    for group in sorted(top["affinity_groups"], key=lambda g: g["affinity_group_id"]):
        fields(group, ("affinity_group_id", "servers"), "affinity group")
        gid = integer(group["affinity_group_id"], "affinity_group_id")
        if gid in group_ids or not group["servers"]:
            raise ValueError("Each affinity group needs a unique ID and servers")
        group_ids.add(gid)
        parent = {"id": f"affinity:{gid}", "type": "affinity_group", "accelerator_compute_tflops": 0}
        nodes.append(parent)
        cards = []
        for server in sorted(group["servers"], key=lambda s: s["server_id"]):
            fields(server, ("server_id", "ranks", "leaf_ids", "compute_tflops"), "server")
            sid = integer(server["server_id"], "server_id")
            server_id = f"server:{sid}"
            targets = server.get("leaf_ids", [(len(server_ids)*uplinks+i) % leaves for i in range(uplinks)])
            if (sid in server_ids or not server["ranks"] or not targets or len(set(targets)) != len(targets)
                    or any(type(i) is not int or i < 0 or i >= leaves for i in targets)):
                raise ValueError("Invalid server ID, ranks or leaf connections")
            server_ids.add(sid)
            host = {"id": server_id, "type": "server", "affinity_group_id": gid,
                    "accelerator_compute_tflops": 0}
            if "compute_tflops" in server:
                host["compute_tflops"] = number(server["compute_tflops"], "host compute", zero=True)
            nodes.append(host)
            containment.append({"parent": parent["id"], "child": server_id})
            for i in sorted(targets):
                connect(server_id, f"leaf:{i}", "server_leaf")
            for rank in sorted(server["ranks"]):
                integer(rank, "rank")
                if rank in ranks or rank >= total:
                    raise ValueError("Duplicate or out-of-range rank")
                ranks.add(rank)
                override = overrides.get(rank, {})
                card = {"id": f"device:{rank}", "type": "accelerator", "rank": rank, "server_id": sid,
                        "affinity_group_id": gid, "compute_tflops": override.get("compute_tflops", default_compute),
                        "memory_capacity_bytes": int(override.get("memory_capacity_gb", memory) * 1e9),
                        "hbm_bandwidth_bytes_s": override.get("hbm_bandwidth_bytes_s", hbm)}
                cards.append(card)
                nodes.append(card)
                host["accelerator_compute_tflops"] += card["compute_tflops"]
                containment.append({"parent": server_id, "child": card["id"]})
                connect(card["id"], server_id, "device_server")
            parent["accelerator_compute_tflops"] += host["accelerator_compute_tflops"]
        for a, b in combinations(cards, 2):
            connect(a["id"], b["id"], "intra_server" if a["server_id"] == b["server_id"] else "intra_group")
    if len(ranks) != total:
        raise ValueError("Topology must cover every physical rank exactly once")
    by_pair = {frozenset((e["source"], e["target"])): e for e in links}
    for override in top.get("link_overrides", []):
        fields(override, ("source", "target", "bandwidth_gbps", "efficiency", "latency_s"), "link override")
        pair = frozenset((override["source"], override["target"]))
        edge = by_pair.pop(pair, None)
        if edge is None or len(override) <= 2:
            raise ValueError("Link override needs distinct existing endpoints and a property; apply it once")
        edge.update(_profile({k: override.get(k, edge[k]) for k in ("bandwidth_gbps", "efficiency", "latency_s")}, unit))
    degrees = Counter(end for e in links for end in (e["source"], e["target"]))
    for n in nodes:
        if n["type"] == "switch":
            n["used_ports"] = degrees[n["id"]]
            if "ports" in n and n["used_ports"] > n["ports"]:
                raise ValueError(f"Insufficient ports on {n['id']}")
    return {"format": "est2exp-topology-1", "bandwidth_unit_bits": unit, "nodes": nodes,
            "links": links, "containment": containment, "summary": {
                "total_devices": total, "total_servers": len(server_ids), "total_affinity_groups": len(group_ids),
                "total_compute_tflops": sum(n["compute_tflops"] for n in nodes if n["type"] == "accelerator")}}
