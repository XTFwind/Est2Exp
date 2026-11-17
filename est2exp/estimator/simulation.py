"""Event-driven, full-duplex, cut-through network with max-min fair flow rates."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf, isclose

from ..dag.algorithms.dp import algorithm_options, plan_collective
from .routing import Router


@dataclass
class _Flow:
    node: str
    resources: tuple
    remaining: float
    details: dict


def _fair_rates(active, capacities):
    """Progressive filling gives every flow its path bottleneck's fair share."""
    pending = set(active)
    remaining = {}
    rates = dict.fromkeys(active, 0.0)
    while pending:
        users = Counter(resource for flow_id in pending for resource in active[flow_id].resources)
        shares = {r: remaining.get(r, capacities[r]) / n for r, n in users.items()}
        increment = min(shares.values())
        saturated = {r for r, share in shares.items() if isclose(share, increment, rel_tol=1e-13)}
        for r, n in users.items():
            remaining[r] = max(0.0, remaining.get(r, capacities[r]) - increment * n)
        finished = set()
        for flow_id in pending:
            rates[flow_id] += increment
            if saturated.intersection(active[flow_id].resources):
                finished.add(flow_id)
        pending -= finished
    return rates


def estimate(dag, topology, strategy):
    """Simulate one iteration; all ranks, application sends and physical hops are explicit."""
    router = Router(topology)
    placement = {rank: (node["affinity_group_id"], node["server_id"]) for rank, node in router.devices.items()}
    capacities = {}
    for hop in router.hops.values():
        for resource in hop["resources"]:
            capacities[resource] = max(capacities.get(resource, 0), hop["bandwidth_bytes_s"])

    nodes = {node["id"]: node for node in dag["nodes"]}
    if len(nodes) != len(dag["nodes"]):
        raise ValueError("DAG node IDs must be unique")
    parents = defaultdict(list)
    children = defaultdict(list)
    for edge in dag["edges"]:
        a, b = edge["source"], edge["target"]
        if a not in nodes or b not in nodes:
            raise ValueError("Dependency endpoint missing from DAG")
        parents[b].append(a)
        children[a].append(b)
    waiting = {key: len(parents[key]) for key in nodes}
    plans, details = {}, {}
    algorithm, levels = algorithm_options(strategy.get("dp_algorithm", "ring"), strategy.get("dp_level_algorithms"))
    layers, stages, domains = Counter(), Counter(), Counter({"dp": 0.0, "pp": 0.0})
    by_link, by_rank = Counter(), Counter()
    domain_rounds = Counter({"dp": 0, "pp": 0})
    physical_stage = {rank: i // strategy["dp"] for i, rank in enumerate(strategy.get("rank_map", []))}
    for key, node in nodes.items():
        kind = node["kind"]
        entry = {"start_s": 0.0, "end_s": 0.0, "duration_s": 0.0, "rounds": 0, "sent_bytes": 0.0}
        details[key] = entry
        if kind == "pp_send":
            src, dst = node["ranks"]
            if node["bytes"] < 0:
                raise ValueError("PP payload must be nonnegative")
            rounds = [[(src, dst, node["bytes"])]] if node["bytes"] and src != dst else []
            domain = "pp"
        elif kind in ("all_reduce", "reduce_scatter", "all_gather"):
            rounds = plan_collective(kind, node["ranks"], node["bytes"], placement, algorithm, levels)
            domain = "dp"
        else:
            continue

        plans[key] = rounds
        rank_bytes = Counter({str(rank): 0.0 for rank in node["ranks"]})
        for step in rounds:
            for src, _, size in step:
                rank_bytes[str(src)] += size
        sent = sum(rank_bytes.values())
        entry.update({"rounds": len(rounds), "sent_bytes": sent, "sent_bytes_by_rank": dict(rank_bytes),
                      "steps": [], "domain": domain})
        by_rank.update(rank_bytes)
        domains[domain] += sent
        domain_rounds[domain] += len(rounds)
        if physical_stage:
            for rank, size in rank_bytes.items():
                stages[str(physical_stage[int(rank)])] += size
        elif node.get("stage") is not None:
            stages[str(node["stage"])] += sent
        if node.get("layer") is not None:
            layers[str(node["layer"])] += sent
        elif "layer_bytes" in node:
            for layer, size in node["layer_bytes"].items():
                layers[str(layer)] += sent * size / node["bytes"] if node["bytes"] else 0.0

    pending_events = []
    order = count()
    next_flow = count()
    for key in nodes:
        if not waiting[key]:
            heappush(pending_events, (0.0, next(order), "start", key))
    now = 0.0
    completed = {}
    controlling_parent = {}
    path_length = {}
    round_left = {}
    flows, active = {}, {}
    while pending_events or active:
        while pending_events and pending_events[0][0] <= now:
            _, _, event, key = heappop(pending_events)
            if event == "flow":
                active[key] = flows.pop(key)
                continue
            entry = details[key]
            if event == "start":
                entry["start_s"] = now
                if parents[key]:
                    parent = max(parents[key], key=lambda p: (completed[p], path_length[p], p))
                    controlling_parent[key] = parent
                    path_length[key] = path_length[parent] + 1
                else:
                    path_length[key] = 1
                if key in plans and plans[key]:
                    event = "round"
                else:
                    duration = nodes[key].get("duration_s", 0.0)
                    heappush(pending_events, (now + duration, next(order), "finish", key))
                    continue
            if event == "round":
                index = len(entry["steps"])
                messages = []
                entry["steps"].append({"round": index + 1, "start_s": now, "messages": messages})
                step = plans[key][index]
                round_left[key] = len(step)
                for src, dst, size in step:
                    hops = router.route(src, dst)
                    propagation = sum(hop["latency_s"] for hop in hops)
                    resources = tuple(r for hop in hops for r in hop["resources"])
                    path = [hops[0]["source"]] + [hop["target"] for hop in hops]
                    message = {"source": src, "target": dst, "bytes": size, "path": path,
                               "transfer_start_s": now + propagation}
                    messages.append(message)
                    for hop in hops:
                        by_link[hop["resources"][0]] += size
                    fid = next(next_flow)
                    flows[fid] = _Flow(key, resources, size, message)
                    heappush(pending_events, (now + propagation, next(order), "flow", fid))
            elif event == "finish":
                completed[key] = now
                entry["end_s"] = now
                entry["duration_s"] = now - entry["start_s"]
                for child in children[key]:
                    waiting[child] -= 1
                    if waiting[child] == 0:
                        heappush(pending_events, (now, next(order), "start", child))

        if not active:
            if pending_events:
                now = pending_events[0][0]
            continue
        rates = _fair_rates(active, capacities)
        finish = {fid: now + flow.remaining / rates[fid] for fid, flow in active.items()}
        until = min(min(finish.values()), pending_events[0][0] if pending_events else inf)
        for fid, flow in list(active.items()):
            if finish[fid] > until:
                flow.remaining = max(0.0, flow.remaining - (until - now) * rates[fid])
                continue
            del active[fid]
            flow.details["end_s"] = until
            key = flow.node
            round_left[key] -= 1
            if not round_left[key]:
                steps = details[key]["steps"]
                steps[-1]["end_s"] = until
                event = "finish" if len(steps) == len(plans[key]) else "round"
                heappush(pending_events, (until, next(order), event, key))
        now = until
    if len(completed) != len(nodes):
        raise ValueError("Dependency cycle prevents completion of the DAG")

    path = []
    if completed:
        sinks = (key for key in completed if not children[key])
        tip = max(sinks, key=lambda k: (completed[k], path_length[k], k))
        while tip is not None:
            path.append(tip)
            tip = controlling_parent.get(tip)
        path.reverse()
    return {"timing_mode": dag["timing_mode"], "assumptions": list(dag.get("assumptions", [])),
            "latency_s": max(completed.values(), default=0.0), "critical_path": path, "nodes": details,
            "sent_bytes": sum(domains.values()), "sent_bytes_by_domain": dict(domains),
            "sent_bytes_by_rank": dict(by_rank), "sent_bytes_by_layer": dict(layers),
            "sent_bytes_by_stage": dict(stages), "link_bytes": sum(by_link.values()),
            "bytes_by_directed_link": dict(by_link),
            "rounds": sum(e["rounds"] for e in details.values()),
            "rounds_by_domain": dict(domain_rounds),
            "critical_path_rounds": sum(details[key]["rounds"] for key in path)}
