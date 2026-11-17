"""Collectives as ordered rounds of simultaneous directed (sender, receiver, bytes) messages.

Payload is the full gathered tensor (or the per-replica input of AllReduce).
Fractional bytes represent the ideal equal-shard convention. The ring and
doubling schedules below also accept unequal shards, as used by ragged grids.
"""
from collections import defaultdict
from itertools import accumulate
from math import isfinite, isqrt

ALGORITHMS = ("ring", "double_ring", "hd", "rhd", "recursive_doubling", "nb",
              "nhr", "nhr_dual_band_theory", "nhr_dual_band", "hierarchical")
ALIASES = {"recursive_halving_doubling": "rhd", "nonuniform_bruck": "nb",
           "two_ring": "double_ring", "2ring": "double_ring", "nhr_dual_band_actual": "nhr_dual_band"}
LEVELS = {"within_server": "ring", "within_group": "nhr", "across_groups": "nhr"}


def _reverse(rounds):
    return [[(dst, src, size) for src, dst, size in step] for step in reversed(rounds)]


def _scatter(ranks, shards, algorithm):
    """Reduce-scatter one vector onto its owners; gather is its transpose."""
    count = len(ranks)
    if count <= 1:
        return []
    if algorithm == "mesh":
        return [[(src, dst, shards[j]) for src in ranks for j, dst in enumerate(ranks)
                 if src != dst and shards[j]]]
    if algorithm == "ring":
        return [[(src, ranks[(i + 1) % count], shards[(i - step - 1) % count])
                 for i, src in enumerate(ranks) if shards[(i - step - 1) % count]]
                for step in range(count - 1)]
    # A Bruck gather sends the cyclic interval already accumulated at each
    # rank. Truncate the final exchange for non-power-of-two cardinalities.
    if algorithm == "nb" or (count & (count - 1)):
        gathered = []
        doubled = list(accumulate(shards + shards, initial=0))
        offset = 1
        while offset < count:
            width = min(offset, count - offset)
            step = []
            for i, src in enumerate(ranks):
                size = doubled[i + width] - doubled[i]
                if size:
                    step.append((src, ranks[(i - offset) % count], size))
            gathered.append(step)
            offset *= 2
        return _reverse(gathered)
    prefix = list(accumulate(shards, initial=0))
    rounds = []
    mask = count // 2
    while mask:
        step = []
        for i, src in enumerate(ranks):
            peer = i ^ mask
            start = (peer // mask) * mask
            size = prefix[start + mask] - prefix[start]
            if size:
                step.append((src, ranks[peer], size))
        rounds.append(step)
        mask //= 2
    return rounds


def _grouped_scatter(groups, shard, level_choices):
    """Apply RS along each dimension, starting at the leaves of a ragged grid.

    Each live suffix (e.g. the device's position within its server) carries
    the shards for that suffix. Smaller groups share owners for extra suffixes.
    This avoids imaginary ranks and retains all shards on irregular placements.
    """
    locations = {coord: rank for coord, rank in groups}
    owners = dict(locations)
    rounds = []
    for dimension in reversed(range(len(level_choices))):
        siblings = defaultdict(set)
        for coord in locations:
            siblings[coord[:dimension]].add(coord[dimension])
        suffixes = defaultdict(list)
        for coord in locations:
            suffixes[coord[dimension + 1:]].append(coord[dimension])
        pending = {}
        parallel = []
        for prefix, members in siblings.items():
            indices = sorted(members)
            for suffix, targets in suffixes.items():
                ranks = [owners[prefix + (i,) + suffix] for i in indices]
                sizes = [0.0] * len(indices)
                for target in targets:
                    index = target % len(indices)
                    sizes[index] += shard
                    pending[prefix + (target,) + suffix] = ranks[index]
                steps = _scatter(ranks, sizes, level_choices[dimension])
                for i, step in enumerate(steps):
                    if i == len(parallel):
                        parallel.append([])
                    parallel[i].extend(step)
        rounds.extend(parallel)
        owners = pending
    return rounds


def algorithm_options(algorithm, level_algorithms):
    """Resolve public strategy names, including when the graph has no DP nodes."""
    algorithm = ALIASES.get(algorithm.lower(), algorithm.lower())
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown DP algorithm: {algorithm}")
    choices = LEVELS | (level_algorithms or {})
    if choices.keys() != LEVELS.keys() or any(x not in {"ring", "nhr", "recursive_doubling", "mesh"}
                                               for x in choices.values()):
        raise ValueError("Unknown hierarchy level or per-level algorithm")
    return algorithm, choices


def collective_rounds(kind, ranks, payload_bytes, algorithm="ring", *, placement=None, level_algorithms=None):
    """Build an executable communication schedule using physical rank IDs.

    placement maps each rank to (affinity_group_id, server_id). Near-square
    NHR uses the supplied rank order; its dual-band theory variant puts nearby
    ranks next to one another. The actual dual-band variant uses affinity groups
    as rows. Hierarchical uses devices, servers and groups as three dimensions.
    """
    return plan_collective(kind, ranks, payload_bytes, placement, *algorithm_options(algorithm, level_algorithms))


def plan_collective(kind, ranks, payload_bytes, placement, algorithm, choices):
    """Plan an event using resolved strategy options."""
    if kind not in ("all_reduce", "reduce_scatter", "all_gather"):
        raise ValueError(f"Unsupported collective: {kind}")
    if not ranks or len(set(ranks)) != len(ranks) or not isfinite(payload_bytes) or payload_bytes < 0:
        raise ValueError("Supply unique participating ranks and a nonnegative finite byte count")
    n = len(ranks)
    if algorithm == "hd" and n & (n - 1):
        raise ValueError("HD requires a power-of-two domain; use RHD for other sizes")
    if n == 1 or payload_bytes == 0:
        return []
    shard = payload_bytes / n
    if algorithm in ("hd", "rhd", "recursive_doubling"):
        core = 1 << (n.bit_length() - 1)
        extra = n - core
        pairs = [(ranks[2*i], ranks[2*i + 1]) for i in range(extra)]
        active = [b for _, b in pairs] + list(ranks[2*extra:])
        if kind == "all_reduce":
            if algorithm == "recursive_doubling":
                middle = [[(src, active[i ^ (1 << bit)], payload_bytes) for i, src in enumerate(active)]
                          for bit in range(core.bit_length() - 1)]
            else:
                scatter = _scatter(active, [payload_bytes / core] * core, "hd")
                middle = scatter + _reverse(scatter)
            if not pairs:
                return middle
            # Split-vector fold: each partner reduces a half, one of the two
            # combines both halves, and returns the final vector after the core.
            exchange = [(a, b, payload_bytes / 2) for a, b in pairs]
            return [exchange + [(b, a, size) for a, b, size in exchange],
                    exchange] + middle + [[(b, a, payload_bytes) for a, b in pairs]]
        sizes = [2 * shard] * extra + [shard] * (core - extra)
        scatter = _scatter(active, sizes, "hd")
        if pairs:
            scatter = [[(a, b, payload_bytes) for a, b in pairs]] + scatter + [
                [(b, a, shard) for a, b in pairs]]
    elif algorithm in ("nhr", "nhr_dual_band_theory", "nhr_dual_band", "hierarchical"):
        groups = []
        if algorithm != "nhr" and (placement is None or any(r not in placement for r in ranks)):
            raise ValueError("Provide the group and server of every rank for topology-based algorithms")
        if algorithm in ("nhr", "nhr_dual_band_theory"):
            ordered = ranks if algorithm == "nhr" else sorted(ranks, key=lambda r: (*placement[r], r))
            width = isqrt(n - 1) + 1
            groups = [(divmod(i, width), r) for i, r in enumerate(ordered)]
            levels = ["ring", "ring"]
        else:
            by_group = defaultdict(lambda: defaultdict(list))
            for rank in ranks:
                group, server = placement[rank]
                by_group[group][server].append(rank)
            for g, (_, servers) in enumerate(sorted(by_group.items())):
                local = 0
                for s, (_, devices) in enumerate(sorted(servers.items())):
                    for d, rank in enumerate(sorted(devices)):
                        coord = (g, s, d) if algorithm == "hierarchical" else (g, local)
                        groups.append((coord, rank))
                        local += 1
            levels = ([choices["across_groups"], choices["within_group"], choices["within_server"]]
                      if algorithm == "hierarchical" else ["ring", "ring"])
        scatter = _grouped_scatter(groups, shard, levels)
    elif algorithm == "double_ring":
        forward = _scatter(ranks, [shard / 2] * n, "ring")
        backward = _scatter(list(reversed(ranks)), [shard / 2] * n, "ring")
        scatter = [left + right for left, right in zip(forward, backward)]
    else:
        scatter = _scatter(ranks, [shard] * n, algorithm)
    gather = _reverse(scatter)
    if kind == "reduce_scatter":
        return [step for step in scatter if step]
    if kind == "all_gather":
        return [step for step in gather if step]
    return [step for step in scatter + gather if step]
