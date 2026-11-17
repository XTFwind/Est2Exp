"""Deterministic routing on a two-tier folded Clos with an intra-group fabric."""
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256


class Router:
    def __init__(self, topology):
        self.devices = {n["rank"]: n for n in topology["nodes"] if n["type"] == "accelerator"}
        self.hops = {}
        self.leaves = defaultdict(list)
        self.spines = defaultdict(list)
        for edge in topology["links"]:
            a, b = edge["source"], edge["target"]
            for u, v in ((a, b), (b, a)):
                resources = [f"{u}>{v}"]
                if edge["type"] in ("intra_server", "intra_group"):
                    resources.extend((f"fabric_send:{u}", f"fabric_receive:{v}"))
                self.hops[u, v] = {"source": u, "target": v, "type": edge["type"],
                                   "bandwidth_bytes_s": edge["effective_bandwidth_bytes_s"],
                                   "latency_s": edge["latency_s"], "resources": tuple(resources)}
                if u.startswith("server:") and v.startswith("leaf:"):
                    self.leaves[u].append(v)
                elif u.startswith("leaf:") and v.startswith("spine:"):
                    self.spines[u].append(v)
        # Restrict the cache lifetime to this particular scenario.
        self.route = lru_cache(maxsize=65536)(self._route)

    def _route(self, src_rank, dst_rank):
        if src_rank == dst_rank:
            return []
        a, b = self.devices[src_rank], self.devices[dst_rank]
        if a["affinity_group_id"] == b["affinity_group_id"]:
            return [self.hops[a["id"], b["id"]]]
        src, dst = f"server:{a['server_id']}", f"server:{b['server_id']}"
        shared = sorted(set(self.leaves[src]).intersection(self.leaves[dst]))
        if shared:
            paths = [(leaf,) for leaf in shared]
        else:
            paths = [(left, spine, right) for left in sorted(self.leaves[src])
                     for right in sorted(self.leaves[dst])
                     for spine in sorted(set(self.spines[left]).intersection(self.spines[right]))]
        if not paths:
            raise ValueError(f"No inter-group route from device {src_rank} to {dst_rank}")
        # A stable flow-to-path assignment distributes equal-hop choices across the fabric.
        flow = int.from_bytes(sha256(f"{src_rank}/{dst_rank}".encode()).digest()[:8], "big")
        path = [a["id"], src, *paths[flow % len(paths)], dst, b["id"]]
        return [self.hops[u, v] for u, v in zip(path, path[1:])]
