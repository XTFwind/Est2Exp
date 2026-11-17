"""Evolve strategy-ranking programs against a fixed anonymous workload."""

from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
from urllib.error import HTTPError

from ..dag.algorithms.dp import ALGORITHMS
from ..dag.algorithms.pp import SCHEDULES, SYNCS
from ..config.topology import fields, integer
from .islands import ISLANDS
from .proposer import ChatProposer
from .scoring import ScoringProgram

@dataclass
class Record:
    id: str
    island: str
    generation: int
    origin: str
    source: str
    parents: list
    inspirations: list
    nominations: list
    best_s: float
    mean_s: float

    def fitness(self):
        return self.best_s, self.mean_s, len(self.source), self.id


def _options(search, name, defaults, allowed):
    options = search.get(name, list(dict.fromkeys(defaults)))
    if (not isinstance(options, list) or not options
            or any(not isinstance(item, str) or item not in allowed for item in options)):
        raise ValueError(f"{name} must be a nonempty list drawn from {', '.join(allowed)}")
    return list(dict.fromkeys(options))


def run_evolution(scenario, output_dir, rounds=2):
    """Run four model-driven islands, saving measured nominations and the winning scenario."""
    from ..pipeline import run_scenario

    integer(rounds, "rounds", 1)
    search = scenario.get("search", {})
    fields(search, ("schedules", "dp_algorithms", "dp_syncs", "island_capacity",
                    "migration_interval", "nominations_per_program"), "search")
    capacity = integer(search.get("island_capacity", 8), "island_capacity", 1)
    interval = integer(search.get("migration_interval", 2), "migration_interval", 1)
    quota = integer(search.get("nominations_per_program", 2), "nominations_per_program", 1)
    base = scenario["strategy"]
    schedules = _options(search, "schedules", (base["schedule"], "gpipe", "1f1b"), SCHEDULES)
    algorithms = _options(search, "dp_algorithms", (base["dp_algorithm"], "ring", "hd"), ALGORITHMS)
    syncs = _options(search, "dp_syncs", (base["dp_sync"], "all_reduce", "rs_ag"), SYNCS)
    proposer = ChatProposer()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    active = {n["rank"]: n for n in scenario["hardware"]["nodes"] if n["type"] == "accelerator"}
    devices = [active[rank] for rank in base["rank_map"]]
    dp = base["dp"]
    stages = [devices[start:start + dp] for start in range(0, len(devices), dp)]
    direct_crossings = sum(a["affinity_group_id"] != b["affinity_group_id"]
                           for before, after in zip(stages, stages[1:])
                           for a, b in zip(before, after))
    wrap_crossings = sum(a["affinity_group_id"] != b["affinity_group_id"]
                         for a, b in zip(stages[-1], stages[0]))
    locality = {"server_count": len({n["server_id"] for n in devices}),
                "affinity_group_count": len({n["affinity_group_id"] for n in devices})}
    # DualPipe copies each parameter at both ends of its mirrored stage pair.
    mirrored_groups = [stage + opposite for stage, opposite in zip(stages, reversed(stages))]
    candidates = {}
    features = {}
    for schedule, algorithm, sync in product(schedules, algorithms, syncs):
        key = f"{schedule}/{algorithm}/{sync}"
        strategy = {**base, "schedule": schedule, "dp_algorithm": algorithm, "dp_sync": sync}
        if algorithm != "hierarchical":
            strategy.pop("dp_level_algorithms", None)
        if schedule != base["schedule"]:
            strategy.pop("virtual_stages", None)
        chunks = strategy.get("virtual_stages", 2 if schedule == "vpp" else 1)
        groups = mirrored_groups if schedule == "dual_pipe" else stages
        crossings = direct_crossings * chunks + (chunks - 1) * wrap_crossings if schedule == "vpp" else direct_crossings
        pending = base["microbatches"] if schedule == "gpipe" else min(base["microbatches"], base["pp"] * chunks)
        candidates[key] = strategy
        features[key] = {**locality, **{name: strategy[name] for name in
                         ("pp", "dp", "microbatches", "schedule", "dp_sync", "dp_algorithm")},
                         "virtual_stages": chunks, "cross_group_pp_boundaries": crossings,
                         "max_dp_group_size": max(map(len, groups)),
                         "max_dp_servers": max(len({n["server_id"] for n in group}) for group in groups),
                         "max_dp_affinity_groups": max(len({n["affinity_group_id"] for n in group}) for group in groups),
                         "pending_microbatch_hint": pending}
    population = {name: [] for name in ISLANDS}
    measurements, records, rejections, migrations = {}, [], [], []
    timing_mode = "compute_and_communication" if "forward_s" in scenario["layers"][0] else "communication_only"

    def evaluate(source, island, generation, parents, inspirations):
        program = ScoringProgram(source, (*SCHEDULES, *ALGORITHMS, *SYNCS))
        source = program.source
        ranking = sorted(candidates, key=lambda key: -program(features[key]))
        nominations = []
        for key in ranking:
            if key not in measurements:
                try:
                    _, report = run_scenario({**scenario, "strategy": candidates[key]})
                except ValueError:
                    measurements[key] = {"error": "Incompatible with this workload or topology"}
                else:
                    if report["timing_mode"] != timing_mode:
                        raise RuntimeError("Candidate changed the workload timing mode")
                    measurements[key] = {metric: report[metric] for metric in (
                        "latency_s", "sent_bytes_by_domain", "rounds_by_domain", "critical_path_rounds",
                    )}
            if "latency_s" in measurements[key]:
                nominations.append(key)
            if len(nominations) == quota:
                break
        if not nominations:
            reason = next(iter(measurements.values()))["error"]
            raise ValueError(f"No eligible candidates in the search space: {reason}")
        times = [measurements[key]["latency_s"] for key in nominations]
        origin = "seed" if generation == 0 else "model"
        record = Record(f"{generation}/{island}/{origin}", island, generation, origin,
                        source, parents, inspirations, nominations, min(times), sum(times) / len(times))
        records.append(record)
        members = population[island]
        # Equal source and nominations are the same ranking program in this fixed workload.
        if all(item.source != source for item in members):
            members.append(record)
        members.sort(key=Record.fitness)
        del members[capacity:]

    def save(status, generation, best=None):
        summary = {
            "status": status, "generation": generation, "timing_mode": timing_mode,
            "objective": "minimum_estimated_latency_s", "candidate_count": len(candidates),
            "measurements": measurements, "programs": [asdict(record) for record in records],
            "populations": {island: [record.id for record in members]
                            for island, members in population.items()},
            "migrations": migrations, "rejections": rejections, "best": best,
        }
        pending = out / "evolution.json.tmp"
        pending.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        pending.replace(out / "evolution.json")
        return summary

    for name, module in ISLANDS.items():
        evaluate(module.SEED.strip(), name, 0, [], [])
    save("running", 0)
    names = list(ISLANDS)
    for generation in range(1, rounds + 1):
        previous = {name: list(members) for name, members in population.items()}
        accepted = 0
        for index, (name, module) in enumerate(ISLANDS.items()):
            parent = previous[name][0]
            inspirations = previous[name][1:2] + previous[names[index - 1]][:1]
            evidence = {
                "island": name, "generation": generation, "objective": "minimize latency_s",
                "timing_mode": timing_mode, "parent": asdict(parent),
                "inspirations": [asdict(item) for item in inspirations],
                "parent_feedback": {k: measurements[k] for k in parent.nominations},
                "all_measured": dict(measurements),
                "candidate_features": features,
            }
            try:
                source = proposer.propose(module.PROMPT, evidence)
                evaluate(source, name, generation, [parent.id], [item.id for item in inspirations])
            except (OSError, ValueError, SyntaxError, LookupError, TypeError, ArithmeticError) as error:
                if isinstance(error, HTTPError):
                    reason = f"Model service returned HTTP {error.code}"
                elif isinstance(error, OSError):
                    reason = "Model service connection failed"
                else:
                    reason = "Invalid model proposal or scoring expression"
                rejections.append({"island": name, "generation": generation,
                                   "parent": parent.id, "reason": reason})
            else:
                accepted += 1
        if not accepted:
            save("failed", generation)
            raise RuntimeError(f"All model proposals failed in generation {generation}: {rejections[-1]['reason']}")
        if generation % interval == 0:
            # A snapshot prevents a migrant from moving twice in the same round.
            champions = {name: members[0] for name, members in population.items()}
            for index, name in enumerate(names):
                target = names[(index + 1) % len(names)]
                donor = champions[name]
                migrant = Record(f"{generation}/{target}/migration", target, generation, "migration",
                                 donor.source, [donor.id], [], list(donor.nominations), donor.best_s, donor.mean_s)
                records.append(migrant)
                members = population[target]
                if all(item.source != migrant.source for item in members):
                    members.append(migrant)
                    members.sort(key=Record.fitness)
                    del members[capacity:]
                migrations.append({"generation": generation, "from": name, "to": target,
                                   "parent": donor.id, "record": migrant.id,
                                   "retained": migrant in members})
        save("running", generation)
    winner = min((key for key in measurements if "latency_s" in measurements[key]),
                 key=lambda key: (measurements[key]["latency_s"], key))
    winning_scenario = {**scenario, "strategy": candidates[winner]}
    _, replay = run_scenario(winning_scenario, out / "best")
    if replay["latency_s"] != measurements[winner]["latency_s"] or replay["timing_mode"] != timing_mode:
        raise RuntimeError("Replayed estimate differs from the winning nomination")
    return save("complete", rounds, {"candidate": winner, "strategy": candidates[winner],
                                     "latency_s": replay["latency_s"], "directory": "best"})
