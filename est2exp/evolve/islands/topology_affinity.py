"""Evolution instructions and initial program for the locality island."""

PROMPT = """Evolve a ranking program for the topology-affinity island. Consider the
available servers and locality groups, the collective family, and the group size.
Test whether hierarchical or flat exchanges suit the public workload. Adapt the
parent using measured nominations. Treat locality rules as hypotheses."""

SEED = '''def score_strategy(strategy):
    hierarchical = strategy["dp_algorithm"] == "hierarchical" or strategy["dp_algorithm"] == "nhr_dual_band"
    remote = strategy["max_dp_servers"] > 1
    ring_local = strategy["dp_algorithm"] == "ring" and not remote
    return hierarchical * remote * strategy["max_dp_affinity_groups"] + ring_local - 0.1 * strategy["cross_group_pp_boundaries"]
'''
