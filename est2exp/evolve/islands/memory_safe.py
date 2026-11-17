"""Evolution instructions and initial program for the memory-pressure island."""

PROMPT = """Evolve a ranking program for the memory-pressure island. Prefer schedules
that release pending work early. Consider pipeline depth, microbatch count and the
choice of synchronization phases. Use these as heuristics only. The estimator
measures time and does not predict capacity or out-of-memory failures. Revise
the parent using its measured nominations and the other programs supplied."""

SEED = '''def score_strategy(strategy):
    return -strategy["pending_microbatch_hint"] + 0.1 * (strategy["dp_sync"] == "rs_ag")
'''
