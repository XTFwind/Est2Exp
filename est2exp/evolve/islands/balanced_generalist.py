"""Evolution instructions and initial program for the balanced island."""

PROMPT = """Evolve a ranking program for the balanced-generalist island. Weigh
pipeline ordering, synchronization, collective choice and topology together.
Explore tradeoffs suggested by the supplied programs and measurements. Favor
nominations with low estimated latency for this fixed workload and timing mode."""

SEED = '''def score_strategy(strategy):
    alternates = strategy["schedule"] == "1f1b"
    return alternates + (strategy["dp_algorithm"] == "hd") + 0.5 * (strategy["dp_sync"] == "all_reduce")
'''
