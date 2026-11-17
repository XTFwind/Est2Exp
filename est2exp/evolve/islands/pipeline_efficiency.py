"""Evolution instructions and initial program for the pipeline-efficiency island."""

PROMPT = """Evolve a ranking program for the pipeline-efficiency island. Consider
pipeline depth, microbatch count, virtual stages and the order of forward and
backward work. Look for overlap opportunities. Use estimator feedback to revise
schedule preferences when nominal bubble heuristics disagree with observations."""

SEED = '''def score_strategy(strategy):
    interleaved = strategy["schedule"] == "vpp" or strategy["schedule"] == "dual_pipe"
    return (strategy["schedule"] == "1f1b") + 1.5 * (interleaved and strategy["microbatches"] >= strategy["pp"])
'''
