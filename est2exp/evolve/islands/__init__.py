"""The order of these islands defines the migration ring."""

from . import balanced_generalist, memory_safe, pipeline_efficiency, topology_affinity

ISLANDS = {
    "memory_safe": memory_safe,
    "topology_affinity": topology_affinity,
    "pipeline_efficiency": pipeline_efficiency,
    "balanced_generalist": balanced_generalist,
}
