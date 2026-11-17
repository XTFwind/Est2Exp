# Reading a communication estimate

The input is one iteration of the model-level DAG. Each collective specifies the full unpartitioned payload on one replica and the physical ranks that participate. A PP node specifies one sender, one receiver and the boundary payload for that microbatch. A zero-byte operation or a single-rank collective uses zero rounds.

## Choosing a collective

Set `strategy.dp_algorithm` to one of the following values. All support AllReduce, ReduceScatter and AllGather. Payloads divide into ideal equal shards, allowing fractional byte counts.

| Algorithm | Exchanges |
|---|---|
| `ring` | Each rank sends one shard to a neighbor in each of the N−1 rounds of ReduceScatter. AllGather reverses those exchanges. AllReduce runs both phases. |
| `double_ring` | Two counter-rotating rings run simultaneously, each carrying half the payload. Their bandwidth comes from their actual routes and shared endpoints. |
| `hd` | Recursive halving and doubling on a power-of-two number of ranks. Each peer exchange discards half the remaining reduce-scatter range. |
| `rhd` | Uses HD on the largest power-of-two core. Extra rank pairs exchange and reduce half vectors, send the other reduced half to the core, then recover the resulting vector after AllReduce. Standalone ReduceScatter folds full inputs, runs a weighted core ReduceScatter and returns the extra ranks' shards. AllGather reverses the scatter. |
| `recursive_doubling` | Exchanges the full vector with an exclusive-or partner in each AllReduce core round. Includes RHD folding at non-power-of-two sizes. Uses recursive shard exchange for standalone scatter and gather. |
| `nb` | Bruck's cyclic doubling exchanges progressively longer intervals. The last interval is truncated to the number of missing shards. |
| `nhr` | Forms near-square rows in the supplied rank order. Runs reduce-scatter within rows, then down columns. AllGather traverses the dimensions in reverse. |
| `nhr_dual_band_theory` | Forms the same near-square grid after sorting by affinity group, server and rank. Routes each row and column exchange through the configured topology. |
| `nhr_dual_band` | Uses actual affinity-group membership for the rows. Within-row and cross-row exchanges run on their respective routes. |
| `hierarchical` | Groups participants by device position, server and affinity group. Set `dp_level_algorithms` keys `within_server`, `within_group` and `across_groups` to `ring`, `nhr` or `mesh`. Defaults are ring, NHR and NHR. At one level, NHR uses doubling on power-of-two groups and truncated cyclic doubling otherwise. Mesh sends every peer its share in one round. |

Ragged groups have real participants at every step. A smaller group assigns extra column positions to its existing owners and sends the actual sizes of those shards. Levels with one participant contribute no network rounds. The estimate always derives step counts from the generated messages. For example, four ranks and a 400-byte full payload give 600 bytes sent per rank for Ring AllReduce in six rounds or HD in four. A full-vector recursive-doubling AllReduce needs two rounds and sends 800 bytes per rank.

## Elapsed time and traffic

Dependencies release a DAG node once its last predecessor finishes. All messages in one collective round may start together. The slowest message releases the next round. Independent nodes can overlap. Compute durations come from the DAG and occupy the slots specified by its schedule.

Each message uses its selected physical route. The simulator adds the propagation delays along the path, then streams data through all its links at a single end-to-end rate. Active transfers fairly share every directed resource they need. Rates increase together until a resource fills, and flows using that resource keep their assigned rates while the other flows continue to increase. The simulator reallocates rates on each arrival and completion. Reverse traffic has separate capacity. Direct fabric sends also share each source's outgoing and each destination's incoming capacity, set to the fastest incident logical link. Two simultaneous half-size ring transfers can therefore compete at a shared endpoint.

The report includes per-message routes and completion times in `nodes[id].steps`. `sent_bytes` counts each sender's application bytes once. `link_bytes` counts the bytes on every traversed link. Directional link, sender, physical stage, domain and anonymous-layer breakdowns are available. A shared gradient bucket attributes bytes in proportion to each layer's contribution. Its rounds are counted once. `rounds_by_domain` and `rounds` sum rounds over events even when those events overlap.

`latency_s` is the final completion time in this simulation. The critical path follows the predecessors that last released each event on the way to the final completion. Its elapsed durations include the effects of concurrent traffic. `critical_path_rounds` sums the communication rounds on that path. The reported timing mode identifies whether compute durations were supplied.
