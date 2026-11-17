# Est2Exp

Est2Exp builds a complete training-iteration DAG from anonymous layers and estimates the latency of a DP and PP strategy on a configurable network. It runs on Python 3.10+ using the standard library. The examples use synthetic messages, timings and hardware.

## Run a complete example

From the project directory, run

```powershell
python -m est2exp run examples/gpipe.py --output runs/gpipe
```

Open `runs/gpipe/dag.html` in a browser. Every event and dependency is drawn. Use **Fit full graph** for an overview, **Actual size** to read labels and click an event to inspect its timings and transmissions. Drag the background to pan. The critical path switch highlights the events that determine the estimate.

| Schedule | Input | Complete offline graph |
|---|---|---|
| GPipe | [gpipe.py](examples/gpipe.py) | [GPipe DAG](examples/graphs/gpipe/dag.html) |
| Backward-ready-first 1F1B | [1f1b.py](examples/1f1b.py) | [1F1B DAG](examples/graphs/1f1b/dag.html) |
| Interleaved virtual pipeline | [vpp.py](examples/vpp.py) | [VPP DAG](examples/graphs/vpp/dag.html) |
| Bidirectional DualPipe | [dual_pipe.py](examples/dual_pipe.py) | [DualPipe DAG](examples/graphs/dual_pipe/dag.html) |

The `run` command writes `dag.json`, `dag.html`, `report.json`, `topology.json` and a replayable `scenario.json`. Use `python -m est2exp run runs/gpipe/scenario.json --output runs/replay` to reproduce a run. The HTML and JSON files under `examples/graphs/` can be opened locally after download.

## Configure the workload and hardware

Start with [hardware.json](examples/hardware.json) and [sample_workload.py](examples/sample_workload.py). Configuration accepts JSON objects and trusted Python files defining `CONFIG` or `get_config()`. Each layer has a unique integer ID. List order defines model order.

| Layer field | Meaning |
|---|---|
| `gradient_bytes` | Full gradient payload on one replica before collective partitioning. Synchronization combines the gradients of the layers owned by a stage after microbatch accumulation. |
| `forward_bytes`, `backward_bytes` | Bytes crossing the boundary after this layer for one microbatch in the indicated direction. An internal boundary on one device generates no network traffic. |
| `forward_s`, `backward_s` | Elapsed time for one microbatch on the assigned device. Supply both for every layer for compute-and-communication timing. Omit both throughout for communication-only timing. |
| `weight_gradient_s` | Optional part of `backward_s` consumed by weight-gradient work when the schedule separates backward phases. |

`pp` and `dp` specify the number of physical stages and replicas. `microbatches` is the number per replica per iteration. `rank_map` lists physical devices in stage-major, replica-minor order. `vpp` uses `virtual_stages` chunks per physical stage. The ordered layer list is partitioned across the chunks. Each microbatch follows the same logical layer order.

Hardware membership is explicit. Set the default device compute rate in TFLOPS, memory in decimal GB and memory bandwidth in bytes/s. Set individual values in `topology.device_overrides`, keyed by rank. Servers may have their own host compute rate. The cluster total counts accelerator compute. Layer times remain the supplied observed or synthetic durations.

Use `network.intra_server` for logical device pairs in a server and `intra_group` for pairs across servers in an affinity group. Traffic between groups uses device access links, server uplinks and the two-tier folded Clos. `clos` sets leaf, spine and port counts. Each server can select its uplinks with `leaf_ids`. Set `server_leaf`, and optionally `device_server` and `leaf_spine`, to calibrate each link class. Override a single existing connection using `topology.link_overrides` with its two endpoints. Each full-duplex link has bandwidth per direction. Effective bytes/s equals `bandwidth_gbps × bandwidth_unit_bits × efficiency / 8`; the default unit is decimal Gbit/s. Latencies are seconds.

To build only the hardware data, run

```powershell
python -m est2exp topology examples/hardware.json --output runs/topology.json
```

A scenario can reference the exported hardware with a `topology_file` path relative to the scenario file. The same topology is used for rank placement, routing and timing.

## Strategy evaluation and evolution

Choose `strategy.dp_sync` as `all_reduce` or `rs_ag`. The communication ledger separates application payload sent by devices from link traffic accumulated over multiple hops. Collective rounds describe the algorithm. Summing overlapping event durations does not give the iteration latency.

| `dp_algorithm` | Round schedule and domain |
|---|---|
| `ring`, `double_ring` | Ring reduce-scatter and all-gather. Double ring sends half the tensor in each direction. Both directions count toward the total bytes sent. |
| `hd`, `rhd` | Recursive halving and doubling; HD uses a power-of-two group. RHD folds extra ranks into a power-of-two core and returns the results. |
| `recursive_doubling` | Full-vector exchanges for AllReduce, partition exchanges for the individual scatter/gather operations. |
| `nb` | Cyclic doubling with a truncated final exchange for irregular group sizes. |
| `nhr` | Row and column exchanges on a near-square grid in communication-group order. Incomplete rows use their actual members. |
| `nhr_dual_band_theory`, `nhr_dual_band` | The first applies a near-square layout to locality-ordered ranks. The second takes row membership directly from physical affinity groups. Both price actual routed messages with the configured links. |
| `hierarchical` | Three levels: within a server, between servers in a group, and across groups. Configure `dp_level_algorithms` with keys `within_server`, `within_group`, `across_groups`. Each supports `ring`, `nhr`, `recursive_doubling` or `mesh`. |

RS and AG each take the full tensor byte size as input. Equal partitioning permits fractional analytical byte counts. A one-device group or a zero-byte message has no network rounds. The shortest eligible Clos path is chosen deterministically per device pair. The simulation adds hop startup delays and shares directional bandwidth fairly between simultaneous transfers. Intra-group sends also share device send and receive capacity. Available endpoint bandwidth can therefore limit both channels of a double ring. These are analytical latencies, without packet queues or calibrated kernel overlap.

The report gives per-event start and finish times, each collective round with its directed messages, and totals by layer, stage, rank and domain. Stage-level collectives allocate sent bytes to layers by their gradient payloads. The sum of all event rounds and rounds on the latency-critical path are reported separately.

DualPipe requires even `pp` and an even microbatch count of at least twice `pp`. The simulator represents paired work with an aggregate overlap window and holds sends until the paired work completes. Without an explicit weight-gradient timing, the full backward time is assigned to the backward event and the deferred weight-gradient node retains its dependencies. VPP uses at least two virtual chunks per device and groups microbatches in multiples of `pp`. See the [DualPipe reference schedule](https://github.com/deepseek-ai/DualPipe) and [interleaved pipeline scheduling reference](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.pipeline_parallel.schedules.html).

Each island has its English instructions and initial scoring program under [evolve/islands](est2exp/evolve/islands). Configure a chat-completions service through the environment:

```powershell
$env:EST2EXP_LLM_BASE_URL = "http://localhost:8000/v1"
$env:EST2EXP_LLM_MODEL = "your-service-model"
python -m est2exp evolve examples/gpipe.py --rounds 4 --output runs/search
```

Set `EST2EXP_LLM_API_KEY` in the environment if the service requires authentication. A proposal receives the island instructions, parent and inspiration programs, public strategy features and timing feedback. Programs nominate strategies. The simulator supplies the latency used to rank them. The optional `search` object sets `schedules`, `dp_algorithms`, `dp_syncs`, `nominations_per_program`, `island_capacity` and `migration_interval`. Comparisons use the same devices, workload and timing mode. `evolution.json` records lineage and results. `best/` contains the winning replay configuration and full DAG.

## Modules and verification

| Module | Responsibility |
|---|---|
| [config](est2exp/config) | Load public settings and construct the hardware topology. |
| [dag](est2exp/dag) | Define DP communication rounds and PP schedules in [algorithms](est2exp/dag/algorithms), build dependencies and render the complete training iteration. |
| [estimator](est2exp/estimator) | Map transfers to the network, account for shared resources and estimate completion times. |
| [evolve](est2exp/evolve) | Generate candidates with a model service and select by estimated latency. |
| [experience_bank](est2exp/experience_bank) | Skills to [extract deployment rules](est2exp/experience_bank/extract-deployment-rules.md), [classify experience families and scenario motifs](est2exp/experience_bank/classify-experience-families-and-scenario-motifs.md), and [resolve deployment rule conflicts](est2exp/experience_bank/resolve-deployment-rule-conflicts.md). |

Run `python -B -m unittest discover -s tests -v` for the regression suite. The model-service tests run against a local controlled HTTP response. To inspect the actual offline browser output, install Playwright for Node.js and run `node tests/check_graph.cjs examples/graphs/gpipe examples/graphs/1f1b examples/graphs/vpp examples/graphs/dual_pipe`. Set `BROWSER_PATH` to a Chromium executable and `PLAYWRIGHT_MODULE` to a Playwright installation if needed.
