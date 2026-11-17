"""Build layer-level compute, communication, and resource-order dependencies."""

from collections import defaultdict

from .algorithms.pp import stage_slots, validate_schedule


def build_dag(scenario):
    strategy, layers = scenario["strategy"], scenario["layers"]
    validate_schedule(strategy, len(layers))
    pp, dp, batches = (strategy[k] for k in ("pp", "dp", "microbatches"))
    schedule = strategy["schedule"]
    dual = schedule == "dual_pipe"
    virtual = strategy.get("virtual_stages", 2 if schedule == "vpp" else 1)
    mapping = strategy["rank_map"]
    num_chunks = pp * virtual
    size, extra = divmod(len(layers), num_chunks)
    chunks = []
    start = 0
    for chunk in range(num_chunks):
        end = start + size + int(chunk < extra)
        chunks.append(layers[start:end])
        start = end

    nodes = []
    edges = {}

    def connect(source, target, kind):
        edges.setdefault((source, target, kind), {"source": source, "target": target, "kind": kind})

    # Ports point to event IDs: paired work starts and completes as a single
    # slot, but both sets of layer nodes retain their own durations and edges.
    stages = stage_slots(strategy)
    entry, exit_port = {}, {}
    gradient_ready = defaultdict(list)
    for replica in range(dp):
        for stage, slots in enumerate(stages):
            rank = mapping[stage * dp + replica]
            previous = None
            for slot_index, slot in enumerate(slots):
                metadata = {"stage": stage, "replica": replica, "ranks": [rank]}
                overlap = len(slot) == 2
                if overlap:
                    stem = f"slot:r{replica}:s{stage}:{slot_index}"
                    begin, end = stem + ":start", stem + ":end"
                    for barrier in (begin, end):
                        nodes.append({"id": barrier, "kind": "schedule_barrier", "duration_s": 0.0, **metadata})
                    if previous is not None:
                        connect(previous, begin, "stage_order")
                for work in slot:
                    kind, chunk, mb, deferred = work
                    key = (replica, chunk, mb, kind)
                    ordered = chunks[chunk] if kind == "forward" else reversed(chunks[chunk])
                    first = last = None
                    for layer in ordered:
                        lid = layer["id"]
                        node_id = f"{kind}:r{replica}:m{mb}:l{lid}"
                        duration = layer.get(kind + "_s", 0.0)
                        if deferred:
                            duration -= layer.get("weight_gradient_s", 0.0)
                        compute = {"id": node_id, "kind": kind, "layer": lid, "microbatch": mb,
                                   "chunk": chunk, "duration_s": duration, **metadata}
                        if dual:
                            compute["pipeline_direction"] = "increasing_stage" if mb < batches // 2 else "decreasing_stage"
                        nodes.append(compute)
                        if kind == "backward":
                            connect(f"forward:r{replica}:m{mb}:l{lid}", node_id, "activation_ready")
                        elif kind == "weight_gradient":
                            connect(f"backward:r{replica}:m{mb}:l{lid}", node_id, "weight_input_ready")
                        if last is None:
                            first = node_id
                        else:
                            connect(last, node_id, "layer_order")
                        last = node_id
                    if overlap:
                        connect(begin, first, "paired_compute")
                        connect(last, end, "paired_compute")
                        entry[key], exit_port[key] = begin, end
                    else:
                        if previous is not None:
                            connect(previous, first, "stage_order")
                        entry[key], exit_port[key] = first, last
                    if kind == "weight_gradient" or (kind == "backward" and not deferred):
                        gradient_ready[chunk].append(exit_port[key])
                previous = end if overlap else last

    # Each P2P operation is a directed message. Both byte counts describe the
    # boundary after the last logical layer in the lower-numbered chunk.
    for replica in range(dp):
        for mb in range(batches):
            mirrored = dual and mb >= batches // 2
            for chunk in range(num_chunks):
                prefix = (replica, chunk, mb)
                connect(exit_port[(*prefix, "forward")], entry[(*prefix, "backward")], "activation_ready")
                if (*prefix, "weight_gradient") in entry:
                    connect(exit_port[(*prefix, "backward")], entry[(*prefix, "weight_gradient")], "weight_input_ready")
            for lower in range(num_chunks - 1):
                boundary = chunks[lower][-1]
                for direction, src, dst in (("forward", lower, lower + 1), ("backward", lower + 1, lower)):
                    src_stage, dst_stage = src % pp, dst % pp
                    if mirrored:
                        src_stage, dst_stage = pp - 1 - src_stage, pp - 1 - dst_stage
                    message = f"pp:{direction}:r{replica}:m{mb}:boundary{lower}"
                    send = {"id": message, "kind": "pp_send", "replica": replica, "stage": src_stage,
                            "layer": boundary["id"], "microbatch": mb, "direction": direction,
                            "source_chunk": src, "target_chunk": dst, "bytes": boundary[direction + "_bytes"],
                            "ranks": [mapping[src_stage * dp + replica], mapping[dst_stage * dp + replica]]}
                    if dual:
                        send["pipeline_direction"] = "decreasing_stage" if mirrored else "increasing_stage"
                    nodes.append(send)
                    connect(exit_port[replica, src, mb, direction], message, "send_ready")
                    connect(message, entry[replica, dst, mb, direction], "receive_ready")

    # Accumulate every microbatch before stage-level synchronization. DualPipe
    # has two physical copies per replica of each logical chunk's parameters.
    if dp > 1 or dual:
        kinds = ("all_reduce",) if strategy["dp_sync"] == "all_reduce" else ("reduce_scatter", "all_gather")
        for stage in range(pp):
            chunk_ids = list(range(stage, num_chunks, pp))
            layer_bytes = {str(layer["id"]): layer["gradient_bytes"] for c in chunk_ids for layer in chunks[c]}
            holders = [stage, pp - 1 - stage] if dual else [stage]
            ranks = [mapping[s * dp + r] for s in holders for r in range(dp)]
            deps = [event for c in chunk_ids for event in gradient_ready[c]]
            for k, kind in enumerate(kinds):
                node_id = f"dp:{kind}:s{stage}"
                nodes.append({"id": node_id, "kind": kind, "stage": stage, "ranks": ranks,
                              "bytes": sum(layer_bytes.values()), "layer_bytes": layer_bytes})
                for dep in deps:
                    connect(dep, node_id, "gradients_ready" if k == 0 else "shards_ready")
                deps = [node_id]

    timed = "forward_s" in layers[0]
    assumptions = ["Compute times apply to one anonymous layer and one microbatch on its assigned device."]
    if dual:
        assumptions.extend([
            "DualPipe uses two mirrored copies of each partition and synchronizes their accumulated gradients.",
            "Paired forward/backward work uses ideal aggregate overlap. Payloads become ready at slot completion; kernel and runtime send/receive launch timing are not modeled.",
            "Deferred weight-gradient work uses weight_gradient_s from backward_s. With no split supplied its duration is zero.",
        ])
    return {"nodes": nodes, "edges": list(edges.values()),
            "timing_mode": "compute_and_communication" if timed else "communication_only",
            "assumptions": assumptions}
