"""Pipeline schedules expressed as ordered compute slots on each physical stage."""
from collections import deque
from typing import NamedTuple


SCHEDULES = ("gpipe", "1f1b", "vpp", "dual_pipe")
SYNCS = ("all_reduce", "rs_ag")


class Work(NamedTuple):
    kind: str
    chunk: int
    microbatch: int
    deferred: bool = False


def validate_schedule(strategy, layer_count):
    """Validate scheduling constraints on a normalized strategy."""
    pp, batches, schedule = (strategy[k] for k in ("pp", "microbatches", "schedule"))
    if schedule not in SCHEDULES or strategy["dp_sync"] not in SYNCS:
        raise ValueError("Unknown PP schedule or DP synchronization strategy")
    chunks = strategy.get("virtual_stages", 2 if schedule == "vpp" else 1)
    if type(chunks) is not int or chunks < 1 or layer_count < pp * chunks:
        raise ValueError("Each physical or virtual pipeline stage needs at least one layer")
    if schedule == "vpp":
        if pp < 2 or chunks < 2 or batches % pp:
            raise ValueError("VPP requires multiple stages, at least two chunks per stage, and microbatches divisible by pp")
    elif chunks != 1:
        raise ValueError("Only VPP uses virtual_stages greater than one")
    if schedule == "dual_pipe" and (pp % 2 or batches % 2 or batches < 2 * pp):
        raise ValueError("DualPipe requires even pp and even microbatches >= 2 * pp")


def _ready_first(pp, batches):
    slots = [[] for _ in range(pp)]
    completed = set()
    forward, backward = [0] * pp, [0] * pp
    while len(completed) < 2 * pp * batches:
        choices = []
        for stage in range(pp):
            mb = backward[stage]
            if (mb < batches and Work("forward", stage, mb) in completed and
                    (stage == pp - 1 or Work("backward", stage + 1, mb) in completed)):
                choices.append(Work("backward", stage, mb))
                backward[stage] += 1
            else:
                mb = forward[stage]
                if mb < batches and (stage == 0 or Work("forward", stage - 1, mb) in completed):
                    choices.append(Work("forward", stage, mb))
                    forward[stage] += 1
        for work in choices:
            slots[work.chunk].append((work,))
        completed.update(choices)
    return slots


def _dual_slots(stage, pp, batches):
    """Bidirectional eight-phase schedule, including the deferred FIFO weight drain.

    Reference: DeepSeek DualPipe, dualpipe/dualpipe.py, DualPipe.step.
    The two halves of the batch traverse the same logical chunks on mirror ranks.
    """
    half = pp // 2
    distance = min(stage, pp - stage - 1)
    outer = half - distance - 1
    counters = {kind: [0, 0] for kind in ("forward", "backward")}
    pending = deque()
    result = []

    def next_work(kind, relative_lane, defer=False):
        lane = relative_lane ^ int(stage >= half)
        mb = counters[kind][lane] + lane * (batches // 2)
        counters[kind][lane] += 1
        chunk = stage if lane == 0 else pp - stage - 1
        if defer:
            pending.append(Work("weight_gradient", chunk, mb))
        return Work(kind, chunk, mb, defer)

    for _ in range(2 * outer):
        result.append((next_work("forward", 0),))
    for _ in range(distance + 1):
        result.append((next_work("forward", 0),))
        result.append((next_work("forward", 1),))
    for _ in range(outer):
        result.append((next_work("backward", 1, True),))
        result.append((pending.popleft(),))
        result.append((next_work("forward", 1),))
    for index in range(batches // 2 - pp + distance + 1):
        f, b = next_work("forward", 0), next_work("backward", 1)
        if index == 0 and distance == half - 1:
            result.extend([(f,), (b,)])
        else:
            result.append((f, b))
        result.append((next_work("forward", 1), next_work("backward", 0)))
    for _ in range(outer):
        result.append((next_work("backward", 1),))
        result.append((next_work("forward", 1), next_work("backward", 0)))
    count = distance + 1
    for index in range(count):
        defer_first = index > count // 2 or (index == count // 2 and distance % 2 == 1)
        result.append((next_work("backward", 1, defer_first),))
        result.append((next_work("backward", 0, index >= count // 2),))
    for _ in range(outer):
        result.append((pending.popleft(),))
        result.append((next_work("backward", 0, True),))
    result.extend((weight,) for weight in pending)
    return result


def stage_slots(strategy):
    """Return one list of slots per physical stage. A pair shares one overlap window."""
    pp, batches, schedule = (strategy[k] for k in ("pp", "microbatches", "schedule"))
    if schedule == "1f1b":
        return _ready_first(pp, batches)
    if schedule == "dual_pipe":
        return [_dual_slots(stage, pp, batches) for stage in range(pp)]
    if schedule == "gpipe":
        return [[(Work(kind, stage, mb),) for kind in ("forward", "backward")
                 for mb in range(batches)] for stage in range(pp)]

    # Classic interleaved 1F1B. Each group of pp microbatches traverses a
    # virtual chunk before the device moves to its next chunk in the round.
    virtual = strategy.get("virtual_stages", 2)
    result = []
    for stage in range(pp):
        forward, backward = [], []
        for start in range(0, batches, pp):
            for chunk in range(virtual):
                for mb in range(start, start + pp):
                    forward.append(Work("forward", chunk * pp + stage, mb))
                    backward.append(Work("backward", (virtual - chunk - 1) * pp + stage, mb))
        warmup = min(len(forward), 2 * (pp - stage - 1) + (virtual - 1) * pp)
        order = list(forward[:warmup])
        for f, b in zip(forward[warmup:], backward):
            order.extend((f, b))
        order.extend(backward[len(forward) - warmup:])
        result.append([(work,) for work in order])
    return result
