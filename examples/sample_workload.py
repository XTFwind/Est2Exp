"""Synthetic, anonymous layers. Byte sizes are per layer per replica."""
import json
from pathlib import Path


def scenario(schedule):
    hardware = json.loads(Path(__file__).with_name("hardware.json").read_text(encoding="utf-8"))
    layers = [
        {"id": index, "gradient_bytes": size, "forward_bytes": 8192, "backward_bytes": 12288,
         "forward_s": 0.0001, "backward_s": 0.00018, "weight_gradient_s": 0.00007}
        for index, size in enumerate((81920, 131072, 98304, 65536))
    ]
    strategy = {"schedule": schedule, "pp": 2, "dp": 2, "microbatches": 4,
                "dp_sync": "all_reduce", "dp_algorithm": "ring", "rank_map": [0, 1, 2, 3]}
    if schedule == "vpp":
        strategy["virtual_stages"] = 2
    return {**hardware, "layers": layers, "strategy": strategy}
