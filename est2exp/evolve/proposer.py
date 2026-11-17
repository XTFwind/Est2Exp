"""Model proposals over a user-configured chat completions service."""

import json
import os
import urllib.request
from urllib.parse import urlsplit

from .scoring import FEATURES


class ChatProposer:
    def __init__(self):
        base = os.getenv("EST2EXP_LLM_BASE_URL", "").rstrip("/")
        self.model = os.getenv("EST2EXP_LLM_MODEL")
        if not base or not self.model:
            raise ValueError("Set EST2EXP_LLM_BASE_URL and EST2EXP_LLM_MODEL to run evolution")
        url = urlsplit(base)
        if url.scheme not in ("http", "https") or not url.netloc:
            raise ValueError("EST2EXP_LLM_BASE_URL must be an HTTP or HTTPS API base URL")
        self.endpoint = base + "/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        key = os.getenv("EST2EXP_LLM_API_KEY")
        self.private_values = [value for value in (base, key) if value]
        if key:
            self.headers["Authorization"] = "Bearer " + key

    def propose(self, guidance, evidence):
        contract = (
            "\nReturn a JSON object with one string field named source, containing a Python "
            "function score_strategy(strategy). Higher scores nominate a strategy first. "
            "Use only local assignments followed by one return. Expressions may read "
            "strategy[key], numeric and string literals, use +, -, *, /, comparisons, "
            "and/or/not, conditional expressions, abs, min and max. "
            "Use only public feature and strategy names as string literals. "
            "pending_microbatch_hint estimates forward work held for backward; it is not a measured peak. "
            "max_dp_servers and max_dp_affinity_groups count destinations in the widest DP group. "
            "cross_group_pp_boundaries counts inter-group adjacent stage boundaries across replicas. "
            "Measured rounds_by_domain sums collective rounds and point-to-point steps over events; "
            "parallel events can overlap. sent_bytes_by_domain counts application sends. "
            "The available keys are " + ", ".join(FEATURES) + "."
        )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": guidance + contract},
                {"role": "user", "content": json.dumps(evidence, allow_nan=False)},
            ],
        }
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(body).encode("utf-8"), headers=self.headers, method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            content = json.load(response)["choices"][0]["message"]["content"]
        source = json.loads(content)["source"]
        if not isinstance(source, str) or not source.strip():
            raise ValueError("The model response must contain a nonempty source string")
        if any(value in source for value in self.private_values):
            raise ValueError("The model returned private connection details")
        return source.strip()
