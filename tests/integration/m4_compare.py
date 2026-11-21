"""Compare fresh-engine T10 outputs; emit exact differences, never hide drift.

Usage: python m4_compare.py CONNECTOR BASELINE NATIVE
Native reuse must be bitwise identical in final hidden states and sampled
logprobs. Recompute must retain greedy tokens and cosine >= 0.999 per step.
Full-vocabulary logit KL is not available from these top-5 logprob captures.
"""

import json
import math
import sys
from pathlib import Path


def compare(connector, baseline, native):
    result = {"kind": connector["kind"], "stats": connector["stats"], "requests": []}
    for i, output in enumerate(connector["outputs"]):
        assert output == native["outputs"][i], f"request {i}: native outputs differ"
        assert connector["hidden"][i] == native["hidden"][i], f"request {i}: native hidden differs"
        assert output["tokens"] == baseline["outputs"][i]["tokens"], "recompute tokens differ"
        pairs = list(zip(connector["hidden"][i], baseline["hidden"][i], strict=True))
        assert pairs, "hidden-state instrumentation must be active"
        cosine = []
        for x, y in pairs:
            norm = math.sqrt(sum(a * a for a in x) * sum(b * b for b in y))
            cosine.append(sum(a * b for a, b in zip(x, y, strict=True)) / norm)
        assert min(cosine) >= 0.999, f"recompute hidden cosine {min(cosine)}"
        result["requests"].append(
            {
                "native_exact": True,
                "recompute_tokens_exact": True,
                "recompute_hidden_max_abs": max(abs(a - b) for x, y in pairs for a, b in zip(x, y)),
                "recompute_hidden_min_cosine": min(cosine),
            }
        )
    return result


if __name__ == "__main__":
    data = [json.loads(Path(path).read_text()) for path in sys.argv[1:]]
    print(json.dumps(compare(*data), indent=2))
