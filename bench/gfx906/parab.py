#!/usr/bin/env python3
"""Parallel-throughput probe: N concurrent chat completions with distinct prompts (no prompt-cache
reuse between them), thinking off at sampling. Prints per-request and aggregate tok/s.

    python3 bench/gfx906/parab.py 4 [--url http://127.0.0.1:8080] [--model default] [--max-tokens 600]

The first run after a restart pays a ~40 s warm-up chunk and reads 24-31 tok/s; re-run before
comparing. 2026-11-01/02 on 2x MI50 + IQ3_S with the RP stack: 44.1 (N=2), 44.9 (N=3), 45.7 (N=4)
system tok/s - flat from N=1 (38.6 solo), because decode on this box is instruction-rate bound.
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SUBJECTS = [
    "a lighthouse keeper on a stormy night",
    "an astronaut stranded on a space station",
    "a desert caravan crossing badlands",
    "a harbor tug captain during a fog bank",
]
PROMPT = ("Write a detailed 300-word short story about %s. "
          "Continue until you have written roughly 300 words of story.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("n", type=int)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="default")
    ap.add_argument("--max-tokens", type=int, default=600)
    a = ap.parse_args()
    url = a.url.rstrip("/") + "/v1/chat/completions"

    def one(i: int):
        t0 = time.time()
        body = {"model": a.model,
                "messages": [{"role": "user", "content": PROMPT % SUBJECTS[i % len(SUBJECTS)]}],
                "max_tokens": a.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=600))
        wall = time.time() - t0
        tok = d.get("usage", {}).get("completion_tokens") or 0
        return i, wall, tok

    with ThreadPoolExecutor(max_workers=a.n) as ex:
        results = list(ex.map(one, range(a.n)))

    total_tok = sum(t for _, _, t in results)
    wall_max = max(w for _, w, _ in results)
    print("N=%d  wall_max=%.1fs  total_tokens=%d  system=%.1f tok/s" %
          (a.n, wall_max, total_tok, total_tok / wall_max))
    for i, w, t in sorted(results, key=lambda r: r[0]):
        print("  req%d: %d tok in %.1fs = %.1f tok/s" % (i, t, w, t / w))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
