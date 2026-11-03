#!/usr/bin/env python3
"""Prefill + decode benchmark for a running strata server at a given context size.

Runs on the host (stdlib only). One request per run with a fresh random-word prompt, so the
prompt-cache and the n-gram paths cannot reuse a previous run's prefix. Writes one JSON record.

    python3 bench/gfx906/context-bench.py --context 65536 --out /tmp/ab-65536-base.json \
        [--url http://127.0.0.1:8080] [--model default] [--max-new 128]

pp = timings.prompt_per_second (tokens/s reading the prompt), tg = timings.predicted_per_second
(the whole generation window). temperature 0 + enable_thinking false keeps the comparison between
context sizes honest: same sampling, no thinking spilling over into the measured tokens.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request

WORDS = """system memory kernel thread buffer stream packet router expert layer weight token graph
cache page table block shard index scalar matrix column plane window slot ring queue device driver
signal clock domain fabric bridge channel sample filter window record column journal shard replica
leader witness quorum ledger anchor beacon harbor lantern meadow timber granite copper silver amber
cobalt walnut cedar basalt glacier canyon prairie thicket orchard pasture village hamlet market""".split()


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(url: str, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=8) as r:
                if json.loads(r.read()).get("loaded"):
                    return
        except Exception:
            pass
        time.sleep(5)
    raise SystemExit("server not ready in time")


def ask_words(url: str, model: str, words: list, max_new: int, timeout: float) -> dict:
    return post(url, {"model": model,
                      "messages": [{"role": "user", "content": " ".join(words) + "\nAnswer briefly."}],
                      "max_tokens": max_new, "temperature": 0,
                      "chat_template_kwargs": {"enable_thinking": False}}, timeout)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="default")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ready-s", type=float, default=2400)
    ap.add_argument("--request-timeout", type=float, default=5400)
    a = ap.parse_args()

    wait_ready(a.url, a.ready_s)
    # Calibrate on this server: tokens for the template alone, then tokens per random word.
    base = ask_words(a.url, a.model, [], 1, 600)["usage"]["prompt_tokens"]
    rng = random.Random(a.context)                       # a different prompt per context size
    probe = [rng.choice(WORDS) for _ in range(120)]
    per_word = (ask_words(a.url, a.model, probe, 1, 600)["usage"]["prompt_tokens"] - base) / 120.0
    target = a.context - base - a.max_new - 64          # stay strictly under the window
    n_words = int((target - 32) / per_word)
    words = [rng.choice(WORDS) for _ in range(n_words)]

    d = ask_words(a.url, a.model, words, a.max_new, a.request_timeout)
    t = d.get("timings", {})
    m = d["choices"][0]["message"]
    result = {"context": a.context, "prompt_tokens": d["usage"]["prompt_tokens"],
              "output_tokens": d["usage"]["completion_tokens"], "finish": d["choices"][0]["finish_reason"],
              "pp_tok_s": t.get("prompt_per_second"), "prompt_ms": t.get("prompt_ms"),
              "tg_tok_s": t.get("predicted_per_second"), "decode_ms": t.get("predicted_ms"),
              "drafts_accepted": t.get("draft_n_accepted"), "drafts": t.get("draft_n"),
              "first_token_s": t.get("predicted_ms") and (t.get("prompt_ms", 0) + t.get("predicted_ms", 0)) / 1000,
              "reply_head": (m.get("content") or "")[:60], "raw_timings": t}
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: result[k] for k in ("context", "prompt_tokens", "pp_tok_s", "tg_tok_s",
                                             "drafts_accepted", "drafts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
