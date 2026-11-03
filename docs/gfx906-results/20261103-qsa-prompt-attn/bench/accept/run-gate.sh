#!/bin/sh
# The MTP-acceptance half of the gfx906 prompt-attention kernel's gate, run through the deployment:
# STRATA_PROMPT_ATTN_GFX906 off, on, off (blocked A/B/A - greedy drafting is deterministic, so the two
# off arms must reproduce their counts exactly). Three request types, greedy, 600 tokens, per arm one
# compose recreate. Reads timings.draft_n_accepted / draft_n from each response.
BASE=/home/me/Documents/strata
OUT=$BASE/data/rp-opt/accept-attn
cd "$BASE" || exit 1
: > "$OUT/mtp4.log"
for arm in 0 1 0; do
  {
    echo "=== arm prompt_attn=$arm at $(date -Is) ==="
    STRATA_PROMPT_ATTN_GFX906=$arm docker compose up -d --force-recreate >/dev/null 2>&1
    i=0
    while [ $i -lt 120 ]; do
      code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8080/health 2>/dev/null)
      [ "$code" = "200" ] && break
      sleep 5; i=$((i+1))
    done
    echo "health after ${i} polls: $code"
    echo "env in container: $(docker inspect strata --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -c '^STRATA_PROMPT_ATTN_GFX906=')"
    n=0
    python3 -c "import json;[print(json.dumps(r)) for r in json.load(open('/tmp/mtp-reqs.json'))]" | \
    while read -r body; do
      n=$((n+1))
      curl -s -m 300 -o "$OUT/mtp4-$arm-$n.json" -w "req $n: http=%{http_code} wall=%{time_total}\n" \
        http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d "$body"
    done
  } >> "$OUT/mtp4.log" 2>&1
done
echo "=== done at $(date -Is) ===" >> "$OUT/mtp4.log"
