#!/usr/bin/env python3
"""Analyze run_matrix_overnight A/B(/C/D) logs on Ascend NPU — comprehensive.

Captures everything useful for the KVCAware experiment + gives an analysis verdict.
No hardcoded paths: pass any dirs/files; logs are auto-discovered.

Metrics:
  - sanity   : Mean RM Score, walltime (from driver GROUP start/done)
  - pool     : GPU KV cache size (tokens/blocks/block_size)
  - pressure : retained-cache OCCUPANCY = retained_blocks/num_gpu_blocks (kv-events).
               NOT vllm kv_cache_usage_perc (running-only, excludes prefix cache —
               undercounts ~3x; worthless for pressure). Two log formats normalized:
                 * standalone collector (A/C): `kv_cache_load: replica-X=0.02 ...` (ratio)
                 * KVCAwareBalancer (B/D)    : `retained_blocks/replica={'X':N,...}` (blocks)
  - prefix   : cache-hit% (evidence) + Prefix/External cache hit rate (engine-stats)
  - prefill  : TTFT / queue / prefillT (real prefill cost) — pct dist reveals hit/miss bimodality
               prefill(recompute) vs cached(prefix-hit) totals + prefill share
  - decode   : TPOT, decode tokens
  - throughput: generation tokens/s (engine-stats)
  - kv-events flow check: did retained data actually stream? (the #4 concern)

Output: per-group summary + A-vs-B (and C-vs-D if present) comparison + VERDICT.

Usage:
  python analyze_matrix.py <path> [<path> ...] [<driver_log>]
  # <path>: a dir (searched for {A,B,C,D}.log, *_scrape.log, matrix_*/ subdirs) or a log file.
  # driver  : any path whose content has `=== GROUP X start/done ===` (auto-detected if in a dir).
  # Examples:
  #   python analyze_matrix.py /mnt/data/h00500767/logs
  #   python analyze_matrix.py results/matrix_20260711_1742_smoke matrix_ab.log
  #   python analyze_matrix.py A.log B.log driver.log
"""
import re
import sys
import ast
import statistics as st
from pathlib import Path

GROUPS = ["A", "B", "C", "D"]
DEFAULT_BLOCK_SIZE = 128  # vllm-ascend (3090 was 16)

EVIDENCE_PAT = re.compile(
    r"vllm-evidence replica=(?P<rep>\S+) kv=(?P<kv>\S+)(?:\s+usage=\S+)? run=(?P<run>\S+) wait=(?P<wait>\S+) \| "
    r"TTFT=(?P<ttft>\S+) queue=(?P<queue>\S+) prefillT=(?P<preft>\S+) TPOT=(?P<tpot>\S+) \| "
    r"prefill=(?P<pre>\d+) cached=(?P<cac>\d+) \(hit=(?P<hit>\S+%)\) "
    r"decode=(?P<dec>\d+) external=(?P<ext>\d+)")
ENGINE_PAT = re.compile(
    r"Avg prompt throughput: (?P<prompt>[\d.]+) tokens/s.*?"
    r"Avg generation throughput: (?P<gen>[\d.]+) tokens/s.*?"
    r"Running: (?P<run>\d+) reqs, Waiting: (?P<wait>\d+) reqs, "
    r"GPU KV cache usage: (?P<kv>[\d.]+)%, Prefix cache hit rate: (?P<ph>[\d.]+)%"
    r"(?:, External prefix cache hit rate: (?P<ext>[\d.]+)%)?")
GPU_TOKENS_PAT = re.compile(r"GPU KV cache size: (?P<n>[\d,]+) tokens")
GPU_BLOCKS_PAT = re.compile(r"# GPU blocks: (?P<n>\d+)")
NUM_BLOCKS_METRIC_PAT = re.compile(r"'num_gpu_blocks': (?P<n>\d+)")
KV_LOAD_PAT = re.compile(r"kv_cache_load:\s*(?P<rest>.+)")
RETAINED_PAT = re.compile(r"retained_blocks/replica=(?P<d>\{[^}]*\})")
# KVCAwareBalancer score() line: `score(): replica=R kv=X running=.. waiting=..` where X is
# kv_cache_load (retained occupancy) — kvc_aware.py:188 kv_usage=store.kv_cache_load(...).
# This is B/D's pressure source (they don't emit kv_cache_load:/retained_blocks lines).
SCORE_KV_PAT = re.compile(r"score\(\): replica=\S+ kv=(?P<kv>[\d.]+)")
RM_PAT = re.compile(r"Mean RM Score:\s*(?P<v>[\d.]+)")
GROUP_TS_PAT = re.compile(r"(?P<ts>\d{2}:\d{2}:\d{2}).*=== GROUP (?P<g>[A-D]) (?P<phase>start|done)")
GINI_HINT_PAT = re.compile(r"gini[:=]\s*(?P<v>[\d.]+)", re.I)


def _f(s):
    return None if s in ("-", "-ms", "-%") else float(re.sub(r"[ms%]", "", s))


def _hms(s):
    h, m, sec = map(int, s.split(":"))
    return h * 3600 + m * 60 + sec


def _pct(xs, p):
    """percentile of a list."""
    if not xs:
        return float("nan")
    xs = sorted(xs); k = (len(xs) - 1) * p / 100; lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _stat(xs, unit=""):
    if not xs:
        return "n/a"
    return (f"avg={st.mean(xs):.0f}{unit} p50={st.median(xs):.0f}{unit} "
            f"p90={_pct(xs,90):.0f}{unit} max={max(xs):.0f}{unit} (n={len(xs)})")


# ── discovery ────────────────────────────────────────────────────────────
def discover(paths):
    """Return {group: [lines], '_driver': [lines]}.

    Merges G.log + G_scrape.log per group (standalone collector writes evidence +
    kv_cache_load to G_scrape.log). Dedups exact line content so it's safe whether
    or not run_matrix_overnight.sh already folded G_scrape.log into G.log
    (`cat G_scrape.log >> G.log`) — avoids double-counting evidence/kv_cache_load.
    """
    raw = {g: [] for g in GROUPS}
    raw["_driver"] = []
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files += [f for f in p.rglob("*") if f.is_file() and f.suffix in ("", ".log", ".txt")]
        elif p.is_file():
            files.append(p)
    for f in dict.fromkeys(files):  # de-dup file paths, keep order
        name = f.name
        lines = f.read_text(errors="ignore").splitlines()
        if not lines:
            continue
        if any("=== GROUP" in l and ("start" in l or "done" in l) for l in lines[-2000:] + lines[:200]):
            raw["_driver"] += lines
            continue
        m = re.match(r"([A-D])\.log$|([A-D])_scrape\.log$", name)
        if m:
            raw[m.group(1) or m.group(2)] += lines
    # dedup exact lines per group/driver (handles folded-vs-separate scrape logs)
    out = {}
    for k, lines in raw.items():
        seen = set(); dd = []
        for ln in lines:
            if ln not in seen:
                seen.add(ln); dd.append(ln)
        out[k] = dd
    return out


# ── extractors ───────────────────────────────────────────────────────────
def extract_evidence(lines):
    rows = []
    for line in lines:
        m = EVIDENCE_PAT.search(line)
        if not m:
            continue
        d = m.groupdict()
        hit = None if d["hit"] == "-%" else float(d["hit"].rstrip("%"))
        rows.append({"ttft": _f(d["ttft"]), "queue": _f(d["queue"]), "preft": _f(d["preft"]),
                     "tpot": _f(d["tpot"]), "pre": int(d["pre"]), "cac": int(d["cac"]),
                     "dec": int(d["dec"]), "ext": int(d["ext"]), "hit": hit})
    return rows


def extract_engine(lines):
    rows = []
    for line in lines:
        m = ENGINE_PAT.search(line)
        if not m:
            continue
        d = m.groupdict()
        rows.append({"prompt": float(d["prompt"]), "gen": float(d["gen"]),
                     "ph": float(d["ph"]),
                     "ext": float(d["ext"]) if d.get("ext") is not None else None})
    return rows


def extract_pool(lines):
    text = "\n".join(lines)
    tokens_m = GPU_TOKENS_PAT.search(text)
    blocks_m = GPU_BLOCKS_PAT.search(text)
    metric_m = NUM_BLOCKS_METRIC_PAT.search(text)
    tokens = int(tokens_m.group("n").replace(",", "")) if tokens_m else None
    blocks = int(blocks_m.group("n")) if blocks_m else None
    metric_blocks = int(metric_m.group("n")) if metric_m else None
    nb = metric_blocks or blocks or (tokens // DEFAULT_BLOCK_SIZE if tokens else None)
    bs = (tokens // nb) if (tokens and nb) else DEFAULT_BLOCK_SIZE
    return nb, tokens, bs


def extract_pressure(lines, num_blocks):
    """Retained-occupancy snapshots (0-1) from any of 3 log formats + the source tag."""
    snaps, per_rep_final = [], None
    score_vals = []  # balancer score() kv= (per-replica retained; may be numerous)
    src = None
    for line in lines:
        m = KV_LOAD_PAT.search(line)  # A/C standalone collector (already a ratio)
        if m:
            vals = re.findall(r"replica-\S+=([\d.]+)", m.group("rest"))
            vs = [float(v) for v in vals if float(v) >= 0]
            if vs:
                snaps.append(st.mean(vs)); src = src or "kv_cache_load (standalone)"
            continue
        m = RETAINED_PAT.search(line)  # older balancer tally (blocks; normalize)
        if m:
            try:
                d = ast.literal_eval(m.group("d"))
                vs = [int(v) for v in d.values()]
            except (ValueError, SyntaxError):
                vs = []
            if vs and num_blocks:
                snaps.append(st.mean(vs) / num_blocks); per_rep_final = vs
                src = src or "retained_blocks/replica (balancer tally)"
            continue
        m = SCORE_KV_PAT.search(line)  # B/D balancer score() kv= (retained, per-replica)
        if m:
            v = float(m.group("kv"))
            if v >= 0:
                score_vals.append(v); src = src or "score() kv= (balancer)"
    # score() values are per-replica samples; cap to avoid huge-memory runs
    if score_vals:
        if len(score_vals) > 20000:
            score_vals = score_vals[::len(score_vals) // 20000]
        snaps += score_vals
    return snaps, per_rep_final, src


def extract_walltime(driver_lines, group):
    starts, dones = [], []
    for l in driver_lines:
        m = GROUP_TS_PAT.search(l)
        if m and m.group("g") == group:
            (starts if m.group("phase") == "start" else dones).append(_hms(m.group("ts")))
    if not starts or not dones:
        return None
    dt = dones[-1] - starts[0]
    return dt + 86400 if dt < 0 else dt


# ── summary ──────────────────────────────────────────────────────────────
def summarize(label, lines):
    print(f"\n{'='*64}\n=== GROUP {label} ===\n{'='*64}")
    if not lines:
        print("  (no log found for this group)"); return None

    rm = [float(RM_PAT.search(l).group("v")) for l in lines if RM_PAT.search(l)]
    nb, tokens, bs = extract_pool(lines)
    ev = extract_evidence(lines)
    eng = extract_engine(lines)
    pressure, rep_final, psrc = extract_pressure(lines, nb)

    if rm:
        print(f"Mean RM Score : {rm[-1]} (last of {len(rm)})")
    if tokens:
        print(f"KV pool       : {tokens:,} tokens = {nb:,} blocks (block_size={bs})")

    # pressure (THE metric)
    flowed = bool(pressure)
    print(f"\n[pressure] source: {psrc or 'NONE — no kv_cache_load/retained/score lines'}")
    print(f"[pressure] retained data flowed: {'YES' if flowed else 'NO  <- NOT produced (check collector/balancer)'}")
    if flowed:
        pmax = max(pressure)
        print(f"  retained occupancy: avg={st.mean(pressure)*100:.1f}% p50={st.median(pressure)*100:.1f}% "
              f"max={pmax*100:.1f}% (n={len(pressure)} snapshots)")
        if rep_final and nb:
            bal = max(rep_final) / min(rep_final) if min(rep_final) > 0 else float("inf")
            print(f"  final per-replica balance: max/min={bal:.2f} (1.0=perfectly balanced)")
        verdict = "HIGH (> =70%) -> pressure regime OK" if pmax >= 0.70 else "LOW (<70%) -> insufficient pressure (cf 910B3); bump MAX_SAMPLES"
        print(f"  -> {verdict}")

    # prefix cache
    if ev:
        hits = [r["hit"] for r in ev if r["hit"] is not None]
        if hits:
            print(f"\n[prefix cache] hit%: avg={st.mean(hits):.1f} p50={st.median(hits):.1f} max={max(hits):.1f}")
    if eng:
        phs = [r["ph"] for r in eng]
        exts = [r["ext"] for r in eng if r["ext"] is not None]
        print(f"  engine Prefix hit: avg={st.mean(phs):.1f}% max={max(phs):.1f}%")
        if exts:
            print(f"  engine External hit: avg={st.mean(exts):.2f}% max={max(exts):.2f}%")

    # prefill economics
    if ev:
        ttfts = [r["ttft"] for r in ev if r["ttft"] is not None]
        queues = [r["queue"] for r in ev if r["queue"] is not None]
        prefts = [r["preft"] for r in ev if r["preft"] is not None]
        tpots = [r["tpot"] for r in ev if r["tpot"] is not None]
        pre = sum(r["pre"] for r in ev); cac = sum(r["cac"] for r in ev)
        dec = sum(r["dec"] for r in ev); ext = sum(r["ext"] for r in ev)
        print(f"\n[prefill] ({len(ev)} evidence windows)")
        if ttfts:
            print(f"  TTFT     {_stat(ttfts,'ms')}")
        if queues:
            print(f"  queue    {_stat(queues,'ms')}")
        if prefts:
            print(f"  prefillT {_stat(prefts,'ms')}  <- real prefill cost; bimodal = hit(fast)/miss(slow)")
        if tpots:
            print(f"  TPOT     {_stat(tpots,'ms')}")
        print(f"  totals: prefill(recompute)={pre:,}  cached(prefix-hit)={cac:,}  decode={dec:,}  external(mc)={ext}")
        if pre + cac > 0:
            print(f"  prefill share={100*pre/(pre+cac):.1f}% of prompt tokens (lower = kvcare aggregates better)")

    # throughput
    if eng:
        gens = [r["gen"] for r in eng]; prompts = [r["prompt"] for r in eng]
        print(f"\n[throughput] gen tok/s avg={st.mean(gens):.0f} max={max(gens):.0f} | "
              f"prompt tok/s avg={st.mean(prompts):.0f} max={max(prompts):.0f}")

    return {
        "tokens": tokens, "nb": nb,
        "pressure_max": max(pressure) if pressure else None,
        "pressure_avg": st.mean(pressure) if pressure else None,
        "pressure_n": len(pressure),
        "hit_avg": st.mean([r["hit"] for r in ev if r["hit"] is not None]) if ev and any(r["hit"] is not None for r in ev) else None,
        "prefillt_med": st.median([r["preft"] for r in ev if r["preft"] is not None]) if ev and any(r["preft"] is not None for r in ev) else None,
        "prefillt_p90": _pct([r["preft"] for r in ev if r["preft"] is not None], 90) if ev and any(r["preft"] is not None for r in ev) else None,
        "ext": sum(r["ext"] for r in ev) if ev else 0,
        "gen_avg": st.mean([r["gen"] for r in eng]) if eng else None,
    }


def compare(label_x, label_y, sx, sy, driver_lines):
    """y vs x (e.g. B vs A): did y (kvcare) beat x (sticky)?"""
    print(f"\n{'='*64}\n=== {label_y} − {label_x} COMPARISON ===\n{'='*64}")
    wx = extract_walltime(driver_lines, label_x)
    wy = extract_walltime(driver_lines, label_y)
    if wx is not None and wy is not None:
        delta = wy - wx; pct = 100 * delta / wx if wx else 0
        print(f"walltime   : {label_x}={wx//60}m{wx%60}s  {label_y}={wy//60}m{wy%60}s  "
              f"delta={delta//60}m{delta%60}s ({pct:+.1f}%)  {'<- y FASTER (3090 reproduced)' if delta < 0 else ''}")
    else:
        print("walltime   : (need driver_log with GROUP start/done markers — run the matrix with `> driver.log 2>&1`)")
    if sx and sy:
        for k, lbl, fmt in [
            ("pressure_max", "pressure max", lambda v: f"{v*100:.1f}%"),
            ("hit_avg", "cache-hit avg", lambda v: f"{v:.1f}%"),
            ("prefillt_med", "prefillT p50", lambda v: f"{v:.0f}ms"),
            ("prefillt_p90", "prefillT p90", lambda v: f"{v:.0f}ms"),
            ("gen_avg", "gen tok/s", lambda v: f"{v:.0f}")]:
            vx, vy = sx.get(k), sy.get(k)
            if vx is not None and vy is not None:
                print(f"{lbl:14}: {label_x}={fmt(vx)}  {label_y}={fmt(vy)}")
        print(f"external(mc) : {label_x}={sx.get('ext',0)}  {label_y}={sy.get('ext',0)}")
    print("\nVERDICT:")
    pmax = max((sx or {}).get("pressure_max") or 0, (sy or {}).get("pressure_max") or 0)
    if pmax < 0.70:
        print(f"  ⚠ pressure LOW (max {pmax*100:.0f}% <70%) — NOT the 3090 regime; rerun with MAX_SAMPLES=300/400")
        return
    print(f"  ✓ pressure OK (max {pmax*100:.0f}% ≥70%)")
    if wx and wy and wy < wx:
        print(f"  ✓ {label_y} walltime < {label_x} — KVCAware saved walltime (3090 result REPRODUCED on NPU)")
    elif wx and wy:
        print(f"  ✗ {label_y} walltime ≥ {label_x} — KVCAware did NOT save walltime (cf 910B3 no-benefit)")
    if sx and sy and (sy.get("prefillt_med") or 0) < (sx.get("prefillt_med") or float("inf")):
        print(f"  ✓ {label_y} prefillT < {label_x} — KVCAware DID aggregate prefill (mechanism works)")


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    data = discover(sys.argv[1:])
    driver = data["_driver"]
    if not driver:
        print("(no driver log found — walltime needs a file with `=== GROUP X start/done ===`; "
              "run the matrix with `> driver.log 2>&1`)")
    present = [g for g in GROUPS if data[g]]
    if not present:
        print("No group logs (A.log/B.log/...) found in the given paths."); sys.exit(1)
    summaries = {}
    for g in present:
        summaries[g] = summarize(g, data[g])
    if "A" in summaries and "B" in summaries:
        compare("A", "B", summaries["A"], summaries["B"], driver)
    if "C" in summaries and "D" in summaries:
        compare("C", "D", summaries["C"], summaries["D"], driver)
    print(f"\n(analyzed {len(present)} group(s): {', '.join(present)})")


if __name__ == "__main__":
    main()
