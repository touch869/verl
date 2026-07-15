#!/usr/bin/env python3
"""Plot per-replica KV-cache + MFU signals on one time-aligned figure.

Nine panels share the time x-axis; the same replica colour repeats down the
figure so the eye tracks one replica vertically:

  1. KV Load (retained)        — kv_cache_load = retained_blocks / num_gpu_blocks
  2. vLLM usage_perc           — running-only fraction (1 - free/total)
  3. MFU                       — vLLM estimated_flops / peak_flops (solid=realtime
                                 1-min avg, dashed=total avg). --peak-tflops sets
                                 the denominator (default 560 = Atlas 800I A3 / NPU)
  4. running requests          — num_requests_running
  5. waiting requests          — num_requests_waiting
  6. cumulative gpu evictions  — retained_blocks drops between tally snapshots
  7. gpu prefix hit %          — windowed local prefix-cache hit rate
  8. prefill recompute tokens  — cache-miss prefill tokens over the window
  9. external-hit rate         — cross-replica (mooncake) hits / prefix lookups

Sources (collector evidence + kv-events tally; loguru timestamps):
  `vllm-evidence replica=<id> kv=<f> usage=<f> run=<n> wait=<n> | TTFT=.. .. |
   prefill=<n> cached=<n> (hit=<p>%) decode=<n> external=<n> flops=<n> [poll #..]`
  `kv-events tally: .. | retained_blocks/replica={'replica-X': N, ..}`

  kv      = DataStore.kv_cache_load (retained, hash-bearing fraction)
  usage   = vLLM kv_cache_usage_perc (running-only: 1 - free/total)
  flops   = vLLM estimated_flops_per_gpu_total window delta (analytic, architecture-
            accurate; gated by --enable-mfu-metrics). MFU = flops/s / peak_flops.
  run/wait = num_requests_running / num_requests_waiting
  hit     = 100 * cached / (cached + prefill) over the window
  prefill = PROMPT_TOKENS window delta (cache-miss tokens — the recompute cost)
  ext-rate = external / (prefill + cached)

MFU denominator: `--peak-tflops` (default 560 = Atlas 800I A3 FP16/NPU; use 750 for
800T A3). The flops numerator is per-GPU already (vLLM divides by device count), so
for TP1 the denominator is the single-NPU peak. NOTE: LLM decode is bandwidth-bound,
so MFU is naturally low (10-40%) in decode-heavy phases — that's expected, not idle.

`usage=` and `flops=` are optional (added 2026-07-13/14); older logs parse fine and
those panels stay empty.

Usage:
    python plot_eviction.py LOG [LOG ...] [--out evict.png] [--title "..."]
    python plot_eviction.py D.log --frac 0.3 --max-points 1500 --peak-tflops 750

Lines without a parseable timestamp are skipped — the x-axis is always real
time, never a line-order index.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

# vllm-evidence replica=X kv=0.021 usage=0.450 run=240 wait=12 | TTFT=.. .. |
#   prefill=3365 cached=41472 (hit=92.5%) decode=4175 external=0 flops=8400000000000000
# `usage=` and `flops=` are optional so older logs still parse.
EVIDENCE_PAT = re.compile(
    r"vllm-evidence\s+replica=(?P<rep>\S+)\s+kv=(?P<kv>\S+)"
    r"(?:\s+usage=(?P<usage>\S+))?"
    r"\s+run=(?P<run>\S+)\s+wait=(?P<wait>\S+)"
    r".*?prefill=(?P<pre>\d+)\s+cached=(?P<cac>\d+)\s+\(hit=(?P<hit>\S+?)\)"
    r"\s+decode=\d+\s+external=(?P<ext>\d+)(?:\s+flops=(?P<flops>-?\d+))?"
)

# Anchor on `retained_blocks/replica=` so the earlier events={...} dict isn't captured.
TALLY_PAT = re.compile(r"retained_blocks/replica=(?P<d>\{[^}]*\})")

TS_LOGURU = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:[,.]\d+)?")
TS_VLLM = re.compile(r"INFO\s+(?P<ts>\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _parse_ts(line: str):
    """Leading loguru/vllm timestamp, or None if none is present."""
    m = TS_LOGURU.search(line)
    if m:
        return datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
    m = TS_VLLM.search(line)
    if m:
        return datetime.strptime(f"2026 {m.group('ts')}", "%Y %m-%d %H:%M:%S")
    return None


def parse_signal_line(line: str):
    """Parse one log line into ``(ts, kind, fields)`` or ``None``.

    Returns ``None`` for non-signal lines (vllm engine stats, ray noise). ``kind``
    is ``'evidence'`` or ``'tally'``; ``fields`` is the regex groupdict. ``ts`` may
    be None for a matched signal line (caller skips it to keep the time axis honest).
    """
    m = EVIDENCE_PAT.search(line)
    if m:
        return _parse_ts(line), "evidence", m.groupdict()
    m = TALLY_PAT.search(line)
    if m:
        return _parse_ts(line), "tally", m.groupdict()
    return None


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _downsample(points: list, max_n: int) -> list:
    n = len(points)
    if max_n <= 0 or n <= max_n:
        return points
    idx = sorted(set(round(i) for i in (j * (n - 1) / (max_n - 1) for j in range(max_n))))
    return [points[i] for i in idx]


def _norm_usage(v: float) -> float:
    """usage_perc is 0-1 (vLLM docs); tolerate a stray 0-100 from older scrapes."""
    if v == v and v > 1.5:
        return v / 100.0
    return v


def _mfu_series(flops_pts: list, window_s: float, peak_flops: float):
    """FLOPs-window-delta series → (realtime_pts, total_pts) MFU time series.

    flops_pts: [(ts, d_flops)] where d_flops is the flops done in that evidence
    window (which spans from the PREVIOUS point's ts to this one). We convert each
    point to a throughput over its own duration (d_flops / (ts_j - ts_{j-1})) so
    the realtime window is flops/time over ~window_s — NOT raw d_flops summed over
    a wall-clock band (that overcounts when >1 ~30s evidence point falls inside the
    band). realtime and total are both O(n) sliding/cumulative.
    """
    if len(flops_pts) < 2 or peak_flops <= 0:
        return [], []
    pts = sorted((p for p in flops_pts if p[0] is not None), key=lambda p: p[0])
    # Build evidence windows: (ts_end, duration_s, d_flops).
    wins = []
    for j in range(1, len(pts)):
        t1, df1 = pts[j]
        dur = (t1 - pts[j - 1][0]).total_seconds()
        if dur > 0:
            wins.append((t1, dur, df1 if df1 == df1 else 0.0))
    if not wins:
        return [], []
    win_q: deque = deque()   # (duration, d_flops)
    dur_sum = flop_sum = 0.0
    cum = 0.0
    t0 = pts[0][0]           # counting start = first evidence ts
    real_out, tot_out = [], []
    for (t, dur, df) in wins:
        cum += df
        win_q.append((dur, df)); dur_sum += dur; flop_sum += df
        # Keep the window covering ~window_s: drop oldest while the remainder still
        # covers >= window_s.
        while len(win_q) > 1 and dur_sum - win_q[0][0] >= window_s:
            d_old, f_old = win_q.popleft(); dur_sum -= d_old; flop_sum -= f_old
        real_mfu = (flop_sum / dur_sum) / peak_flops if dur_sum > 0 else 0.0
        elapsed = (t - t0).total_seconds()
        tot_mfu = (cum / elapsed) / peak_flops if elapsed > 0 else 0.0
        real_out.append((t, real_mfu)); tot_out.append((t, tot_mfu))
    return real_out, tot_out


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-replica KV + MFU signals (9 time-aligned panels)")
    ap.add_argument("logs", nargs="+", help="log file(s)")
    ap.add_argument("-o", "--out", default=None,
                   help="output image (default: <first-log-name>.png)")
    ap.add_argument("--title", default=None, help="plot title")
    ap.add_argument("--frac", type=float, default=1.0, help="plot first FRAC of the time window; (0,1]")
    ap.add_argument("--max-points", type=int, default=2000, help="downsample each curve; 0 disables")
    ap.add_argument("--ymax", type=float, default=1.05, help="top of the 0-1 panels (load, usage)")
    ap.add_argument("--peak-tflops", type=float, default=560.0,
                   help="per-NPU peak FLOPs/s in TFLOPS (MFU denominator). "
                        "560=Atlas 800I A3 FP16/NPU (default), 750=800T A3.")
    ap.add_argument("--mfu-window-s", type=float, default=60.0,
                   help="realtime MFU sliding-window length in seconds (default 60 = last 1 min)")
    args = ap.parse_args()
    if not (0.0 < args.frac <= 1.0):
        ap.error(f"--frac must be in (0, 1], got {args.frac}")
    if args.out is None:
        # Default: same name as the first log, with the suffix swapped to .png.
        args.out = str(Path(args.logs[0]).with_suffix(".png"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not installed.  pip install matplotlib", file=sys.stderr)
        return 2

    peak_flops = args.peak_tflops * 1e12

    # per-replica signal buffers: {replica: [(t, value), ...]}
    load_pts: dict[str, list] = defaultdict(list)
    usage_pts: dict[str, list] = defaultdict(list)
    run_pts: dict[str, list] = defaultdict(list)
    wait_pts: dict[str, list] = defaultdict(list)
    hit_pts: dict[str, list] = defaultdict(list)
    prefill_pts: dict[str, list] = defaultdict(list)
    ext_pts: dict[str, list] = defaultdict(list)
    flops_pts: dict[str, list] = defaultdict(list)   # window delta flops → MFU
    retained_history: dict[str, list] = defaultdict(list)
    all_replicas: set[str] = set()
    n_ev = n_tally = n_no_ts = 0

    for path in args.logs:
        try:
            f = open(path, errors="replace")
        except OSError as e:
            print(f"WARN: cannot open {path}: {e}", file=sys.stderr)
            continue
        with f:
            for line in f:
                parsed = parse_signal_line(line)
                if parsed is None:
                    continue
                ts, kind, g = parsed
                if ts is None:
                    n_no_ts += 1
                    continue
                if kind == "evidence":
                    rep = g["rep"]
                    all_replicas.add(rep)
                    load_pts[rep].append((ts, _to_float(g["kv"])))
                    usage_pts[rep].append((ts, _norm_usage(_to_float(g.get("usage")))))
                    run_pts[rep].append((ts, _to_float(g["run"])))
                    wait_pts[rep].append((ts, _to_float(g["wait"])))
                    hit_pts[rep].append((ts, _to_float(str(g["hit"]).rstrip("%"))))
                    pre = int(g["pre"]); cac = int(g["cac"]); ext = int(g["ext"])
                    prefill_pts[rep].append((ts, pre))
                    denom = pre + cac
                    ext_pts[rep].append((ts, (ext / denom) if denom > 0 else 0.0))
                    if g.get("flops") is not None:
                        flops_pts[rep].append((ts, float(g["flops"])))
                    n_ev += 1
                else:  # tally
                    try:
                        d = ast.literal_eval(g["d"])
                    except Exception:
                        continue
                    for rep, n in d.items():
                        all_replicas.add(rep)
                        retained_history[rep].append((ts, int(n)))
                    n_tally += 1

    if n_no_ts:
        print(f"WARN: {n_no_ts} signal lines had no parseable timestamp — skipped", file=sys.stderr)
    if not all_replicas:
        print(f"ERROR: no vllm-evidence / kv-events tally lines found in {args.logs}", file=sys.stderr)
        return 1

    # Reconstruct cumulative evictions per replica.
    evict_pts: dict[str, list] = defaultdict(list)
    for rep, hist in retained_history.items():
        if not hist:
            continue
        cum = 0
        prev = hist[0][1]
        out = []
        for (x, n) in hist:
            if n < prev:
                cum += (prev - n)
            out.append((x, cum))
            prev = n
        evict_pts[rep] = out

    # Time-based truncation (global across all panels/replicas), then downsample.
    all_times = [p[0] for buf in (load_pts, usage_pts, run_pts, wait_pts,
                                  hit_pts, prefill_pts, ext_pts, evict_pts, flops_pts)
                 for pts in buf.values() for p in pts]
    t_cut = None
    if all_times and args.frac < 1.0:
        t_min, t_max = min(all_times), max(all_times)
        t_cut = t_min + (t_max - t_min) * args.frac

    def prep(buf: dict[str, list]) -> dict[str, list]:
        out = {}
        for rep, pts in buf.items():
            pts = sorted((p for p in pts if p[0] is not None), key=lambda p: p[0])
            if t_cut is not None:
                pts = [p for p in pts if p[0] <= t_cut]
                if not pts:
                    continue
            out[rep] = _downsample(pts, args.max_points)
        return out

    load_pts = prep(load_pts)
    usage_pts = prep(usage_pts)
    run_pts = prep(run_pts)
    wait_pts = prep(wait_pts)
    hit_pts = prep(hit_pts)
    prefill_pts = prep(prefill_pts)
    ext_pts = prep(ext_pts)
    evict_pts = prep(evict_pts)

    # MFU: compute from the full (truncated) flops-delta series, THEN downsample the
    # resulting MFU curves — summing downsampled deltas would corrupt the window/cum.
    mfu_real: dict[str, list] = {}
    mfu_tot: dict[str, list] = {}
    for rep, pts in flops_pts.items():
        pts = sorted((p for p in pts if p[0] is not None), key=lambda p: p[0])
        if t_cut is not None:
            pts = [p for p in pts if p[0] <= t_cut]
        if not pts:
            continue
        real, tot = _mfu_series(pts, args.mfu_window_s, peak_flops)
        mfu_real[rep] = _downsample(real, args.max_points)
        mfu_tot[rep] = _downsample(tot, args.max_points)

    max_evict = max((pts[-1][1] for pts in evict_pts.values() if pts), default=0)

    order = sorted(
        all_replicas,
        key=lambda r: (load_pts.get(r, [(datetime.min,)])[0][0]
                       if load_pts.get(r) else datetime.min, r),
    )

    fig, axes = plt.subplots(
        9, 1, sharex=True, figsize=(14, 31),
        gridspec_kw={"height_ratios": [2, 2, 2.4, 2, 2, 2.4, 2, 2.4, 2]},
    )
    (ax_load, ax_usage, ax_mfu, ax_run, ax_wait,
     ax_evict, ax_hit, ax_pre, ax_ext) = axes
    cmap = plt.get_cmap("tab10" if len(order) <= 10 else "tab20")
    colors = {rep: cmap(i % cmap.N) for i, rep in enumerate(order)}

    def _draw(ax, buf, ylabel, ylim_top=None, hline=None):
        for rep in order:
            pts = buf.get(rep)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=colors[rep], linestyle="-", linewidth=1.5, alpha=0.9)
        if hline is not None:
            ax.axhline(hline, color="red", linestyle=":", linewidth=1.0, alpha=0.5)
        ax.set_ylabel(ylabel)
        if ylim_top is not None:
            ax.set_ylim(top=ylim_top)
        ax.grid(True, alpha=0.3)

    _draw(ax_load, load_pts, "KV Load\n(retained_blocks / num_gpu_blocks)", args.ymax, hline=1.0)
    _draw(ax_usage, usage_pts, "vLLM usage_perc\n(running blocks / num_gpu_blocks)", args.ymax, hline=1.0)
    # MFU: realtime (solid) + total (dashed), per replica.
    for rep in order:
        if mfu_real.get(rep):
            xs = [p[0] for p in mfu_real[rep]]; ys = [p[1] for p in mfu_real[rep]]
            ax_mfu.plot(xs, ys, color=colors[rep], linestyle="-", linewidth=1.6, alpha=0.9)
        if mfu_tot.get(rep):
            xs = [p[0] for p in mfu_tot[rep]]; ys = [p[1] for p in mfu_tot[rep]]
            ax_mfu.plot(xs, ys, color=colors[rep], linestyle="--", linewidth=1.2, alpha=0.7)
    ax_mfu.set_ylabel(f"MFU\n(solid=realtime {int(args.mfu_window_s)}s, dashed=total)\n"
                      f"peak={args.peak_tflops:.0f} TFLOPS/NPU")
    ax_mfu.grid(True, alpha=0.3)
    _draw(ax_run, run_pts, "running requests\n(num_requests_running)")
    _draw(ax_wait, wait_pts, "waiting requests\n(num_requests_waiting)")
    _draw(ax_evict, evict_pts, "total gpu block evict")
    _draw(ax_hit, hit_pts, "gpu prefix hit %", 100.0)
    _draw(ax_pre, prefill_pts, "prefill recompute\n(cache-miss tokens / window)")
    _draw(ax_ext, ext_pts, "external-hit", 1.05)

    # Legend on the top panel only (colours repeat down the figure).
    for rep in order:
        if load_pts.get(rep):
            ax_load.plot([], [], color=colors[rep], linestyle="-", linewidth=1.5, label=rep)
    ax_load.legend(loc="best", fontsize=8, ncol=max(1, (len(order) + 7) // 8))

    ax_ext.set_xlabel("time")
    fig.suptitle(
        args.title or "KV Load / usage / MFU / run / wait / evictions / prefix-hit / prefill-recompute / external-hit (time-aligned)",
        y=0.995,
    )
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(args.out, dpi=140)

    def _last(buf):
        return buf[-1][1] if buf else float("nan")

    def _rng(buf):
        if not buf:
            return float("nan"), float("nan")
        vs = [p[1] for p in buf]
        return min(vs), max(vs)

    print(f"OK: {n_ev} evidence lines, {n_tally} tally lines, {len(all_replicas)} replicas -> {args.out}")
    print(f"   peak={args.peak_tflops:.0f} TFLOPS/NPU, mfu_window={args.mfu_window_s:.0f}s, max cum evictions={max_evict}")
    print("   per-replica summary:")
    for rep in order:
        lp = load_pts.get(rep, []); up = usage_pts.get(rep, [])
        rp = run_pts.get(rep, []); wp = wait_pts.get(rep, [])
        ep = evict_pts.get(rep, [])
        hp = hit_pts.get(rep, []); pp = prefill_pts.get(rep, []); xp = ext_pts.get(rep, [])
        mp = mfu_real.get(rep, [])
        lmin, lmax = _rng(lp); umin, umax = _rng(up)
        print(f"     {rep:>16s}: "
              f"load[min={lmin:.3f} max={lmax:.3f} last={_last(lp):.3f}] "
              f"usage[min={umin:.3f} max={umax:.3f} last={_last(up):.3f}] "
              f"mfu[last={_last(mp):.3f}] "
              f"run[last={_last(rp):.0f}] wait[last={_last(wp):.0f}] "
              f"evict[cum={_last(ep)}] "
              f"hit[last={_last(hp):.1f}%] "
              f"prefill[last={_last(pp)}] "
              f"ext[last={_last(xp):.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
