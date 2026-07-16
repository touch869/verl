#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Plot per-replica KV-cache + MFU signals on one time-aligned figure.

Each metric is a :class:`Panel` subclass owning its own parsing, series transform,
axis style, and summary line; :func:`build_panels` returns the ordered list and the
plot/summary loops just iterate it — adding a panel is writing one subclass and
appending an instance. Nine panels share the time x-axis; the same replica colour
repeats down the figure so the eye tracks one replica vertically:

  1. KV Load (retained)        — kv_cache_load = retained_blocks / num_gpu_blocks
  2. vLLM usage_perc           — running-only fraction (1 - free/total)
  3. MFU                       — vLLM estimated_flops / peak_flops (realtime
                                 60s window average). --peak-tflops sets the
                                 denominator (default 560 = Atlas 800I A3 / NPU)
  4. running requests          — num_requests_running
  5. waiting requests          — num_requests_waiting
  6. cumulative gpu evictions  — retained_blocks drops between tally snapshots
  7. gpu prefix hit %          — windowed local prefix-cache hit rate
  8. prefill recompute tokens  — cache-miss prefill tokens over the window
  9. external-hit rate         — cross-replica (mooncake) hits / prefix lookups

The panel hierarchy captures the two reusable shapes under the leaf panels:
:class:`SlidingPanel` (a windowed per-interval rate — :class:`MFUPanel` is its
throughput instance) and :class:`CumulativePanel` (a running total folded from
tally history — :class:`EvictPanel` is its drop-counter instance).

Sources are the ``vllm-evidence ...`` and ``kv-events tally: ... retained_blocks/replica=...``
loguru lines emitted by the kvcaware collector (parsed by :class:`LogParser`).
``usage=`` and ``flops=`` are optional (older logs parse fine; those panels stay
empty). LLM decode is bandwidth-bound, so MFU is naturally low (10-40%) in
decode-heavy phases — expected, not idle.

Usage:
    python plot_metrics.py LOG [LOG ...]
    python plot_metrics.py D.log --frac 0.3 --max-points 1500 --peak-tflops 750

The output image is written next to the first log as ``<log-name>.png``.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

# ---- fixed defaults (not exposed as CLI flags) -----------------------------
_YMAX = 1.05  # top of the 0-1 panels (load, usage)
_MFU_WINDOW_S = 60.0  # realtime MFU sliding-window length (last 1 min)
_TITLE = "KV Load / usage / MFU / run / wait / evictions / prefix-hit / prefill-recompute / external-hit (time-aligned)"


# ---- pipeline downsampling (generic, not metric-specific) -------------------
def _downsample(points: list, max_n: int) -> list:
    n = len(points)
    if max_n <= 0 or n <= max_n:
        return points
    idx = sorted(set(round(i) for i in (j * (n - 1) / (max_n - 1) for j in range(max_n))))
    return [points[i] for i in idx]


# ---- log parsing ------------------------------------------------------------
class LogParser:
    """Parses kvcaware-collector loguru lines into ``(ts, kind, fields)``.

    ``kind`` is ``'evidence'`` or ``'tally'``; ``fields`` is the regex groupdict;
    ``ts`` may be None (the caller skips it to keep the time axis honest).
    """

    # vllm-evidence replica=X kv=0.021 usage=0.450 run=240 wait=12 | TTFT=.. .. |
    #   prefill=3365 cached=41472 (hit=92.5%) decode=4175 external=0 flops=8400000000000000
    # `usage=` and `flops=` are optional so older logs still parse.
    _EVIDENCE = re.compile(
        r"vllm-evidence\s+replica=(?P<rep>\S+)\s+kv=(?P<kv>\S+)"
        r"(?:\s+usage=(?P<usage>\S+))?"
        r"\s+run=(?P<run>\S+)\s+wait=(?P<wait>\S+)"
        r".*?prefill=(?P<pre>\d+)\s+cached=(?P<cac>\d+)\s+\(hit=(?P<hit>\S+?)\)"
        r"\s+decode=\d+\s+external=(?P<ext>\d+)(?:\s+flops=(?P<flops>-?\d+))?"
    )
    # Anchor on `retained_blocks/replica=` so the earlier events={...} dict isn't captured.
    _TALLY = re.compile(r"retained_blocks/replica=(?P<d>\{[^}]*\})")
    _TS_LOGURU = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:[,.]\d+)?")
    _TS_VLLM = re.compile(r"INFO\s+(?P<ts>\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    _ASSUMED_YEAR = 2026  # vLLM engine logs omit the year; router (loguru) logs carry a full date.
    _ANCHORS = ("vllm-evidence", "retained_blocks/replica=")

    @classmethod
    def is_signal(cls, line: str) -> bool:
        """Cheap pre-filter so noise lines skip the regex scan entirely."""
        return any(a in line for a in cls._ANCHORS)

    @classmethod
    def parse(cls, line: str):
        m = cls._EVIDENCE.search(line)
        if m:
            return cls._ts(line), "evidence", m.groupdict()
        m = cls._TALLY.search(line)
        if m:
            return cls._ts(line), "tally", m.groupdict()
        return None

    @classmethod
    def _ts(cls, line: str):
        m = cls._TS_LOGURU.search(line)
        if m:
            return datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
        m = cls._TS_VLLM.search(line)
        if m:
            return datetime.strptime(f"{cls._ASSUMED_YEAR} {m.group('ts')}", "%Y %m-%d %H:%M:%S")
        return None


# ---- panels -----------------------------------------------------------------
class Panel:
    """One time-aligned subplot.

    Subclass to customise behaviour; the base draws one solid line per replica
    from ``data[self.key]`` and leaves the series unchanged.

    - :meth:`extract`   — value from a parsed evidence line (None to skip / not evidence-derived).
    - :meth:`transform` — post-process the per-replica series after truncation, before downsample.
    - :meth:`derive`    — build the series from non-evidence input (e.g. tally history); default None.
    - :meth:`draw` / :meth:`summarize` — render and report.
    """

    def __init__(
        self,
        key: str,
        ylabel: str,
        *,
        height: float = 2.0,
        ylim_top: float | None = None,
        hline: float | None = None,
        summary=None,
    ):
        self.key = key
        self.ylabel = ylabel
        self.height = height
        self.ylim_top = ylim_top
        self.hline = hline
        self._summary = summary  # callable(points) -> str

    def points_for(self, data: dict, rep: str) -> list:
        return data.get(self.key, {}).get(rep, [])

    # -- per-line / per-series behaviour (override in subclasses) --
    def extract(self, fields: dict):
        return None

    def transform(self, points: list, peak_flops: float) -> list:
        return points

    def derive(self, retained: dict):
        return None

    # -- render --
    def draw(self, ax, data, colors, order) -> None:
        for rep in order:
            pts = self.points_for(data, rep)
            if pts:
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    color=colors[rep],
                    linestyle="-",
                    linewidth=1.5,
                    alpha=0.9,
                )
        if self.hline is not None:
            ax.axhline(self.hline, color="red", linestyle=":", linewidth=1.0, alpha=0.5)
        ax.set_ylabel(self.ylabel)
        if self.ylim_top is not None:
            ax.set_ylim(top=self.ylim_top)
        ax.grid(True, alpha=0.3)

    def summarize(self, data: dict, rep: str) -> str:
        return self._summary(self.points_for(data, rep)) if self._summary else ""

    # -- numeric helpers over a [(ts, val)] series --
    @staticmethod
    def to_float(v) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    @staticmethod
    def last_val(pts):
        return pts[-1][1] if pts else float("nan")

    @staticmethod
    def minimum(pts):
        return min((p[1] for p in pts), default=float("nan"))

    @staticmethod
    def maximum(pts):
        return max((p[1] for p in pts), default=float("nan"))

    # -- summary-formatter factories (per-instance summary strategies) --
    @staticmethod
    def rng(name: str, fmt: str):
        return lambda b: (
            f"{name}[min={Panel.minimum(b):{fmt}} max={Panel.maximum(b):{fmt}} last={Panel.last_val(b):{fmt}}]"
        )

    @staticmethod
    def last(name: str, fmt: str, suffix: str = ""):
        return lambda b: f"{name}[last={Panel.last_val(b):{fmt}}{suffix}]"

    @staticmethod
    def cum(name: str):
        return lambda b: f"{name}[cum={Panel.last_val(b)}]"

    # -- value helpers for FieldPanel --
    @staticmethod
    def field(name: str):
        return lambda f: Panel.to_float(f.get(name))

    @staticmethod
    def norm_usage(v: float) -> float:
        """usage_perc is 0-1 (vLLM docs); tolerate a stray 0-100 from older scrapes."""
        if v == v and v > 1.5:
            return v / 100.0
        return v

    @staticmethod
    def ext_rate(f: dict) -> float:
        denom = int(f["pre"]) + int(f["cac"])
        return (int(f["ext"]) / denom) if denom > 0 else 0.0


class FieldPanel(Panel):
    """Panel whose per-line value comes from one evidence field (or a callable over fields)."""

    def __init__(self, key, ylabel, value, **kw):
        super().__init__(key, ylabel, **kw)
        self._value = value  # field name (str) or callable(fields) -> value | None

    def extract(self, fields: dict):
        if callable(self._value):
            return self._value(fields)
        v = fields.get(self._value)
        return Panel.to_float(v) if v is not None else None


class SlidingPanel(Panel):
    """Sliding-window statistics panel.

    Each output point is the value at ``t_j`` divided by its own elapsed interval
    ``(t_j - t_{j-1})``, averaged over a ``~window_s`` sliding window, then scaled
    to ``[0, 1]`` by ``peak`` (passed in as ``peak_flops`` from the prepare step).
    Subclasses fix the per-snapshot value via :meth:`extract`; :class:`MFUPanel` is
    the realtime-throughput instance (per-snapshot FLOPs → MFU).
    """

    def __init__(self, key: str, ylabel: str, window_s: float, **kw):
        super().__init__(key, ylabel, **kw)
        self.window_s = window_s

    def transform(self, points: list, peak_flops: float) -> list:
        return self._windowed_rate(points, self.window_s, peak_flops)

    @staticmethod
    def _windowed_rate(pts: list, window_s: float, peak: float) -> list:
        """Windowed average of ``value_j / (t_j - t_{j-1})`` over ``~window_s``, scaled by ``peak``.

        Each consecutive pair becomes throughput over its own interval; a sliding
        window then averages that throughput over ``~window_s`` — not a raw sum
        over a wall-clock band, which overcounts when >1 point lands in-band.
        """
        if len(pts) < 2 or peak <= 0:
            return []
        pts = sorted(pts, key=lambda p: p[0])
        win_q: deque = deque()
        dur_sum = val_sum = 0.0
        out = []
        for j in range(1, len(pts)):
            t1, v1 = pts[j]
            dur = (t1 - pts[j - 1][0]).total_seconds()
            if dur <= 0:
                continue
            dv = v1 if v1 == v1 else 0.0
            win_q.append((dur, dv))
            dur_sum += dur
            val_sum += dv
            # Keep ~window_s of coverage: drop oldest while the remainder still covers >= window_s.
            while len(win_q) > 1 and dur_sum - win_q[0][0] >= window_s:
                d_old, v_old = win_q.popleft()
                dur_sum -= d_old
                val_sum -= v_old
            if dur_sum > 0:
                out.append((t1, (val_sum / dur_sum) / peak))
        return out


class MFUPanel(SlidingPanel):
    """MFU: realtime FLOPs throughput as a fraction of peak, via a fixed sliding window."""

    def __init__(self, ylabel: str, **kw):
        super().__init__("mfu", ylabel, _MFU_WINDOW_S, **kw)

    def extract(self, fields: dict):
        return float(fields["flops"]) if fields.get("flops") is not None else None


class CumulativePanel(Panel):
    """Cumulative panel derived from the retained-blocks tally history.

    Walks each replica's history and folds every step into a monotonically
    non-decreasing total via :meth:`accumulate`; :meth:`derive` itself is generic.
    :class:`EvictPanel` counts the drops (block evictions); override
    :meth:`accumulate` for other cumulative semantics.
    """

    def derive(self, retained: dict):
        out = {}
        for rep, hist in retained.items():
            if not hist:
                continue
            cum = 0
            prev = hist[0][1]
            pts = []
            for t, n in hist:
                cum += self.accumulate(prev, n)
                pts.append((t, cum))
                prev = n
            out[rep] = pts
        return out

    @staticmethod
    def accumulate(prev, n):
        """Amount to add to the running total this step. Default: the drop when ``n < prev``."""
        return prev - n if n < prev else 0


class EvictPanel(CumulativePanel):
    """Cumulative GPU block evictions, derived from retained_blocks tally history."""

    def __init__(self, **kw):
        super().__init__("evict", "total gpu block evict", **kw)


def build_panels(peak_tflops: float) -> list[Panel]:
    """The ordered panel list (plot order = axis order = summary order)."""
    return [
        FieldPanel(
            "load",
            "KV Load\n(retained_blocks / num_gpu_blocks)",
            Panel.field("kv"),
            height=2.0,
            ylim_top=_YMAX,
            hline=1.0,
            summary=Panel.rng("load", ".3f"),
        ),
        FieldPanel(
            "usage",
            "vLLM usage_perc\n(running blocks / num_gpu_blocks)",
            lambda f: Panel.norm_usage(Panel.to_float(f.get("usage"))),
            height=2.0,
            ylim_top=_YMAX,
            hline=1.0,
            summary=Panel.rng("usage", ".3f"),
        ),
        MFUPanel(
            f"MFU\n(realtime {int(_MFU_WINDOW_S)}s window)\npeak={peak_tflops:.0f} TFLOPS/NPU",
            height=2.4,
            summary=Panel.last("mfu", ".3f"),
        ),
        FieldPanel(
            "run", "running requests\n(num_requests_running)", Panel.field("run"), summary=Panel.last("run", ".0f")
        ),
        FieldPanel(
            "wait", "waiting requests\n(num_requests_waiting)", Panel.field("wait"), summary=Panel.last("wait", ".0f")
        ),
        EvictPanel(height=2.4, summary=Panel.cum("evict")),
        FieldPanel(
            "hit",
            "gpu prefix hit %",
            lambda f: Panel.to_float(str(f["hit"]).rstrip("%")),
            ylim_top=100.0,
            summary=Panel.last("hit", ".1f", "%"),
        ),
        FieldPanel(
            "prefill",
            "prefill recompute\n(cache-miss tokens / window)",
            lambda f: int(f["pre"]),
            height=2.4,
            summary=Panel.last("prefill", ""),
        ),
        FieldPanel("ext", "external-hit", Panel.ext_rate, ylim_top=1.05, summary=Panel.last("ext", ".4f")),
    ]


# ---- pipeline ---------------------------------------------------------------
class Bundle:
    def __init__(self, series, retained, replicas, t_min, t_max, n_ev, n_tally, n_no_ts):
        self.series = series  # {panel_key: {replica: [(ts, val), ...]}}
        self.retained = retained  # {replica: [(ts, n), ...]}
        self.replicas = replicas
        self.t_min = t_min
        self.t_max = t_max
        self.n_ev = n_ev
        self.n_tally = n_tally
        self.n_no_ts = n_no_ts


def collect(paths, panels: list[Panel]) -> Bundle:
    """Parse logs into per-panel series + retained buffers.

    Tracks the global [t_min, t_max] during the single parse pass so later
    truncation needs no second scan.
    """
    # Cumulative panels derive from the tally history, not from evidence lines.
    extract_panels = [p for p in panels if not isinstance(p, CumulativePanel)]
    series = {p.key: defaultdict(list) for p in extract_panels}
    retained: dict = defaultdict(list)
    replicas: set[str] = set()
    t_min = t_max = None
    n_ev = n_tally = n_no_ts = 0

    for path in paths:
        try:
            f = open(path, errors="replace")
        except OSError as e:
            print(f"WARN: cannot open {path}: {e}", file=sys.stderr)
            continue
        with f:
            for line in f:
                if not LogParser.is_signal(line):
                    continue
                parsed = LogParser.parse(line)
                if parsed is None:
                    continue
                ts, kind, g = parsed
                if ts is None:
                    n_no_ts += 1
                    continue
                if t_min is None:
                    t_min = t_max = ts
                elif ts < t_min:
                    t_min = ts
                elif ts > t_max:
                    t_max = ts
                if kind == "evidence":
                    rep = g["rep"]
                    replicas.add(rep)
                    for p in extract_panels:
                        v = p.extract(g)
                        if v is not None:
                            series[p.key][rep].append((ts, v))
                    n_ev += 1
                else:  # tally
                    try:
                        d = ast.literal_eval(g["d"])
                    except Exception:
                        continue
                    for rep, n in d.items():
                        replicas.add(rep)
                        retained[rep].append((ts, int(n)))
                    n_tally += 1

    return Bundle(series, retained, replicas, t_min, t_max, n_ev, n_tally, n_no_ts)


def prepare(panels: list[Panel], series: dict, t_cut, max_points: int, peak_flops: float) -> dict:
    """Truncate → per-panel transform → downsample, uniformly for every panel."""
    data = {}
    for p in panels:
        buf = series.get(p.key, {})
        prep = {}
        for rep, pts in buf.items():
            if t_cut is not None:
                pts = [x for x in pts if x[0] <= t_cut]
            pts = p.transform(sorted(pts, key=lambda x: x[0]), peak_flops)
            if not pts:
                continue
            prep[rep] = _downsample(pts, max_points)
        data[p.key] = prep
    return data


def compute_order(load_pts: dict, replicas: set) -> list:
    """Replicas ordered by first load timestamp (stable on name)."""
    first_ts = {r: pts[0][0] for r, pts in load_pts.items() if pts}
    return sorted(replicas, key=lambda r: (first_ts.get(r, datetime.min), r))


def plot(plt, panels: list[Panel], data: dict, order: list, colors: dict, out: str) -> None:
    fig, axes = plt.subplots(
        len(panels),
        1,
        sharex=True,
        figsize=(14, 31),
        gridspec_kw={"height_ratios": [p.height for p in panels]},
    )
    for p, ax in zip(panels, axes, strict=False):
        p.draw(ax, data, colors, order)

    # Legend on the top panel only (colours repeat down the figure).
    ax_top = axes[0]
    for rep in order:
        if data.get(panels[0].key, {}).get(rep):
            ax_top.plot([], [], color=colors[rep], linestyle="-", linewidth=1.5, label=rep)
    ax_top.legend(loc="best", fontsize=8, ncol=max(1, (len(order) + 7) // 8))

    axes[-1].set_xlabel("time")
    fig.suptitle(_TITLE, y=0.995)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out, dpi=140)


def print_summary(
    panels: list[Panel], data: dict, bundle: Bundle, order: list, max_evict, peak_tflops: float, out: str
) -> None:
    print(f"OK: {bundle.n_ev} evidence lines, {bundle.n_tally} tally lines, {len(bundle.replicas)} replicas -> {out}")
    print(f"   peak={peak_tflops:.0f} TFLOPS/NPU, mfu_window={_MFU_WINDOW_S:.0f}s, max cum evictions={max_evict}")
    print("   per-replica summary:")
    for rep in order:
        parts = [p.summarize(data, rep) for p in panels]
        print(f"     {rep:>16s}: " + " ".join(parts))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Per-replica KV + MFU signals (9 time-aligned panels)")
    ap.add_argument("logs", nargs="+", help="log file(s)")
    ap.add_argument("--frac", type=float, default=1.0, help="plot first FRAC of the time window; (0,1]")
    ap.add_argument("--max-points", type=int, default=2000, help="downsample each curve; 0 disables")
    ap.add_argument(
        "--peak-tflops",
        type=float,
        default=560.0,
        help="per-NPU peak FLOPs/s in TFLOPS (MFU denominator). 560=Atlas 800I A3 FP16/NPU (default), 750=800T A3.",
    )
    args = ap.parse_args(argv)
    if not (0.0 < args.frac <= 1.0):
        ap.error(f"--frac must be in (0, 1], got {args.frac}")
    args.out = str(Path(args.logs[0]).with_suffix(".png"))  # <log-name>.png, next to the first log
    return args


def _import_mpl():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        print("ERROR: matplotlib not installed.  pip install matplotlib", file=sys.stderr)
        return None


def main(argv=None) -> int:
    args = parse_args(argv)
    plt = _import_mpl()
    if plt is None:
        return 2

    panels = build_panels(args.peak_tflops)
    bundle = collect(args.logs, panels)
    if bundle.n_no_ts:
        print(f"WARN: {bundle.n_no_ts} signal lines had no parseable timestamp — skipped", file=sys.stderr)
    if not bundle.replicas:
        print(f"ERROR: no vllm-evidence / kv-events tally lines found in {args.logs}", file=sys.stderr)
        return 1

    for p in panels:  # populate derived panels (evictions from tally history)
        derived = p.derive(bundle.retained)
        if derived is not None:
            bundle.series[p.key] = derived

    t_cut = None
    if args.frac < 1.0 and bundle.t_min is not None:
        t_cut = bundle.t_min + (bundle.t_max - bundle.t_min) * args.frac
    data = prepare(panels, bundle.series, t_cut, args.max_points, args.peak_tflops * 1e12)

    max_evict = max((pts[-1][1] for pts in data["evict"].values() if pts), default=0)
    order = compute_order(data["load"], bundle.replicas)
    cmap = plt.get_cmap("tab10" if len(order) <= 10 else "tab20")
    colors = {rep: cmap(i % cmap.N) for i, rep in enumerate(order)}

    plot(plt, panels, data, order, colors, args.out)
    print_summary(panels, data, bundle, order, max_evict, args.peak_tflops, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
