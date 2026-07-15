#!/usr/bin/env python3
"""Standalone evidence + retained-cache collector for groups without the KVCAware balancer.

A/C (verl-sticky routing) groups run no balancer, so they get no built-in collector.
This script is a thin wrapper around ``CollectorManager`` (the same one the balancer
uses) so A/C emit the *same* ``vllm-evidence`` + ``kv-events tally`` log lines as B/D —
with ZERO routing involvement (verl keeps routing; this process only observes).

Endpoint discovery is self-contained (no Ray dependency): it probes ``/proc/net/tcp``
LISTEN ports — HTTP ports that serve ``vllm:`` metrics are /metrics endpoints; ports
that deliver ``kv-events``-topic ZMQ messages are kv-events publishers (sub-only,
no replay — ``ZMQTransport`` supports replay="" for this mode).

Usage::

    python standalone_collector.py [--interval 1] [--kv-probe-timeout 3]

Output: vllm-evidence + kv-events tally lines on stdout (same format as the balancer).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import socket
import sys
import time

import httpx
import zmq

# Must be set before importing llm_router (which may read env at import time).
os.environ.setdefault("PYTHONHASHSEED", "0")

# Collector/Transport/Decoder use stdlib logging (not loguru); configure it
# so evidence/tally/zmq logs at INFO/DEBUG reach stdout. Silence httpx INFO
# (per-probe request logs flood the output during endpoint discovery).
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s", stream=sys.stdout)
logging.getLogger("httpx").setLevel(logging.WARNING)

from verl.workers.rollout.router.collectors.manager import CollectorManager
from verl.workers.rollout.router.config.collector import CollectorConfig
from verl.workers.rollout.router.logging import get_router_logger

logger = get_router_logger("standalone-collector")

KV_EVENTS_TOPIC = "kv-events"


# ── Endpoint discovery (self-contained, no Ray) ──────────────────────────


def _detect_local_ip() -> str:
    """Outbound local IP (the host IP vLLM binds to; 127.0.0.1 returns 401)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _listen_sockets() -> list[tuple[str, int]]:
    """(bind_ip, port) for each TCP LISTEN socket from /proc/net/tcp.

    Reads the actual bind IP (decoded from hex, little-endian) so we can probe the
    real interface vLLM bound to — not just a guessed ext_ip/127.0.0.1 (which misses
    on hosts where vLLM binds /metrics to a specific non-loopback IP).
    """
    socks: list[tuple[str, int]] = []
    try:
        with open("/proc/net/tcp") as f:
            for line in f.readlines()[1:]:
                parts = line.split()
                if len(parts) < 4 or parts[3] != "0A":  # 0A = TCP_LISTEN
                    continue
                hex_addr, _, hex_port = parts[1].partition(":")
                try:
                    port = int(hex_port, 16)
                    ip = ".".join(str(b) for b in reversed(bytes.fromhex(hex_addr)))
                except ValueError:
                    continue
                socks.append((ip, port))
    except OSError as exc:
        logger.warning(f"could not read /proc/net/tcp: {exc}")
    return socks


async def discover_vllm_metrics_endpoints() -> dict[str, str]:
    """Return ``{replica_id: "ip:port"}`` for each vLLM HTTP /metrics endpoint.

    Probes all LISTEN ports concurrently on their real bind IP (+ 127.0.0.1 / ext_ip
    fallbacks for 0.0.0.0 binds). Logs a breakdown so discovery failures are visible.
    """
    socks = _listen_sockets()
    ext_ip = _detect_local_ip()
    # probe real bind IP, plus loopback + ext_ip fallbacks (covers 0.0.0.0 binds)
    targets: set[tuple[str, int]] = set()
    for ip, p in socks:
        targets.add((ip, p))
        targets.add(("127.0.0.1", p))
        targets.add((ext_ip, p))

    stats = {"hit": 0, "200_no_vllm": 0, "other_status": 0, "error": 0}

    async def _probe(client: httpx.AsyncClient, ip: str, p: int) -> tuple[int, str] | None:
        try:
            r = await client.get(f"http://{ip}:{p}/metrics")
            if r.status_code == 200:
                if "vllm:" in r.text:
                    stats["hit"] += 1
                    return (p, f"{ip}:{p}")
                stats["200_no_vllm"] += 1
            else:
                stats["other_status"] += 1
        except Exception:
            stats["error"] += 1
        return None

    found: dict[str, str] = {}
    # timeout 5s (not 1.5): under multi-replica startup/graph-capture the /metrics scrape
    # is slow; 1.5s timed most out -> hit fluctuated 0-2. Probes are concurrent so total
    # discovery time ~= this timeout.
    async with httpx.AsyncClient(trust_env=False, timeout=5.0) as client:
        results = await asyncio.gather(*[_probe(client, ip, p) for ip, p in sorted(targets)])
    for res in results:
        if res and res[0] not in found:
            found[f"replica-{res[0]}"] = res[1]
    logger.info(f"/metrics discovery: {len(socks)} LISTEN sockets, {len(targets)} probes -> "
                f"hit={stats['hit']} 200_no_vllm={stats['200_no_vllm']} "
                f"other={stats['other_status']} err={stats['error']} | found {len(found)}")
    return found


def discover_kv_events_endpoints(probe_timeout: float = 3.0) -> dict[str, list[str]]:
    """Probe LISTEN ports with a topic-filtered ZMQ SUB.

    Returns ``{replica_id: [sub_addr, "", "zmq", "kv-events"]}`` — replay left
    empty (sub-only mode); ``ZMQTransport`` handles this.
    """
    ports = sorted({p for _, p in _listen_sockets()})
    ext_ip = _detect_local_ip()
    ctx = zmq.Context()
    socks: dict[zmq.Socket, str] = {}
    for ip in (ext_ip, "127.0.0.1"):
        for p in sorted(ports):
            try:
                sub = ctx.socket(zmq.SUB)
                sub.connect(f"tcp://{ip}:{p}")
                sub.setsockopt_string(zmq.SUBSCRIBE, KV_EVENTS_TOPIC)
                sub.setsockopt(zmq.RCVTIMEO, 0)
                socks[sub] = f"{ip}:{p}"
            except zmq.ZMQError:
                continue
    time.sleep(probe_timeout)
    hit_ports: set[int] = set()
    for sub in list(socks):
        try:
            if sub.poll(100):
                hit_ports.add(int(socks[sub].split(":")[-1]))
        except Exception:
            pass
        finally:
            try:
                sub.close(0)
            except Exception:
                pass
    ctx.term()

    endpoints: dict[str, list[str]] = {}
    for p in sorted(hit_ports):
        # sub_addr, replay_addr (empty=skip), publisher, topic
        endpoints[f"replica-{p}"] = [f"{ext_ip}:{p}", "", "zmq", KV_EVENTS_TOPIC]
    return endpoints


# ── Main ─────────────────────────────────────────────────────────────────


async def main(interval: float, discover_timeout: float, kv_probe_timeout: float,
               num_replicas: int = 0) -> None:
    """Discover vLLM endpoints, start CollectorManager, block until interrupted.

    ``num_replicas``: expected endpoint count (data-parallel replicas). If >0, keep
    probing until BOTH /metrics and kv-events reach it (or deadline) — vLLM replicas
    start staggered, so exiting on the first endpoint found would miss the rest (on
    910C 16-replica this left the collector with 1/16). Best-effort: at the deadline
    it proceeds with whatever was found (warns if short).
    """
    def _enough(found: int) -> bool:
        return found >= num_replicas if num_replicas > 0 else found > 0

    # Discover /metrics HTTP endpoints — wait until all replicas are up.
    logger.info(f"discovering vLLM /metrics endpoints (expect {num_replicas or '>=1'})...")
    metrics_endpoints: dict[str, str] = {}
    deadline = time.monotonic() + discover_timeout
    while time.monotonic() < deadline:
        metrics_endpoints = await discover_vllm_metrics_endpoints()
        if _enough(len(metrics_endpoints)):
            break
        logger.info(f"found {len(metrics_endpoints)}/{num_replicas or '?'} /metrics; "
                    f"vllm still starting, retry in 5s ({int(deadline - time.monotonic())}s left)")
        await asyncio.sleep(5)
    if not metrics_endpoints:
        logger.error(f"no /metrics endpoints found within {discover_timeout}s; exiting")
        return
    if num_replicas > 0 and len(metrics_endpoints) < num_replicas:
        logger.warning(f"only {len(metrics_endpoints)}/{num_replicas} /metrics endpoints up by "
                       f"deadline; proceeding (pressure will undercount missing replicas)")
    logger.info(f"discovered {len(metrics_endpoints)} /metrics endpoint(s): "
                f"{list(metrics_endpoints.values())}")

    # Discover kv-events zmq endpoints (probe, sub-only). Wait until all replicas emit
    # (active replicas emit BlockStored; in a real run all replicas get traffic -> all
    # emit). Best-effort at deadline; under sparse smoke load fewer replicas emit.
    kv_deadline = time.monotonic() + discover_timeout
    kv_endpoints: dict[str, list[str]] = {}
    while time.monotonic() < kv_deadline:
        kv_endpoints = await asyncio.to_thread(discover_kv_events_endpoints, kv_probe_timeout)
        if _enough(len(kv_endpoints)):
            break
        logger.info(f"found {len(kv_endpoints)}/{num_replicas or '?'} kv-events; "
                    f"retry in 5s ({int(kv_deadline - time.monotonic())}s left)")
        await asyncio.sleep(5)
    if kv_endpoints:
        logger.info(f"discovered {len(kv_endpoints)} kv-events endpoint(s): "
                    f"{list(kv_endpoints.keys())}")
    else:
        logger.warning(f"no kv-events endpoints found within {discover_timeout}s "
                       f"(KV churn stayed sparse); kv_events tallies will be empty — "
                       f"/metrics evidence still collected")

    # Build CollectorConfig (same defaults as the balancer).
    config = CollectorConfig()
    config.http_polling["polling_interval"] = interval

    # Pair /metrics and kv-events endpoints by sorted port order — each vllm
    # replica has one of each; sorting by port and zipping gives a best-effort
    # 1:1 mapping (exact pairing requires Ray RPC, which standalone lacks).
    # This ensures kv_cache_load (retained/num_gpu_blocks) uses consistent IDs.
    paired_kv: dict[str, list[str]] = {}
    if kv_endpoints and metrics_endpoints:
        m_sorted = sorted(metrics_endpoints.items(), key=lambda x: int(x[1].split(":")[-1]))
        k_sorted = sorted(kv_endpoints.items(), key=lambda x: int(x[1][0].split(":")[-1]))
        for (m_id, _), (_, k_addrs) in zip(m_sorted, k_sorted):
            paired_kv[m_id] = k_addrs  # use /metrics ID for kv-events too
    else:
        paired_kv = kv_endpoints

    collection_names = ["vllm_metrics", "vllm_zmq"] if paired_kv else ["vllm_metrics"]

    manager = CollectorManager(
        config,
        collection_names,
        server_addresses=metrics_endpoints,
        kv_event_endpoints=paired_kv,
    )
    manager.start()
    logger.info(f"CollectorManager started ({collection_names}); press Ctrl-C to stop")

    # Periodic kv_cache_load log (retained occupancy) — same signal the balancer's
    # strategy uses for routing, so A/C logs are directly comparable to B/D.
    # DataStore holds the retained blocks (from kv-events) + num_gpu_blocks (from
    # /metrics cache_config_info); kv_cache_load divides them.
    from verl.workers.rollout.router.store.data_store import DataStore

    store = DataStore()

    async def _log_kv_cache_load(period: float) -> None:
        while True:
            await asyncio.sleep(period)
            parts = []
            for node_id in sorted(store.get_metric_node_ids()):
                load = store.kv_cache_load(node_id)
                if load is not None:
                    parts.append(f"{node_id}={load:.3f}")
            if parts:
                logger.info(f"kv_cache_load: {' '.join(parts)}")

    try:
        await _log_kv_cache_load(interval * 30)  # every ~30s at default 1s interval
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        manager.stop()
        logger.info("CollectorManager stopped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Standalone KVCAware observer (no balancer)")
    ap.add_argument("--interval", type=float, default=1.0, help="/metrics polling interval (s)")
    ap.add_argument("--discover-timeout", type=float, default=600.0,
                    help="max seconds to wait for /metrics AND kv-events endpoints at startup")
    ap.add_argument("--kv-probe-timeout", type=float, default=3.0,
                    help="seconds to listen for kv-events topic when probing")
    ap.add_argument("--num-replicas", type=int, default=0,
                    help="expected endpoint count (data-parallel replicas); wait until BOTH "
                         "/metrics and kv-events reach it before collecting (0 = any non-empty)")
    args = ap.parse_args()
    logger.info(f"standalone collector start (interval={args.interval}s, "
                f"kv_probe={args.kv_probe_timeout}s, expect={args.num_replicas or '?'} replicas)")
    try:
        asyncio.run(main(args.interval, args.discover_timeout, args.kv_probe_timeout,
                         args.num_replicas))
    except KeyboardInterrupt:
        pass
