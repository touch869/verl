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

"""Tests for the Balancer-callback statistic path.

Covers the Phase-1 components that turn Balancer callbacks into store writes:
``StatisticEvent`` (pack contract), ``StickyDecoder`` / ``InflightDecoder``
(decoders), and ``CallbackTransport`` (the pure-forwarder transport that
registers on the Balancer). Together these mirror what the network collectors
do, but driven by the Balancer's own request-path hooks.
"""

from __future__ import annotations

import asyncio

import pytest

from verl.workers.rollout.router.kvcaware.collectors.decoder import MetricsUpdate, StickyUpdate
from verl.workers.rollout.router.kvcaware.collectors.decoder.basic.inflight import InflightDecoder
from verl.workers.rollout.router.kvcaware.collectors.decoder.basic.sticky import StickyDecoder
from verl.workers.rollout.router.kvcaware.collectors.transport.callback import (
    CallbackTransport,
    StatisticEvent,
)
from verl.workers.rollout.router.kvcaware.types import MetricKey

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


class TestStatisticEvent:
    def test_defaults_and_fields(self):
        ev = StatisticEvent("on_acquire", request_id="r1", replica_id="s0")
        assert ev.event == "on_acquire"
        assert ev.request_id == "r1"
        assert ev.replica_id == "s0"
        assert ev.server_ids == ()

    def test_frozen(self):
        ev = StatisticEvent("on_acquire")
        with pytest.raises(Exception):
            ev.event = "x"  # type: ignore[misc]

    def test_on_servers_removed_carries_tuple(self):
        ev = StatisticEvent("on_servers_removed", server_ids=("s0", "s1"))
        assert ev.server_ids == ("s0", "s1")


class TestStickyDecoder:
    def test_on_acquire_emits_put(self):
        upd = StickyDecoder().decode(
            StatisticEvent("on_acquire", request_id="r1", replica_id="s0"), ""
        )
        assert isinstance(upd, StickyUpdate)
        assert upd.action == "put"
        assert upd.request_id == "r1"
        assert upd.replica_id == "s0"

    def test_on_servers_removed_emits_invalidate_replica(self):
        upd = StickyDecoder().decode(
            StatisticEvent("on_servers_removed", server_ids=["s0", "s1"]), ""
        )
        assert isinstance(upd, StickyUpdate)
        assert upd.action == "invalidate_replica"
        assert upd.replica_ids == ("s0", "s1")

    def test_on_release_returns_none(self):
        assert StickyDecoder().decode(StatisticEvent("on_release", replica_id="s0"), "") is None

    def test_on_acquire_missing_fields_returns_none(self):
        assert StickyDecoder().decode(StatisticEvent("on_acquire"), "") is None

    def test_non_event_payload_returns_none(self):
        d = StickyDecoder()
        assert d.decode(b"bytes", "") is None
        assert d.decode("str", "") is None


class TestInflightDecoder:
    def test_on_acquire_emits_plus_one_delta(self):
        upd = InflightDecoder().decode(
            StatisticEvent("on_acquire", replica_id="s0"), ""
        )
        assert isinstance(upd, MetricsUpdate)
        assert upd.node_id == "s0"
        assert upd.metrics == {MetricKey.INFLIGHT_COUNT: 1}
        assert upd.is_delta is True

    def test_on_release_emits_minus_one_delta(self):
        upd = InflightDecoder().decode(
            StatisticEvent("on_release", replica_id="s0"), ""
        )
        assert isinstance(upd, MetricsUpdate)
        assert upd.metrics == {MetricKey.INFLIGHT_COUNT: -1}
        assert upd.is_delta is True

    def test_on_servers_removed_is_noop(self):
        # Faithful to verl: removal must NOT zero the counter — release
        # symmetry maintains it (zeroing would let a later release drive it
        # negative).
        assert (
            InflightDecoder().decode(
                StatisticEvent("on_servers_removed", server_ids=["s0"]), ""
            )
            is None
        )

    def test_non_event_payload_returns_none(self):
        assert InflightDecoder().decode(b"bytes", "") is None

    def test_missing_replica_returns_none(self):
        assert InflightDecoder().decode(StatisticEvent("on_acquire"), "") is None


class _FakeBalancer:
    """Minimal Balancer stand-in exposing register/un_register_call_back."""

    def __init__(self):
        self.callbacks: dict[str, list] = {}

    def register_call_back(self, event, fn):
        self.callbacks.setdefault(event, []).append(fn)

    def un_register_call_back(self, event, fn):
        lst = self.callbacks.get(event, [])
        if fn in lst:
            lst.remove(fn)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestCallbackTransport:
    def test_is_async_false(self):
        assert CallbackTransport(_FakeBalancer()).is_async is False

    def test_subscribe_registers_three_hooks_and_forwards(self):
        balancer = _FakeBalancer()
        transport = CallbackTransport(balancer)
        received: list = []
        _run(transport.subscribe(lambda raw, nid: received.append(raw)))

        assert set(balancer.callbacks) == {"on_acquire", "on_release", "on_servers_removed"}
        # one callback per hook
        assert all(len(lst) == 1 for lst in balancer.callbacks.values())

        balancer.callbacks["on_acquire"][0]("r1", "s0")
        balancer.callbacks["on_release"][0]("s0")
        balancer.callbacks["on_servers_removed"][0](["s1", "s2"])

        assert received == [
            StatisticEvent("on_acquire", request_id="r1", replica_id="s0"),
            StatisticEvent("on_release", replica_id="s0"),
            StatisticEvent("on_servers_removed", server_ids=("s1", "s2")),
        ]

    def test_stop_unregisters_all(self):
        balancer = _FakeBalancer()
        transport = CallbackTransport(balancer)
        _run(transport.subscribe(lambda raw, nid: None))
        transport.stop()
        assert all(not lst for lst in balancer.callbacks.values())

    def test_stop_is_idempotent(self):
        transport = CallbackTransport(_FakeBalancer())
        transport.subscribe  # not yet subscribed
        transport.stop()  # must not raise even with empty registry


class TestCollectorCallbackIntegration:
    """End-to-end: Collector(CallbackTransport, decoder) → handler → DataStore.

    Exercises the is_async=False start path (tmp loop runs the loop-free
    subscribe), the handler's StickyUpdate/MetricsUpdate dispatch, and the
    store writes — the whole Phase-1 statistic chain.
    """

    @pytest.fixture(autouse=True)
    def _reset_singletons(self):
        from verl.workers.rollout.router.kvcaware.store.metrics_store import MetricsStore
        from verl.workers.rollout.router.kvcaware.store.sticky_session_store import StickySessionStore

        StickySessionStore._instance = None
        MetricsStore._instance = None
        yield
        StickySessionStore._instance = None
        MetricsStore._instance = None

    def test_sticky_collector_writes_binding_on_acquire(self):
        from verl.workers.rollout.router.kvcaware.collectors.collector import Collector
        from verl.workers.rollout.router.kvcaware.store.data_store import DataStore

        balancer = _FakeBalancer()
        collector = Collector(CallbackTransport(balancer), StickyDecoder())
        collector.start()
        try:
            balancer.callbacks["on_acquire"][0]("r1", "s0")
            assert DataStore().get_sticky_binding("r1") == "s0"
        finally:
            collector.stop()

    def test_inflight_collector_applies_acquire_release_delta(self):
        from verl.workers.rollout.router.kvcaware.collectors.collector import Collector
        from verl.workers.rollout.router.kvcaware.store.data_store import DataStore

        balancer = _FakeBalancer()
        collector = Collector(CallbackTransport(balancer), InflightDecoder())
        collector.start()
        try:
            balancer.callbacks["on_acquire"][0]("r1", "s0")  # +1
            balancer.callbacks["on_acquire"][0]("r2", "s0")  # +1
            balancer.callbacks["on_release"][0]("s0")  # -1
            assert DataStore().get_metric("s0", MetricKey.INFLIGHT_COUNT) == 1
        finally:
            collector.stop()
