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

"""Tests for MetricsStore.incr and DataStore sticky-session delegation.

Covers the Phase-1 store-layer additions: incremental writes (inflight ±1,
keeping the decoder stateless) and the DataStore facade delegating to the
``StickySessionStore`` singleton.
"""

from __future__ import annotations

import pytest

from verl.workers.rollout.router.kvcaware.store.data_store import DataStore
from verl.workers.rollout.router.kvcaware.store.metrics_store import MetricsStore
from verl.workers.rollout.router.kvcaware.store.sticky_session_store import StickySessionStore
from verl.workers.rollout.router.kvcaware.types import MetricKey

pytestmark = [pytest.mark.ut, pytest.mark.cpu]


# ── MetricsStore.incr (plain instances — isolated, not the singleton) ──


class TestMetricsStoreIncr:
    def test_incr_from_default(self):
        s = MetricsStore()
        s.incr("n0", MetricKey.INFLIGHT_COUNT)
        assert s.get("n0", MetricKey.INFLIGHT_COUNT) == 1
        s.incr("n0", MetricKey.INFLIGHT_COUNT, 1)
        assert s.get("n0", MetricKey.INFLIGHT_COUNT) == 2

    def test_incr_negative_delta(self):
        s = MetricsStore()
        s.incr("n0", MetricKey.INFLIGHT_COUNT, 5)
        s.incr("n0", MetricKey.INFLIGHT_COUNT, -2)
        assert s.get("n0", MetricKey.INFLIGHT_COUNT) == 3

    def test_incr_default_delta_is_one(self):
        s = MetricsStore()
        s.incr("n0", MetricKey.INFLIGHT_COUNT)
        assert s.get("n0", MetricKey.INFLIGHT_COUNT) == 1

    def test_incr_isolates_nodes(self):
        s = MetricsStore()
        s.incr("n0", MetricKey.INFLIGHT_COUNT)
        s.incr("n1", MetricKey.INFLIGHT_COUNT, 3)
        assert s.get("n0", MetricKey.INFLIGHT_COUNT) == 1
        assert s.get("n1", MetricKey.INFLIGHT_COUNT) == 3

    def test_incr_does_not_clobber_other_keys(self):
        s = MetricsStore()
        s.refresh({"n0": {MetricKey.NUM_REQUESTS_RUNNING: 7}})
        s.incr("n0", MetricKey.INFLIGHT_COUNT)
        assert s.get("n0", MetricKey.NUM_REQUESTS_RUNNING) == 7
        assert s.get("n0", MetricKey.INFLIGHT_COUNT) == 1

    def test_incr_unknown_key_raises(self):
        s = MetricsStore()
        with pytest.raises(KeyError):
            s.incr("n0", "not_a_real_key")


# ── DataStore sticky-session delegation + inflight (singleton-backed) ──


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Isolate each test from the global MetricsStore / StickySessionStore singletons."""
    StickySessionStore._instance = None
    MetricsStore._instance = None
    yield
    StickySessionStore._instance = None
    MetricsStore._instance = None


class TestDataStoreStickyDelegation:
    def test_put_then_get(self):
        ds = DataStore()
        ds.put_sticky_binding("r1", "s0")
        assert ds.get_sticky_binding("r1") == "s0"

    def test_get_missing_is_none(self):
        assert DataStore().get_sticky_binding("ghost") is None

    def test_invalidate_binding(self):
        ds = DataStore()
        ds.put_sticky_binding("r1", "s0")
        ds.invalidate_sticky_binding("r1")
        assert ds.get_sticky_binding("r1") is None

    def test_invalidate_replica_clears_all_bound(self):
        ds = DataStore()
        ds.put_sticky_binding("r1", "s0")
        ds.put_sticky_binding("r2", "s1")
        ds.put_sticky_binding("r3", "s0")
        ds.invalidate_sticky_replica("s0")
        assert ds.get_sticky_binding("r1") is None
        assert ds.get_sticky_binding("r3") is None
        assert ds.get_sticky_binding("r2") == "s1"

    def test_sticky_status_reports_size(self):
        ds = DataStore()
        ds.put_sticky_binding("r1", "s0")
        ds.put_sticky_binding("r2", "s1")
        assert ds.sticky_status()["size"] == 2
