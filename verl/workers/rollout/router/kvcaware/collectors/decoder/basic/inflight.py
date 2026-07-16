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

"""InflightDecoder — Balancer callback → MetricsUpdate delta (inflight ±1).

Mirrors verl ``GlobalRequestLoadBalancer._inflight_requests``: acquire bumps
the chosen replica's in-flight count by +1, release decrements it by -1. The
decoder is stateless — it emits only the signed delta; the store's ``incr``
owns the running counter.

Event → delta mapping:
- ``on_acquire(replica_id)`` → ``MetricsUpdate(replica_id, {INFLIGHT_COUNT: +1}, is_delta=True)``
- ``on_release(replica_id)`` → ``MetricsUpdate(replica_id, {INFLIGHT_COUNT: -1}, is_delta=True)``

``on_servers_removed`` is intentionally a no-op (returns ``None``): verl never
removes servers and maintains ``_inflight_requests`` purely via symmetric
acquire/release. Clearing on removal would be unsafe — in-flight requests for
a removed replica still complete and fire their ``on_release`` (-1), which
would drive the counter negative if we'd zeroed it on removal. Faithful
simulation = don't touch the counter on removal.
"""

from __future__ import annotations

from typing import Any

from ....collectors.decoder import Decoder, MetricsUpdate
from ....collectors.transport.callback import StatisticEvent
from ....types import MetricKey


class InflightDecoder(Decoder):
    """Decode ``StatisticEvent`` → inflight ``MetricsUpdate`` delta."""

    def decode(self, raw_data: bytes | str | Any, node_id: str) -> MetricsUpdate | None:
        """Dispatch on acquire/release; ignore other events / non-event payloads."""
        if not isinstance(raw_data, StatisticEvent):
            return None

        event = raw_data
        if event.replica_id is None:
            return None

        if event.event == "on_acquire":
            return MetricsUpdate(
                node_id=event.replica_id,
                metrics={MetricKey.INFLIGHT_COUNT: 1},
                is_delta=True,
            )
        if event.event == "on_release":
            return MetricsUpdate(
                node_id=event.replica_id,
                metrics={MetricKey.INFLIGHT_COUNT: -1},
                is_delta=True,
            )
        return None
