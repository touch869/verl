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

"""StickySessionStore — per-request ``request_id → replica_id`` LRU table.

Strategies read bindings through ``DataStore`` (the facade); the ``StickyDecoder``
(wired via the Balancer's ``on_acquire`` / ``on_servers_removed`` callbacks)
writes them. Backed by ``cachetools.LRUCache`` — ``get`` refreshes recency, so a
hot conversation is never evicted for a cold one. The Balancer is a single Ray
actor running ``acquire_server`` serially, so no locking is needed.

``singleton()`` returns the shared instance at ``DEFAULT_STICKY_MAX_SIZE`` (a
code constant, not configurable); tests that need a different capacity construct
``StickySessionStore(max_size=...)`` directly.
"""

from __future__ import annotations

from cachetools import LRUCache

from ..logging import get_router_logger

logger = get_router_logger("sticky-session")

DEFAULT_STICKY_MAX_SIZE = 10000


class StickySessionStore:
    """Singleton LRU store of ``request_id → replica_id`` for sticky routing.

    Read by strategies (``get`` via ``DataStore.get_sticky_binding``); written
    by ``StickyDecoder`` after each ``on_acquire`` (``put``) and cleared on
    server removal (``invalidate_replica``) or per-request expiry
    (``invalidate``).
    """

    _instance: StickySessionStore | None = None

    def __init__(self, max_size: int = DEFAULT_STICKY_MAX_SIZE) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be > 0, got {max_size}")
        self._max_size = int(max_size)
        self._table: LRUCache[str, str] = LRUCache(maxsize=self._max_size)
        logger.info(f"StickySessionStore created: max_size={self._max_size}")

    @classmethod
    def singleton(cls) -> StickySessionStore:
        """Return the shared singleton (fixed capacity; tests construct a fresh instance)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def max_size(self) -> int:
        """Configured LRU capacity."""
        return self._max_size

    def get(self, request_id: str) -> str | None:
        """Return the bound replica_id, refreshing LRU recency on hit.

        ``None`` means no sticky binding (cold start, or evicted by LRU).
        """
        if request_id in self._table:
            return self._table[request_id]
        return None

    def put(self, request_id: str, replica_id: str) -> None:
        """Bind / refresh ``request_id → replica_id``.

        Inserting an existing key refreshes recency and updates the bound
        replica (e.g. when overload-fallback routed to a different server).
        """
        self._table[request_id] = replica_id

    def invalidate(self, request_id: str) -> None:
        """Drop a single request_id's binding (e.g. stale replica hit)."""
        self._table.pop(request_id, None)

    def invalidate_replica(self, replica_id: str) -> None:
        """Drop every binding pointing at a removed replica.

        Called from the Balancer's ``on_servers_removed`` callback (via
        ``StickyDecoder``) so a server going away doesn't leave sticky entries
        routing into the void. O(n) in table size — acceptable for the rare
        elastic-removal path.
        """
        stale = [rid for rid, sid in self._table.items() if sid == replica_id]
        for rid in stale:
            self._table.pop(rid, None)
        if stale:
            logger.info(f"invalidate_replica: replica={replica_id} cleared {len(stale)} bindings")

    def __len__(self) -> int:
        return len(self._table)

    def status(self) -> dict:
        """Return a debugging snapshot of the table state."""
        return {
            "max_size": self._max_size,
            "size": len(self._table),
        }
