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

"""StickySessionStore — per-request ``request_id → replica_id`` LRU store.

Owns the sticky-session affinity table for the KVCAware router. The physical
LRU used to live in ``strategies/sticky_session.py::StickySessionTable`` and
was held by the Balancer — that was a layering leak (the Balancer docstring
claims to be a pure framework shell with no routing algorithm). State
ownership is sunk here: strategies read bindings through ``DataStore`` (the
facade), and the ``StickyDecoder`` (fed by the Balancer's ``on_acquire`` /
``on_servers_removed`` callbacks) writes bindings here via ``StickyUpdate``.

Design notes:
- **LRU eviction**: backed by ``cachetools.LRUCache`` (same dep verl's
  ``GlobalRequestLoadBalancer`` uses). Access (``get``) refreshes recency, so a
  hot conversation is never evicted in favour of a cold one.
- **No locking**: the ``KVCAwareBalancer`` is a single Ray actor running
  ``acquire_server`` serially, so the table is touched from one thread.
  Cross-thread callers must wrap it themselves.
- **Replica removal**: ``invalidate_replica`` bulk-clears every request_id
  bound to a removed replica, so stale stickiness never routes to a dead
  server. O(n) in table size; ``remove_servers`` is a rare elastic event.
- **Singleton**: ``singleton()`` returns the shared instance, fixed at
  ``DEFAULT_STICKY_MAX_SIZE`` (mirrors verl ``DEFAULT_ROUTING_CACHE_SIZE`` — a
  code constant, NOT configurable). Tests that need a different capacity
  construct a plain instance (``StickySessionStore(max_size=...)``) directly.

Reference: verl ``router.py`` ``DEFAULT_ROUTING_CACHE_SIZE = 10000``.
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
