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

"""Strategy-specific configs.

Concrete routing strategy configs. The matching runtime strategy classes
(e.g. ``KVCacheAwareStrategy``) live under ``verl.workers.rollout.router.kvcaware.strategies``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import Layer, SlowCut
from .base import ConfigError, StrategyConfig, _multiline_repr


@dataclass(repr=False)
class KVCAwareStrategyConfig(StrategyConfig):
    """Config for KVCache-Aware routing strategy.

    S = α × S_cache + (1-α) × S_load
    """

    alpha: float = 0.7
    load_threshold: float = 0.9
    layer_weights: dict[Layer, float] = field(default_factory=lambda: {Layer.GPU: 0.7, Layer.CPU: 0.2, Layer.SSD: 0.1})
    # Sticky short-circuit: when True, a returning session is sent back to its
    # bound replica only if that replica is NOT overloaded (load > load_threshold).
    memory_overload_filter: bool = True
    # Fallback scoring mode used after the sticky short-circuit misses.
    slow_cut: SlowCut = SlowCut.PREFIX_LOAD_AWARE
    # Capacity-gate fraction (CAPACITY_TOKEN_AWARE only): replicas whose free
    # token capacity ``cap × (1 - kv_cache_usage_perc)`` is below
    # ``cap × capacity_filter_frac`` are excluded before picking the one with
    # the largest remaining capacity after assigning this request's prefill.
    capacity_filter_frac: float = 0.05

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.load_threshold < 1:
            raise ConfigError(f"load_threshold must be in (0, 1), got {self.load_threshold}")
        if not isinstance(self.memory_overload_filter, bool):
            raise ConfigError(f"memory_overload_filter must be a bool, got {self.memory_overload_filter!r}")
        if not 0.0 <= self.capacity_filter_frac < 1.0:
            raise ConfigError(f"capacity_filter_frac must be in [0, 1), got {self.capacity_filter_frac}")
        # Normalize yaml str → SlowCut (also validates the value is a known mode).
        try:
            self.slow_cut = SlowCut(self.slow_cut)
        except ValueError as exc:
            raise ConfigError(f"slow_cut must be one of {[m.value for m in SlowCut]}, got {self.slow_cut!r}") from exc
        # Normalize yaml str keys → Layer (also validates each key is a known layer).
        try:
            self.layer_weights = {Layer(k): v for k, v in self.layer_weights.items()}
        except ValueError as exc:
            raise ConfigError(f"layer_weights keys must be layer names, got {set(self.layer_weights)}") from exc
        if set(self.layer_weights.keys()) != {Layer.GPU, Layer.CPU, Layer.SSD}:
            raise ConfigError(
                f"layer_weights must be exactly {{{Layer.GPU}, {Layer.CPU}, {Layer.SSD}}}, "
                f"got {set(self.layer_weights.keys())}"
            )
        weights_sum = sum(self.layer_weights.values())
        if abs(weights_sum - 1.0) > 1e-6:
            raise ConfigError(f"layer_weights values must sum to 1.0, got {weights_sum}")

    __repr__ = _multiline_repr
