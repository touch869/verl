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

"""Runtime strategies (public surface: ``route`` entrypoint + registry)."""

from .base import ReplicaInfo
from .registry import StrategyRegistry
from .routing import route

# Concrete strategies self-register on import; kvc_aware lands in a later
# commit, so the registry is empty at this commit.
__all__ = ["ReplicaInfo", "StrategyRegistry", "route"]
