"""Active-labeling / RL external adapters (pipeline step 11).

Reinforcement-learning agents that rank which unlabeled instances to
label next (active learning), reducing annotation cost. Subprocess
adapters like the rest, but their templates launch a training/rollout
loop rather than a single inference.

* :class:`GymnasiumAdapter` — Gymnasium custom env rollout
  (``agent_episode`` + ``reward_trace`` outputs).
* :class:`StableBaselines3Adapter` — Stable-Baselines3 agent
  training/rollback (``policy_checkpoint`` + ``decision_trace`` outputs).
"""

from __future__ import annotations

from .gymnasium import GymnasiumAdapter
from .stable_baselines3 import StableBaselines3Adapter

__all__ = ["GymnasiumAdapter", "StableBaselines3Adapter"]