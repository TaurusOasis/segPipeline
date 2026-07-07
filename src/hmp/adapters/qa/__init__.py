"""QA-stage external adapters (pipeline step 8, temporal consistency).

* :class:`RaftAdapter` — RAFT optical-flow temporal-consistency QA. When the
  external RAFT repo is not available, :meth:`RaftAdapter.run_qa` falls back
  to the CPU :func:`hmp.eval.temporal_metrics.frame_diff_flow` path (zero
  flow), matching the roadmap's "RAFT/GMFlow with frame-diff fallback for CPU
  smoke tests", and still writes the two expected outputs
  (``temporal_error`` + ``flow_consistency_score``) so the contract validates.
"""

from __future__ import annotations

from .raft import RaftAdapter

__all__ = ["RaftAdapter"]