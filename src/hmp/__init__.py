"""human-matting-pipeline (hmp) core package.

The core package is intentionally lightweight: importing ``hmp`` must NOT pull
in any heavy / GPU dependency (torch, ultralytics, SAM, RKNN, ...). Those are
lazy-imported inside the modules that need them.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]