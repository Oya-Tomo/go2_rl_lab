"""Project-local physics backend extensions."""

from .randomize import BodyComBucketRandomizationCfg, RandomizedMJWarpSolverCfg

__all__ = [
    "BodyComBucketRandomizationCfg",
    "NewtonMJWarpResettableManager",
    "RandomizedMJWarpSolverCfg",
    "ResettableNewtonMJWarpManager",
]


def __getattr__(name: str):
    """Load solver implementations only when the active runtime requests them."""
    if name in {"NewtonMJWarpResettableManager", "ResettableNewtonMJWarpManager"}:
        from . import resettable_mjwarp_manager

        return getattr(resettable_mjwarp_manager, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
