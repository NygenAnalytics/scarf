"""Harmony correction for reduced cell embeddings."""

from .api import fit_harmony, run_harmony
from .models import ClusterFn, HarmonyResult
from .optimizer import Harmony, moe_correct_ridge, safe_entropy

__all__ = [
    "ClusterFn",
    "Harmony",
    "HarmonyResult",
    "fit_harmony",
    "moe_correct_ridge",
    "run_harmony",
    "safe_entropy",
]

for _public_object in (
    Harmony,
    HarmonyResult,
    fit_harmony,
    moe_correct_ridge,
    run_harmony,
    safe_entropy,
):
    _public_object.__module__ = __name__

for _method in Harmony.__dict__.values():
    if isinstance(_method, staticmethod):
        _method = _method.__func__
    if callable(_method) and hasattr(_method, "__module__"):
        _method.__module__ = __name__

del _method
del _public_object
