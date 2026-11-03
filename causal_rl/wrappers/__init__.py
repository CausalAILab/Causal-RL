"""Shared wrappers. Legacy helpers load only when explicitly requested."""
from importlib import import_module

from .ctf import CTFWrapper

_LEGACY = {
    "ClipReward": "clip_reward",
    "DiscreteActions": "discrete_actions",
    "ReacherRewardWrapper": "reacher_weighted_reward",
    "RelativePosition": "relative_position",
    "DeployPolicy": "deploy_policy",
}
__all__ = ["CTFWrapper", *_LEGACY]


def __getattr__(name):
    if name not in _LEGACY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_LEGACY[name]}", __name__), name)
    globals()[name] = value
    return value
