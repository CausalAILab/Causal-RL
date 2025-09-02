from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np
import re

from causal_gym import PCH

import gymnasium
from gymnasium import spaces

from imitation import imitate

def parse_name_time(var: str) -> Tuple[str, int]:
    m = re.match(r'^([A-Za-z_]+)(\d+)$', var)
    if not m:
        raise ValueError(f'Cannot parse variable name {var}.')

    prefix, t_str = m.group(1), m.group(2)
    time = int(t_str)
    return prefix, time

def make_zset_extractor(z_sets: Dict[str, List[str]], allow_missing_time: bool = False, dtype: np.dtype = np.float32) -> Callable[[Dict[str, List[Any]], int], np.ndarray]:

    def extractor(obs: Dict[str, List[Any]], t: int) -> np.ndarray:
        key = f'X{t}'
        cond_vars = z_sets.get(key, [])
        feats: List[float] = []

        for v in cond_vars:
            prefix, ti = parse_name_time(v)
            if prefix not in obs:
                raise KeyError({f'Extractor missing prefix {prefix}.'})

            series = obs[prefix]

            if ti < 0 or ti >= len(series):
                if allow_missing_time:
                    feats.append(0.0)
                    continue

                raise IndexError(f'Time index {ti} for var {v} out of range.')

            feats.append(float(series[ti]))

        return np.array(feats, dtype=dtype)

    return extractor

def make_observed_extractor(obs_prefixes: List[str], exclude_prefixes: Optional[List[str]] = None, dtype: np.dtype = np.float32) -> Callable[[Dict[str, List[Any]], int], np.ndarray]:
    excl = set(exclude_prefixes or [])
    kept = [p for p in obs_prefixes if p not in excl]

    def extractor(obs: Dict[str, List[Any]], t: int) -> np.ndarray:
        feats: List[float] = []

        for prefix in kept:
            if prefix not in obs:
                feats.append(0.0)
                continue

            series = obs[prefix]

            if t < 0 or t >= len(series):
                feats.append(0.0)
            else:
                feats.append(float(series[t]))

        return np.array(feats, dtype=dtype)

    return extractor

def compute_z_sets_from_env(pch_env: PCH, *, action_prefix: str = "X", outcome_prefix: str = 'Y') -> Dict[str, List[str]]:
    G = imitate.parse_graph(*pch_env.get_graph)

    ou = pch_env.observed_unobserved_vars
    if isinstance(ou, dict) and "observed" in ou:
        obs_prefix = list(ou["observed"])

    z_sets = imitate.find_sequential_pi_backdoor(G, action_prefix, outcome_prefix, obs_prefix)
    return z_sets

def _peek_obs_prefixes_from_reset(pch_env: PCH) -> List[str]:
    reset_out = pch_env.reset()

    if isinstance(reset_out, tuple) and len(reset_out) >= 1:
        obs = reset_out[0]
    else:
        obs = reset_out

    return list(obs.keys())

class GAILAdapter:
    def __init__(self, pch_env: PCH, state_extractor: Callable[[Dict[str, List[Any]], int], np.ndarray], *, n_actions: Optional[int] = None, use_env_reward: bool = False, dtype: np.dtype = np.float32) -> None:
        self.env = pch_env
        self.state_extractor = state_extractor
        self.use_env_reward = use_env_reward
        self.dtype = dtype

        self._t: int = 0
        self._last_obs: Optional[Dict[str, List[Any]]] = None
        self._obs_dim: Optional[int] = None

        self.observation_space = None
        self.action_space = None
        if n_actions is not None and spaces.Discrete is not None:
            self.action_space = spaces.Discrete(n_actions)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        reset_out = self.env.reset(seed=seed)

        if isinstance(reset_out, tuple) and len(reset_out) >= 1:
            obs = reset_out[0]
        else:
            obs = reset_out

        self._t = 0
        self._last_obs = obs

        state = self.state_extractor(obs, self._t).astype(self.dtype, copy=False)
        self._obs_dim = int(state.size)

        if self.observation_space is None and spaces.Box is not None:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=self.dtype)

        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        action = int(np.asarray(action).item())

        obs, env_reward, terminated, truncated, info = self.env.do(action, show_reward=self.use_env_reward)
        done = bool(terminated or truncated)
 
        self._t += 1
        self._last_obs = obs

        state = self.state_extractor(obs, self._t).astype(self.dtype, copy=False)
        if self._obs_dim is None:
            self._obs_dim = int(state.size)

        reward = float(env_reward) if self.use_env_reward else 0.0
        return state, reward, bool(done), info

    @property
    def t(self) -> int:
        return self._t

    @property
    def obs_dim(self) -> Optional[int]:
        return self._obs_dim

    def close(self) -> None:
        self.env.close()

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)

def make_adapter_with_zsets(pch_env: Any, *, imitate_module=None, action_prefix: str = 'X', outcome_prefix: str = 'Y', use_env_reward: bool = False, dtype: np.dtype = np.float32) -> Tuple[GAILAdapter, Dict[str, List[str]]]:
    z_sets = compute_z_sets_from_env(pch_env, imitate_module=imitate_module, action_prefix=action_prefix, outcome_prefix=outcome_prefix)
    extractor = make_zset_extractor(z_sets)
    n_actions = _infer_n_actions(pch_env)
    adapter = GAILAdapter(pch_env, extractor, n_actions=n_actions, use_env_reward=use_env_reward, dtype=dtype)
    return adapter, z_sets

def make_adapter_with_observed(pch_env: Any, *, exclude: Optional[List[str]] = None, use_env_reward: bool = False, dtype: np.dtype = np.float32) -> Tuple[GAILAdapter, List[str]]:
    observed = _get_observed_prefixes(pch_env)
    extractor = make_observed_extractor(observed, exclude_prefixes=exclude, dtype=dtype)
    n_actions = _infer_n_actions(pch_env)
    adapter = GAILAdapter(pch_env, extractor, n_actions=n_actions, use_env_reward=use_env_reward, dtype=dtype)
    return adapter, observed

def _infer_n_actions(pch_env: Any) -> Optional[int]:
    if hasattr(pch_env, 'action_space') and hasattr(pch_env.action_space, 'n'):
        try:
            return int(pch_env.action_space.n)
        except Exception:
            pass

    if hasattr(pch_env, 'n_actions'):
        try:
            return int(pch_env.n_actions)
        except Exception:
            pass

    return None

def _get_observed_prefixes(pch_env: Any) -> List[str]:
    if hasattr(pch_env, 'observed_unobserved_vars'):
        ou = pch_env.observed_unobserved_vars if not callable(pch_env.observed_unobserved_vars) else pch_env.observed_unobserved_vars()
        if isinstance(ou, dict) and 'observed' in ou:
            return list(ou['observed'])

    keys = _peek_obs_prefixes_from_reset(pch_env)

    return [k for k in keys if not (k.startswith('_') or k in ('X', 'Y'))]