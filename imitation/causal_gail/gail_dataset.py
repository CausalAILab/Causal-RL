from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np

def convert_records_to_arrays(records: Sequence[Dict[str, Any]], state_extractor: Callable[[Dict[str, List[Any]], int], np.ndarray], discrete: bool = True, num_actions: Optional[int] = None, one_hot_actions: bool = True, strict: bool = False, dtype: np.dtype = np.float32) -> Tuple[np.ndarray, np.ndarray]:
    step_iter = _iterate_steps(records, strict=strict)

    feats: List[np.ndarray] = []
    acts_raw: List[Any] = []

    max_action_id: int = -1

    for step in step_iter:
        obs = step['obs']
        t = step['t']
        a = step['action']

        try:
            x = state_extractor(obs, t)

        except Exception as e:
            if strict:
                raise

            continue

        x = _ensure_1d_float_array(x, dtype=dtype)
        feats.append(x)

        if discrete:
            if not isinstance(a, (int, np.integer)):
                if strict:
                    raise TypeError(f"discrete action must be int at step t={t}, got {type(a)}")

                try:
                    a = int(np.asarray(a).item())

                except Exception:
                    continue

            acts_raw.append(int(a))

            if a > max_action_id:
                max_action_id = int(a)

        else:
            a_arr = _ensure_1d_float_array(a, dtype=dtype)
            acts_raw.append(a_arr)

    rx = np.vstack(feats).astype(dtype, copy=False)

    if discrete:
        if one_hot_actions:
            ra = _to_one_hot(np.asarray(acts_raw, dtype=np.int64), num_actions).astype(dtype, copy=False)

        else:
            ra = np.asarray(acts_raw, dtype=np.int64).reshape(-1, 1)

    else:
        ra = _stack_action_vectors(acts_raw, dtype=dtype)

    return rx, ra

def save_expert_npz(path: str, rx: np.ndarray, ra: np.ndarray, **extra_arrays: np.ndarray) -> None:
    np.savez_compressed(path, rx=rx, ra=ra, **extra_arrays)

def load_expert_npz(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=False)
    rx = data['rx']
    ra = data['ra']
    extras = {k: data[k] for k in data.files if k not in ('rx', 'ra')}
    return rx, ra, extras

def build_expert_arrays_from_env(env: Any, state_extractor: Callable[[Dict[str, List[Any]], int], np.ndarray], episodes: int = 10, max_steps_per_episode: Optional[int] = None, discrete: bool = True, num_actions: Optional[int] = None, one_hot_actions: bool = True, use_see: bool = True, seed: Optional[int] = None, strict: bool = False, dtype: np.dtype = np.float32) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed if seed is not None else 0)

    feats: List[np.ndarray] = []
    acts: List[int] = []

    for ep in range(episodes):
        reset_out = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        t = 0
        steps = 0
        done = False

        x = _ensure_1d_float_array(state_extractor(obs, t), dtype=dtype)

        while not done:
            if use_see:
                obs, reward, terminated, truncated, info = env.see()
                done = bool(terminated or truncated)

                a = None
                if 'natural_action' in info:
                    a = int(info['natural_action'])

                if 'action' in info:
                    a = int(info['action'])

            x_t = _ensure_1d_float_array(state_extractor(obs, t), dtype=dtype)
            feats.append(x_t)
            acts.append(int(a))

            t += 1
            steps += 1
            if done or (max_steps_per_episode is not None and steps >= max_steps_per_episode):
                break

    rx = np.vstack(feats).astype(dtype, copy=False)

    if discrete:
        if one_hot_actions:
            if num_actions is None:
                max_id = int(np.max(acts)) if acts else -1
                num_actions = max_id + 1

            ra = _to_one_hot(np.asarray(acts, dtype=np.int64), num_actions).astype(dtype, copy=False)

        else:
            ra = np.asarray(acts, dtype=np.int64).reshape(-1, 1)

    else:
        raise NotImplementedError

    return rx, ra

def _iterate_steps(records: Sequence[Dict[str, Any]], strict: bool = False) -> Iterator[Dict[str, Any]]:
    for item in records:
        obs = item.get('obs', None)

        if 'actions' in item and isinstance(item['actions'], (list, tuple, np.ndarray)):
            actions_seq = item['actions']
            T = _infer_T_from_obs(obs)

            T_eff = min(T, len(actions_seq))
            for t in range(T_eff):
                a = actions_seq[t]
                yield {'obs': obs, 't': t, 'actions': a}
            continue

        if 'action' in item and not isinstance(item['action'], (list, tuple, np.ndarray)):
            t = item.get('t', None)

            if t is None:
                t = _infer_T_from_obs(obs) - 1
                if t < 0:
                    if strict:
                        raise ValueError('cannot infer time index t for step-major record.')
                    else:
                        continue

            yield {'obs': obs, 't': int(t), 'actions': item['action']}
            continue

        if strict:
            raise ValueError('unsupported record format')

def _infer_T_from_obs(obs: Dict[str, List[Any]]) -> int:
    lengths = [len(v) for v in obs.values() if isinstance(v, (list, tuple, np.ndarray))]

    if not lengths:
        return 0

    return int(min(lengths))

def _ensure_1d_float_array(x: Union[np.ndarray, Sequence[float]], dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype).reshape(-1)
    return arr

def _to_one_hot(actions: np.ndarray, num_actions: int) -> np.ndarray:
    if actions.ndim != 1:
        actions = actions.reshape(-1)

    if np.any(actions < 0) or np.any(actions >= num_actions):
        raise ValueError(f'action id out of range for num_actions={num_actions}.')

    N = actions.shape[0]
    out = np.zeros((N, num_actions), dtype=np.float32)
    out[np.arange(N), actions.astype(np.int64)] = 1.0
    return out

def _stack_action_vectors(vectors: List[np.ndarray], dtype: np.dtype) -> np.ndarray:
    if not vectors:
        return np.zeros((0, 0), dtype=dtype)

    widths = [np.asarray(v).reshape(-1).size for v in vectors]
    w = int(widths[0])

    if any(w_i != w for w_i in widths):
        raise ValueError(f'inconsistent continuous action widths')

    return np.vstack([np.asarray(v, dtype=dtype).reshape(-1) for v in vectors])