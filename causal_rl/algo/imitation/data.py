import numpy as np
import torch
from torch.utils.data import Dataset

class ExpertDataset(Dataset):
    def __init__(self, records, cond_vars, action_var, continuous=False):
        x_list = []
        y_list = []

        for r in records:
            obs = r.get('obs', {})

            xs = []

            for v in cond_vars:
                var = v[0]
                step = int(v[1:])
                val = obs.get(var, [])[step]

                if hasattr(val, 'shape') and len(val.shape) > 0:
                    xs.extend(val.tolist())
                else:
                    xs.append(val)

            if action_var not in r:
                continue

            y = r[action_var]

            x_list.append(xs)
            y_list.append(y)

        if len(x_list) == 0:
            raise ValueError(f'No valid data points in ExpertDataset with cond_vars={cond_vars}')

        x_arr = np.array(x_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32 if continuous else np.int64)

        self.x = torch.from_numpy(x_arr)
        self.y = torch.from_numpy(y_arr)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
# for long horizon
def build_window_features(obs: dict[str, list[np.ndarray]], t: int, Z_t: set[str], slots: list[tuple[str, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    x_parts, m_parts = [], []

    Z_idx = {(v[0], int(v[1:])) for v in Z_t}

    for (var, rel, dim) in slots:
        step = t + rel

        if step < 0 or var not in obs or step >= len(obs[var]):
            x_parts.append(np.zeros(dim, dtype=np.float32))
            m_parts.append(np.zeros(1, dtype=np.float32))
            continue

        use_slot = (var, step) in Z_idx
        if use_slot:
            val = np.asarray(obs[var][step], dtype=np.float32)
            if val.ndim == 0:
                val = np.array([val], dtype=np.float32)

            # match expected dim via reshape
            if val.size != dim:
                if val.size > dim:
                    val = val.ravel()[:dim]

                else:
                    pad = np.zeros(dim - val.size, dtype=np.float32)
                    val = np.concatenate([val.ravel(), pad], axis=0)

            x_parts.append(val.astype(np.float32))
            m_parts.append(np.ones(1, dtype=np.float32))

        else:
            x_parts.append(np.zeros(dim, dtype=np.float32))
            m_parts.append(np.zeros(1, dtype=np.float32))

    x = np.concatenate(x_parts, axis=0).astype(np.float32)
    m = np.concatenate(m_parts, axis=0).astype(np.float32)
    return x, m

class WindowedExpertDataset(Dataset):
    def __init__(self, records, Z_sets_trimmed, slots, action_var, continuous=False):
        self.records = records
        self.Z_sets = Z_sets_trimmed
        self.slots = slots
        self.continuous = continuous

        x_list = []
        y_list = []

        for r in records:
            t = int(r.get('step', 0))
            key = f'X{t}'
            if key not in self.Z_sets:
                continue

            obs = r.get('obs', {})
            Z_t = self.Z_sets[key]

            x, m = build_window_features(obs, t, Z_t, self.slots)
            xm = np.concatenate([x, m], axis=0).astype(np.float32)

            act = r.get('action', None)
            if act is None:
                continue

            y = np.asarray(act, dtype=np.float32 if continuous else np.int64)

            x_list.append(xm)
            y_list.append(y)

        if len(x_list) == 0:
            raise ValueError('No valid datapoints in WindowedExpertDataset; check records or Z_sets_trimmed.')
        
        self.x = torch.from_numpy(np.stack(x_list, axis=0))
        self.y = torch.from_numpy(np.stack(y_list, axis=0))

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]