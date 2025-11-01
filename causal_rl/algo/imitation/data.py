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