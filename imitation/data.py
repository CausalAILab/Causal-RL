import torch
from torch.utils.data import Dataset

class ExpertDataset(Dataset):
    def __init__(self, records, cond_vars, action_var):
        x_list = []
        y_list = []

        for r in records:
            step = r.get('step')
            obs = r.get('obs', {})

            xs = []

            for v in cond_vars:
                var = v[0]
                step = int(v[1:])
                val = obs[var][step]
                xs.append(val)

            if action_var not in r:
                continue

            y = r[action_var]

            x_list.append(xs)
            y_list.append(y)

        if len(x_list) == 0:
            raise ValueError(f'No valid data points in ExpertDataset with cond_vars={cond_vars}')

        self.x = torch.tensor(x_list, dtype=torch.float32)
        self.y = torch.tensor(y_list, dtype=torch.long)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]