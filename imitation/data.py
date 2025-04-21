import torch
from torch.utils.data import Dataset

class ExpertDataset(Dataset):
    def __init__(self, records, cond_vars, action_var):
        x_list = []
        y_list = []

        for r in records:
            if any(r['obs'][v] is None for v in cond_vars):
                continue

            x_list.append([r['obs'][v] for v in cond_vars])
            y_list.append(r[action_var])

        self.x = torch.tensor(x_list, dtype=torch.float32)
        self.y = torch.tensor(y_list, dtype=torch.long)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]