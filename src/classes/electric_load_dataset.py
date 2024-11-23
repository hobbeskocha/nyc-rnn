import torch
from torch.utils.data import Dataset

class ElectricLoadDataset(Dataset):
    def __init__(self, df, seq_len = 24):
        self.seq_len = seq_len
        self.data = df.values

    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len, 0]
        return torch.tensor(x, dtype = torch.float32), torch.tensor(y, dtype = torch.float32)