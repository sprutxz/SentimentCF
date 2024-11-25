"""
This module contains the custom dataset class for the Amazon ratings dataset.
"""

import torch
from torch.utils.data import Dataset

class InferenceDataset(Dataset):
    def __init__(self, user, items):
        self.user_id = user
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return {
            'user_id': torch.tensor(self.user_id, dtype=torch.long),
            'item_id': torch.tensor(self.items[idx], dtype=torch.long)
        }