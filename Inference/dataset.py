"""
This module contains the custom dataset class for the inference
"""

import torch
from torch.utils.data import Dataset

class InferenceDataset(Dataset):
    """
    A singular user and a list of all items is passed
    """
    def __init__(self, user, items):
        self.user_id = user
        self.items = items

    def __len__(self):
        return len(self.items)

    # Returns the user_id and item_id given an index
    def __getitem__(self, idx):
        return {
            'user_id': torch.tensor(self.user_id, dtype=torch.long),
            'item_id': torch.tensor(self.items[idx], dtype=torch.long)
        }