"""
This module contains the custom dataset class for the Amazon ratings dataset.
"""

import torch
from torch.utils.data import Dataset

class AmazonRatingsDataset(Dataset):
    def __init__(self, df):
        self.user_ids = df['reviewerID'].values
        self.item_ids = df['asin'].values
        self.scores = df['so_score'].values
        self.ratings = df['overall'].values - 1

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            'user_id': torch.tensor(self.user_ids[idx], dtype=torch.long),
            'item_id': torch.tensor(self.item_ids[idx], dtype=torch.long),
            'score': torch.tensor(self.scores[idx], dtype=torch.float32),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.long)
        }