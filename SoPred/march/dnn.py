"""
This module contains the model architecture for the Amazon ratings dataset.
"""

import torch
import torch.nn as nn

class SoPredModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=128):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(2*embedding_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ReLU()
        
    def forward(self, user, item):
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        x = torch.cat([user_embedding, item_embedding], dim=-1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x.squeeze()