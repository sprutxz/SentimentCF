"""
This module contains the model architecture for the Amazon ratings dataset.
"""

import torch
import torch.nn as nn

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, nn_emb_dim=32, mf_emb_dim=8):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.mf_usr_emb = nn.Embedding(num_users, mf_emb_dim)
        self.mf_item_emb = nn.Embedding(num_items, mf_emb_dim)
        
        self.nn_usr_emb = nn.Embedding(num_users, nn_emb_dim)
        self.nn_item_emb = nn.Embedding(num_items, nn_emb_dim)
        
        self.fc1 = nn.Linear(2*nn_emb_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        
        self.neumf = nn.Linear(16, 5)
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.ReLU()
        
    def forward(self, user, item):
        mf_user_emb = self.mf_usr_emb(user)
        mf_item_emb = self.mf_item_emb(item)
        mf_x = mf_user_emb * mf_item_emb
        
        nn_user_emb = self.nn_usr_emb(user)
        nn_item_emb = self.nn_item_emb(item)

        x = torch.cat([nn_user_emb, nn_item_emb], dim=-1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        neumf_input = torch.cat([mf_x, x], dim=-1)
        return self.neumf(neumf_input).squeeze()
        