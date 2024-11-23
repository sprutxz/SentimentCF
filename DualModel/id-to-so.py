"""
This module contains the training script for the Amazon ratings dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from dataset import AmazonRatingsDataset
from DualModel.model import NeuralCollaborativeFiltering
from data import load_data, train_test_split, compute_class_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10

df = load_data('/common/home/lms548/dev/CS-439-FinalProject/dataset/dataset.csv')
n_users = df['reviewerID'].max()+1
n_items = df['asin'].max()+1

train_df, test_df = train_test_split(df)

train_dataset = AmazonRatingsDataset(train_df)
test_dataset = AmazonRatingsDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = NeuralCollaborativeFiltering(n_users, n_items, embedding_dim=128).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        user = batch['user_id'].to(DEVICE)
        item = batch['item_id'].to(DEVICE)
        score = batch['score'].to(DEVICE)
        
        optimizer.zero_grad()
        output = model(user, item)
        loss = criterion(output, score)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            user = batch['user_id'].to(DEVICE)
            item = batch['item_id'].to(DEVICE)
            score = batch['score'].to(DEVICE)
            
            output = model(user, item)
            loss = criterion(output, score)
            test_loss += loss.item()
        test_loss /= len(test_loader)
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")