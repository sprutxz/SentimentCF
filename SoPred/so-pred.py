"""
This module contains the training script for the Amazon ratings dataset.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from dataset import AmazonRatingsDataset
from march.neumf import NeuralCollaborativeFiltering
from data import plot_metrics, calculate_metrics, save_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20

# df = load_data('/common/home/lms548/dev/CS-439-FinalProject/dataset/dataset.csv')
# n_users = df['reviewerID'].max()+1
# n_items = df['asin'].max()+1

# train_df, test_df = train_test_split(df)

train_df = pd.read_csv('/common/home/lms548/dev/CS-439-FinalProject/dataset/train.csv')
test_df = pd.read_csv('/common/home/lms548/dev/CS-439-FinalProject/dataset/test.csv')

n_users = train_df['reviewerID'].max()+1
n_items = train_df['asin'].max()+1

train_dataset = AmazonRatingsDataset(train_df)
test_dataset = AmazonRatingsDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

model = NeuralCollaborativeFiltering(n_users, n_items).to(DEVICE)
criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=0.1, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0, 
                              min_lr=1e-6)

best_loss = float('inf')
train_losses = []
test_losses = []
rmse_over_epochs = []
mae_over_epochs = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_preds = []
    train_true = []
    
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
        
        train_preds.extend(output.cpu().detach().numpy())
        train_true.extend(score.cpu().detach().numpy())
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_true = []
    
    with torch.no_grad():
        for batch in test_loader:
            user = batch['user_id'].to(DEVICE)
            item = batch['item_id'].to(DEVICE)
            score = batch['score'].to(DEVICE)
            
            output = model(user, item)
            loss = criterion(output, score)
            test_loss += loss.item()

            test_preds.extend(output.cpu().detach().numpy())
            test_true.extend(score.cpu().detach().numpy())
        
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    
    scheduler.step(test_loss)
    
    rmse, mae = calculate_metrics(test_true, test_preds)
    rmse_over_epochs.append(rmse)
    mae_over_epochs.append(mae)
    
    if test_loss < best_loss:
        best_loss = test_loss
        save_model(model, epoch, optimizer, test_loss, 
                  {'rmse': rmse, 'mae': mae})
        
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current Learning Rate: {current_lr:.6f}")

    
    plot_metrics(train_losses, test_losses, rmse_over_epochs, mae_over_epochs)