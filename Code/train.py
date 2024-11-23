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
from model import NeuralCollaborativeFiltering
from data import load_data, train_test_split, compute_class_weights, plot_metrics, calculate_metrics, save_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10

df = load_data('/common/home/lms548/dev/CS-439-FinalProject/dataset/dataset.csv')
n_users = df['reviewerID'].max()+1
n_items = df['asin'].max()+1

train_df, test_df = train_test_split(df)
train_df.to_csv('/common/home/lms548/dev/CS-439-FinalProject/dataset/train.csv', index=False)
test_df.to_csv('/common/home/lms548/dev/CS-439-FinalProject/dataset/test.csv', index=False)

train_dataset = AmazonRatingsDataset(train_df)
test_dataset = AmazonRatingsDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class_weights = compute_class_weights(train_df)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

model = NeuralCollaborativeFiltering(n_users, n_items, embedding_dim=128).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)

# Training tracking variables
best_loss = float('inf')
train_losses = []
test_losses = []
precisions = []
accuracies = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_preds = []
    train_true = []
    
    for batch in train_loader:
        user = batch['user_id'].to(DEVICE)
        item = batch['item_id'].to(DEVICE)
        score = batch['score'].unsqueeze(-1).to(DEVICE)
        rating = batch['rating'].to(DEVICE)
        
        optimizer.zero_grad()
        output = model(user, item, score)
        loss = criterion(output, rating)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        # Store predictions for metrics
        train_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
        train_true.extend(rating.cpu().numpy())
        
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
            score = batch['score'].to(DEVICE).unsqueeze(-1)
            rating = batch['rating'].to(DEVICE)
            
            output = model(user, item, score)
            loss = criterion(output, rating)
            test_loss += loss.item()
            
            # Store predictions for metrics
            test_preds.extend(torch.argmax(output, dim=1).cpu().numpy())
            test_true.extend(rating.cpu().numpy())
            
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    
    # Calculate metrics
    rmse, mae, precision, accuracy = calculate_metrics(test_true, test_preds)
    precisions.append(precision)
    accuracy.append(accuracy)
    
    # Save best model
    if test_loss < best_loss:
        best_loss = test_loss
        save_model(model, epoch, optimizer, test_loss, 
                  {'rmse': rmse, 'mae': mae, 'precision': precision, 'accuracy': accuracy})
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, Precision: {precision:.4f}")

# Plot final metrics
plot_metrics(train_losses, test_losses, precisions, accuracy)