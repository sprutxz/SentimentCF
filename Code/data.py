"""
This module useful functions for data loading and manipulation.
"""

import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, accuracy_score
import torch
import os

def load_data(path):
    df = pd.read_csv(path)
    return df

def train_test_split(df, test_size=0.2):
    train_indices = []
    test_indices = []
    
    # Group by 'reviewerID' and process each user's data
    for _, user_df in df.groupby('reviewerID'):
        indices = user_df.index.tolist()
        n_test = int(len(indices) * test_size)
        
        # Randomly select test indices for this user
        test_sample = np.random.choice(indices, size=n_test, replace=False)
        train_sample = list(set(indices) - set(test_sample))
        
        # Collect indices directly
        test_indices.extend(test_sample)
        train_indices.extend(train_sample)
    
    # Create train and test splits with indexing (avoids creating many intermediate DataFrames)
    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]
    
    return train_df, test_df

def compute_class_weights(df):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(df['overall']), y=df['overall'])
    return class_weights

def calculate_metrics(y_true, y_pred):
    """Calculate RMSE, MAE, and Precision"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    return rmse, mae, precision, accuracy

def plot_metrics(train_losses, test_losses, precisions, accuracies, save_path='plots'):
    """Plot and save training metrics"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_path}/losses.png')
    plt.close()
    
    # Plot precision and accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(precisions, label='Precision')
    plt.plot(accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()
    plt.savefig(f'{save_path}/metrics.png')
    plt.close()

def save_model(model, epoch, optimizer, loss, metrics, path='checkpoints'):
    """Save model checkpoint"""
    if not os.path.exists(path):
        os.makedirs(path)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    torch.save(checkpoint, f'{path}/model_epoch_{epoch}.pt')