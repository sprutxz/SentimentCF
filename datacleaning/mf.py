import pandas as pd
import numpy as np
import torch
import torch.nn as nn
# from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
# Add to imports at top
import matplotlib.pyplot as plt
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Amazon Electronics data
def load_amazon_data(path='/common/home/lms548/dev/CS-439-FinalProject/dataset/dataset.json', nrows=100000):
    df = pd.read_json(path, lines=True)
    df['user_idx'] = pd.factorize(df['reviewerID'])[0]
    df['item_idx'] = pd.factorize(df['asin'])[0]
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

# Custom Dataset
class RatingDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long).to(device)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long).to(device)
        self.ratings = torch.tensor(ratings, dtype=torch.float).to(device)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]

# Update MatrixFactorization class
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=100, reg_lambda=0.01):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embed_dim)
        self.item_embeddings = nn.Embedding(num_items, embed_dim)
        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)
        self.reg_lambda = reg_lambda
        
    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embeddings(user_ids)
        item_embeds = self.item_embeddings(item_ids)
        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()
        dot = (user_embeds * item_embeds).sum(dim=1)
        return dot + user_bias + item_bias
    
    def get_reg_loss(self):
        reg_loss = 0.0
        reg_loss += torch.norm(self.user_embeddings.weight, p=2)
        reg_loss += torch.norm(self.item_embeddings.weight, p=2)
        reg_loss += torch.norm(self.user_biases.weight, p=2)
        reg_loss += torch.norm(self.item_biases.weight, p=2)
        return self.reg_lambda * reg_loss

# Updated training function
# Modified training function with loss tracking and model saving
def train_model(model, train_loader, val_loader, epochs=40, save_dir='checkpoints'):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    # Create lists to store metrics
    train_losses = []
    val_losses = []
    train_rmses = []
    val_rmses = []
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_rmse = 0
        
        for user_ids, item_ids, ratings in train_loader:
            optimizer.zero_grad()
            predictions = model(user_ids, item_ids)
            mse_loss = criterion(predictions, ratings)
            reg_loss = model.get_reg_loss()
            loss = mse_loss + reg_loss
            
            loss.backward()
            optimizer.step()
            train_loss += mse_loss.item()
            train_rmse += torch.sqrt(mse_loss).item()
            
        # Validation
        model.eval()
        val_loss = 0
        val_rmse = 0
        
        with torch.no_grad():
            for user_ids, item_ids, ratings in val_loader:
                predictions = model(user_ids, item_ids)
                mse_loss = criterion(predictions, ratings)
                val_loss += mse_loss.item()
                val_rmse += torch.sqrt(mse_loss).item()
        
        # Calculate average metrics
        avg_train_loss = train_loss/len(train_loader)
        avg_train_rmse = train_rmse/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        avg_val_rmse = val_rmse/len(val_loader)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_rmses.append(avg_train_rmse)
        val_rmses.append(avg_val_rmse)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(save_dir, 'best_model.pt'))
        
        print(f'Epoch {epoch+1}:')
        print(f'Train - MSE: {avg_train_loss:.4f}, RMSE: {avg_train_rmse:.4f}')
        print(f'Val   - MSE: {avg_val_loss:.4f}, RMSE: {avg_val_rmse:.4f}')
        print('-' * 50)
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_rmses, label='Train RMSE')
    plt.plot(val_rmses, label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training and Validation RMSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()
    
    return train_losses, val_losses, train_rmses, val_rmses

# Main execution
if __name__ == "__main__":
    print(f"Using device: {device}")
    
    # Load data
    df = load_amazon_data()
    print("Data loaded")
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2)
    
    # Create datasets
    train_dataset = RatingDataset(train_df['user_idx'].values, 
                                train_df['item_idx'].values,
                                train_df['overall'].values)
    
    test_dataset = RatingDataset(test_df['user_idx'].values,
                               test_df['item_idx'].values,
                               test_df['overall'].values)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024)
    
    # Initialize model and move to GPU
    num_users = df['user_idx'].nunique()
    num_items = df['item_idx'].nunique()
    model = MatrixFactorization(num_users, num_items, reg_lambda=0.1).to(device)
    
    # Train model
    train_model(model, train_loader, test_loader)