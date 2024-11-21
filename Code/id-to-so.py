print("job started")
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.optim import Adam

# Check for GPU availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess data
chunksize = 200000
chunks = []

for chunk in pd.read_csv('/common/home/lms548/dev/CS-439-FinalProject/dataset/dataset.csv', chunksize=chunksize):
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
print("dataset loaded")

def train_test_split(df, test_size=0.2):
    train_dfs = []
    test_dfs = []
    
    # Group by user
    for _, user_df in df.groupby('reviewerID'):
        # Get indices for this user's ratings
        indices = user_df.index.tolist()
        
        # Randomly select test indices for this user
        n_test = int(len(indices) * test_size)
        test_indices = np.random.choice(indices, size=n_test, replace=False)
        
        # Split user's ratings into train/test
        test_dfs.append(df.loc[test_indices])
        train_dfs.append(df.drop(test_indices))
    
    # Combine all users' train/test data
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    return train_df, test_df

num_users = df['reviewerID'].nunique()
num_items = df['asin'].nunique()

train_df, test_df = train_test_split(df)

class AmazonSODataset(Dataset):
    def __init__(self, df):
        self.user_ids = df['reviewerID'].values
        self.item_ids = df['asin'].values
        self.so_scores = df['so_score'].values

    def __len__(self):
        return len(self.so_scores)

    def __getitem__(self, idx):
        return {
            'user_id': torch.tensor(self.user_ids[idx], dtype=torch.long),
            'item_id': torch.tensor(self.item_ids[idx], dtype=torch.long),
            'so_score': torch.tensor([self.so_scores[idx]], dtype=torch.float32)
        }

train_dataset = AmazonSODataset(train_df)
test_dataset = AmazonSODataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class Model(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=128):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim*2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)  # Output single value for SO score
        self.dropout = nn.Dropout(0.3)
        self.lrelu = nn.LeakyReLU()
        
    def forward(self, user, item):
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        x = torch.cat([user_embedding, item_embedding], dim=-1)
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        return x

model = Model(num_users, num_items).to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

EPOCHS = 50
train_losses = []
test_losses = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for batch in train_loader:
        user = batch['user_id'].to(device)
        item = batch['item_id'].to(device)
        so_score = batch['so_score'].to(device)
        
        optimizer.zero_grad()
        output = model(user, item)
        loss = criterion(output, so_score)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            user = batch['user_id'].to(device)
            item = batch['item_id'].to(device)
            so_score = batch['so_score'].to(device)
            
            output = model(user, item)
            loss = criterion(output, so_score)
            test_loss += loss.item()

    test_losses.append(test_loss / len(test_loader))
    print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss / len(train_loader):.4f}, Test Loss: {test_loss / len(test_loader):.4f}')

plt.figure(figsize=(12, 8))
plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
plt.plot(range(1, EPOCHS + 1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Train and Test Loss over Epochs')
plt.savefig('id-to-so-loss.png')
plt.show()

# Test predictions
model.eval()
with torch.no_grad():
    for batch in test_loader:
        user = batch['user_id'].to(device)
        item = batch['item_id'].to(device)
        so_score = batch['so_score'].to(device)
        
        output = model(user, item)
        print('Predictions:', output.cpu().numpy().flatten()[:5])
        print('Actual:', so_score.cpu().numpy().flatten()[:5])
        break

torch.save(model.state_dict(), 'id_to_so.pth')