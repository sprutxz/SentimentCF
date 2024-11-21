print("job started")
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam
import gc

# Check for GPU availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess data
chunksize = 200000
chunks = []

# Iterating over chunks
for chunk in pd.read_csv('/common/home/lms548/dev/CS-439-FinalProject/dataset/dataset.csv', chunksize=chunksize):
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
print("dataset loaded")

# Define helper functions
def clean_data(df):
    df.drop(columns=['reviewerName', 'verified', 'reviewTime', 'summary', 'unixReviewTime', 'style', 'vote', 'image'], inplace=True)
    df.dropna(subset=['reviewText'], inplace=True)
    
def preprocess(data):
    label_encoder = LabelEncoder()
    data.loc[:, 'reviewerID'] = label_encoder.fit_transform(data['reviewerID'])
    data.loc[:, 'asin'] = label_encoder.fit_transform(data['asin'])
    
def train_test_split(df, test_size=0.2):
    indices = df.index.tolist()
    test_indices = np.random.choice(indices, size=int(len(df)*test_size), replace=False)
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df, test_df

def split_user_group(group):
    group = group.sample(frac=1).reset_index(drop=True)
    train = group.iloc[:3]
    test = group.iloc[3:5]
    return train, test

preprocess(df)
num_users = df['reviewerID'].nunique()
num_items = df['asin'].nunique()

train_df, test_df = train_test_split(df)

# Define dataset class
class AmazonRatingsDataset(Dataset):
    def __init__(self, df):
        self.user_ids = df['reviewerID'].values
        self.item_ids = df['asin'].values
        self.ratings = df['overall'].values

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            'user_id': torch.tensor(self.user_ids[idx], dtype=torch.long),
            'item_id': torch.tensor(self.item_ids[idx], dtype=torch.long),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.long)
        }

# Create datasets and dataloaders
train_dataset = AmazonRatingsDataset(train_df)
test_dataset = AmazonRatingsDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define model
class Model(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=128):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim*2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 5)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, user, item):
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        x = torch.cat([user_embedding, item_embedding], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# Initialize model, criterion and optimizer
# targets = train_df['overall'].values
# class_weights = compute_class_weight(class_weight='balanced',classes=np.unique(targets), y=targets)
# class_weights = torch.tensor(class_weights, dtype=torch.float32)

def calculate_class_weights(targets):
    total_samples = len(targets)
    class_counts = np.bincount(targets)[1:]
    weights = total_samples / class_counts
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights
    
targets = train_df['overall'].values
class_weights = calculate_class_weights(targets).to(device)

model = Model(num_users, num_items).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)

# Training loop
EPOCHS = 50
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0

    for batch in train_loader:
        user = batch['user_id'].to(device)
        item = batch['item_id'].to(device)
        rating = (batch['rating'] - 1).to(device)
        
        optimizer.zero_grad()
        output = model(user, item)
        loss = criterion(output, rating)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct_train += (predicted == rating).sum().item()
        total_train += rating.size(0)

    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(correct_train / total_train)

    model.eval()
    test_loss = 0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for batch in test_loader:
            # Move tensors to GPU
            user = batch['user_id'].to(device)
            item = batch['item_id'].to(device)
            rating = (batch['rating'] - 1).to(device)
            
            output = model(user, item)
            loss = criterion(output, rating)
            test_loss += loss.item()

            _, predicted = torch.max(output, 1)
            correct_test += (predicted == rating).sum().item()
            total_test += rating.size(0)

    test_losses.append(test_loss / len(test_loader))
    test_accuracies.append(correct_test / total_test)

    print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss / len(train_loader)}, Test Loss: {test_loss / len(test_loader)}, Train Accuracy: {correct_train / total_train}, Test Accuracy: {correct_test / total_test}')

# Plot results (remains unchanged)
plt.figure(figsize=(12, 8))
plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
plt.plot(range(1, EPOCHS + 1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Train and Test Loss over Epochs')
plt.show()
plt.figure(figsize=(12, 8))
plt.plot(range(1, EPOCHS + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, EPOCHS + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train and Test Accuracy over Epochs')
plt.show()

# Test predictions
model.eval()
with torch.no_grad():
    for batch in test_loader:
        # Move tensors to GPU
        user = batch['user_id'].to(device)
        item = batch['item_id'].to(device)
        rating = (batch['rating'] - 1).to(device)
        
        output = model(user, item)
        output = F.softmax(output, dim=-1)
        preds = torch.argmax(output, dim=-1)
        
        print('Predictions:', preds.cpu())  
        print('Actual:', rating.cpu())
        break

# Save model
torch.save(model.state_dict(), 'model_big.pth')