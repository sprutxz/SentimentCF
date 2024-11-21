print("job started")
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam

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

train_df, test_df = train_test_split(df)

# Define dataset class
class AmazonRatingsDataset(Dataset):
    def __init__(self, df):
        self.so_score = df['so_score'].values
        self.ratings = df['overall'].values

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return {
            'so_score': torch.tensor([self.so_score[idx]], dtype=torch.float32),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.long)
        }

# Create datasets and dataloaders
train_dataset = AmazonRatingsDataset(train_df)
test_dataset = AmazonRatingsDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define model
class Model(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, 5)
        self.dropout = nn.Dropout(0.3)
        self.lrelu = nn.LeakyReLU()
        
    def forward(self, score):
        x = self.fc1(score)
        x = self.lrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x

def calculate_class_weights(targets):
    total_samples = len(targets)
    class_counts = np.bincount(targets)[1:]
    weights = total_samples / class_counts
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights

    
targets = train_df['overall'].values
class_weights = calculate_class_weights(targets).to(device)

input_shape = 1
model = Model(input_shape).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = Adam(model.parameters(), lr=1e-2)

# Training loop
EPOCHS = 50
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

model.train()
for epoch in range(EPOCHS):
    train_loss = 0
    correct_train = 0
    total_train = 0

    for batch in train_loader:
        score = batch['so_score'].to(device)
        rating = (batch['rating'] - 1).to(device)
        
        optimizer.zero_grad()
        output = model(score)
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
            score = batch['so_score'].to(device)
            rating = (batch['rating'] - 1).to(device)
            
            output = model(score)
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
plt.savefig('so-to-rating-loss.png')


plt.figure(figsize=(12, 8))
plt.plot(range(1, EPOCHS + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, EPOCHS + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Train and Test Accuracy over Epochs')
plt.show()
plt.savefig('so-to-rating-accuracy.png')

# Save model
torch.save(model.state_dict(), 'score_pred.pth')