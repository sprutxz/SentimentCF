import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
import gc
chunksize = 200000
chunks = []

# Iterating over chunks
for chunk in pd.read_json('../dataset/dataset.json', lines=True, chunksize=chunksize):
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)

df.head()
# defining functions here
def clean_data(df):
    df.drop(columns=['reviewerName', 'reviewText', 'verified', 'reviewTime', 'summary', 'unixReviewTime', 'style', 'vote', 'image'], inplace=True)
    #df.dropna(subset=['reviewText'], inplace=True)
    #df.fillna({'vote': 1}, inplace=True)
    
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
clean_data(df)
df.isna().sum()
user_review_counts = df.groupby('reviewerID').size()
df_filtered = df[df['reviewerID'].isin(user_review_counts[user_review_counts >= 5].index)]

unique_users = df_filtered['reviewerID'].unique()
sampled_users = np.random.choice(unique_users, size=1000, replace=False)
df_sampled = df_filtered[df_filtered['reviewerID'].isin(sampled_users)]

preprocess(df_sampled)
num_users = df_sampled['reviewerID'].nunique()
num_items = df_sampled['asin'].nunique()

train_list = []
test_list = []

for user_id, group in df_sampled.groupby('reviewerID'):
    group = group.sample(frac=1).reset_index(drop=True)

    train = group.iloc[:3]
    test = group.iloc[3:5]
    
    train_list.append(train)
    test_list.append(test)
    
train_df = pd.concat(train_list)
test_df = pd.concat(test_list)

print(f'Train set size: {len(train_df)}')
print(f'Test set size: {len(test_df)}')

del df
del df_sampled
gc.collect()
train_df.head()
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
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float32)
        }
train_dataset = AmazonRatingsDataset(train_df)
test_dataset = AmazonRatingsDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
class Model(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim*2, 128)
        self.fc2 = nn.Linear(128, 5)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, user, item):
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        x = torch.cat([user_embedding, item_embedding], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
targets = train_df['overall'].values
class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

model = Model(num_users, num_items)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 500

for epoch in range(EPOCHS):
    for batch in train_loader:
        user = batch['user_id']
        item = batch['item_id']
        rating = batch['rating']
        
        optimizer.zero_grad()
        output = model(user, item)
        print(output.shape)
        loss = criterion(output, rating)
        loss.backward()
        optimizer.step()
        
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item()}')
model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch in test_loader:
        user = batch['user_id']
        item = batch['item_id']
        rating = batch['rating']
        
        output = model(user, item)
        loss = criterion(output, rating)
        test_loss += loss.item()
        
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss}')
    
    
test_df['overall'].value_counts()
# testing a sample
sample = test_df.iloc[1999]
user = torch.tensor(sample['reviewerID'], dtype=torch.long)
item = torch.tensor(sample['asin'], dtype=torch.long)
rating = torch.tensor(sample['overall'], dtype=torch.float32)

output = model(user, item)
print(f'Predicted Rating: {output.item()}, True Rating: {rating.item()}')
