"""
this module provides top-k recommendations for a given user
"""

import pandas as pd
import numpy as np
import torch
from modelarc import NeuralCollaborativeFiltering
from dataset import InferenceDataset
from metrics import calculate_metrics
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 10

testfile = 'dataset/test.csv'
test_df = pd.read_csv(testfile)

n_users = test_df['reviewerID'].max() + 1
n_items = test_df['asin'].max() + 1

model1_pth = 'models/best_model_neumf.pt'

model1 = NeuralCollaborativeFiltering(n_users, n_items).to(DEVICE)
model1.load_state_dict(torch.load(model1_pth)['model_state_dict'])
model1.eval()

items = test_df['asin'].unique()
users = test_df['reviewerID'].value_counts()
users = users[users > 10].index
sampled_users = set(np.random.choice(users, 1, replace=False))

user_items = {}

for user, value in test_df.groupby('reviewerID'):
    if user in sampled_users:
        user_items[user] = list(zip(value['asin'].values, value['overall'].values))
metrics_list = []

for user in sampled_users:
    dataset = InferenceDataset(user, items)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
    recs = []
    with torch.no_grad():
        for batch in dataloader:
            userid = batch['user_id'].to(DEVICE)
            itemid = batch['item_id'].to(DEVICE)
            
            score = model1(userid, itemid)
            
            # pred = model2(userid, itemid, score)
            
            pred = torch.argmax(score, dim=-1) + 1
            
            recs.extend(list(zip(itemid.cpu().numpy(), pred.cpu().numpy())))
        
        sorted_recs = sorted(recs, key=lambda x: x[1], reverse=True)[:K]
    
    metrics = calculate_metrics(sorted_recs, user_items[user], K)
    metrics_list.append(metrics)

# Calculate the average metrics
metrics_array = np.array(metrics_list)
average_metrics = {
    'precision': np.mean(metrics_array[:, 0]),
    'recall': np.mean(metrics_array[:, 1]), 
    'f1': np.mean(metrics_array[:, 2]),
    'NDCG': np.mean(metrics_array[:, 3])
}
print("Average Metrics:", average_metrics)
        