"""
this module provides top-k recommendations for a given user
"""

import pandas as pd
import torch
from SoPred.march.neumf import SoPredModel
from RtPred.march.neumf import RtPredModel

testfile = 'dataset/test.csv'
test_df = pd.read_csv(testfile)

n_users = test_df['reviewerID'].max() + 1
n_items = test_df['asin'].max() + 1

model1_pth = 'models/best_model1.pth'
model2_pth = 'models/best_model2.pth'

model1 = SoPredModel(n_users, n_items)
model1.load_state_dict(torch.load(model1_pth)['model_state_dict'])

model2 = RtPredModel(n_users, n_items)
model2.load_state_dict(torch.load(model2_pth)['model_state_dict'])