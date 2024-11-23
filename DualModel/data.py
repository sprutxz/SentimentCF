"""
This module useful functions for data loading and manipulation.
"""

import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

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

