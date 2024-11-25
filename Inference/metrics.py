""" 
Thi module calculates prec@k, recall@k, F1 score and NDCG@k
"""

import math

def calculate_metrics(recs, true, k):
    recommendations  = set([rec[0] for rec in recs])
    true_recs = set([t[0] for t in true])
    
    # calculate precision@k
    precision = len(recommendations & true_recs) / k
    
    # calculate recall@k
    recall = len(recommendations & true_recs) / len(true_recs)
    
    # calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # calculate NDCG@k
    dcg = 0
    idcg = 0
    for i in range(k):
        if recommendations[i] in true_recs:
            dcg += 1 / math.log2(i + 2)
        idcg += 1 / math.log2(i + 2)
    ndcg = dcg / idcg
    
    return precision, recall, f1, ndcg