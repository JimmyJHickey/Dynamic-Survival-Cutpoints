import torch
import numpy as np

def discrete_ci(pred, cutpoints, t_test_in, s_test_in, t_train_in):
    s_test = torch.from_numpy(s_test_in).float()
    t_test = torch.from_numpy(t_test_in).float()
    t_train = torch.from_numpy(t_train_in).float()
    
    # cdfs
    t_pred_cdf = np.cumsum(pred, axis=1) 

    bucket_boundaries = [0] + sorted([i.item() for i in cutpoints]) + [1]
    # rescale
    bucket_boundaries = [boundary_i * (max(t_train) - min(t_train)) + min(t_train) 
                             for boundary_i in bucket_boundaries]
    
    N = len(pred)
    n_buckets = len(bucket_boundaries) - 1

    # one hot vector for where the 1 is in the most likely bucket
    t_true_idx = np.zeros((N, n_buckets),dtype=int)
    for ii in range(N):
        for jj in range(n_buckets):
            if t_test[ii] < bucket_boundaries[jj+1]:
                t_true_idx[ii][jj] = 1
                break
    
    t_true_idx = np.argmax(t_true_idx, axis=1)
    concordant = 0
    total = 0


    idx = np.arange(N)

    
    
    for i in range(N):

        if s_test[i] == 0:
            continue

        # time bucket of observation for i, then for all but i
        tti_idx = t_true_idx[i]
        
        tt_idx = t_true_idx[idx != i]

        # calculate predicted risk for i at the time of their event
        tpi = t_pred_cdf[i, tti_idx]

        # predicted risk at that time for all but i
        tp = t_pred_cdf[idx != i, tti_idx]
        
        total += np.sum(tti_idx < tt_idx) # observed in i first

        concordant += np.sum((tti_idx < tt_idx) * (tpi > tp)) # and i predicted as higher risk

    return concordant / total
