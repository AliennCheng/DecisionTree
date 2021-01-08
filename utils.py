import numpy as np

def calc_entropy(X_arr, y, sp):
    '''Compute the entropy of array arr.
    
    Args:
        - X: the data of the candidate feature
        - y: the target feature
        - sp: the split point to separate arr into two groups
    
    Return:
        - H: the entropy of array arr
    '''
    if len(np.unique(X_arr)) == 1: # all the elements have the same value
        return 1e6
    if len(y[X_arr <= sp]) == 0 or len(y[X_arr > sp]) == 0: # no need to split
        return 1e6
    
    p1 = y[X_arr <= sp].sum() / len(y[X_arr <= sp]) + 1e-6
    p2 = y[X_arr > sp].sum() / len(y[X_arr > sp]) + 1e-6
    H = -(p1 * np.log1p(p1)) - ((1-p1) * np.log1p(1-p1)) \
        -(p2 * np.log1p(p2)) - ((1-p2) * np.log1p(1-p2))
    return H

def split_node(X, y, idx):
    '''Get the index of the feature to split the node
    
    Args:
        - X: the original training data X
        - y: the target feature
        - idx: the instance index under analysis
        
    Returns:
        - i_max: the feature index with the largest IG
        - split_pt[i_max]: the split point of the return feature
    '''
    
    # TODO: try every possible split point rather than median

    _, n_features = X.shape
    H_X = np.empty(n_features)
    split_pt = np.empty(n_features)
    
    for i in range(n_features):
        
        # decide where to split
        if len(np.unique(X[idx, i])) <= 2:
            sp = np.median(np.unique(X[idx, i]))
        else:
            sp = np.median(X[idx, i])
            
        H_X[i] = calc_entropy(X[idx, i], y[idx], sp)
        split_pt[i] = sp
        
    i_max = np.argmin(H_X)
    return i_max, split_pt[i_max]
