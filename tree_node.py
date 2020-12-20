import numpy as np
from utils import calc_entropy, split_node

class TreeNode:
    '''Tree node structure of the decision tree
    '''

    def __init__(self, X, y, idx, depth, max_depth):
        assert len(X.shape) == 2, 'TreeNode input X doesn\'t match the format.'
        assert len(y.shape) == 1, 'TreeNode input y doesn\'t match the format.'
        assert X.shape[0] == y.shape[0], 'TreeNode input X and y doesn\t match.'
            
        self.X = X
        self.y = y
        self.idx = idx # the instances index in this node and its descendants
        self.depth = depth
        self.is_leaf = (len(idx) <= 1) or (depth >= max_depth) or (len(np.unique(y[idx]))==1)
        if not self.is_leaf:
            self.max_depth = max_depth
            self.left_child, self.right_child = self.grow(X, y)
        if self.is_leaf:
            val, cnt = np.unique(self.y[self.idx], return_counts=True)
            self.response = val[np.argmax(cnt)]
            self.probability = cnt / len(idx)
        else:
            self.response = None
        
    def grow(self, X, y):
        '''Grow the tree until there is an unique y response in the node or hit the depth threshold
        
        Args:
            - X: the original training data X excluding target feature
            - y: the original training target feature y
        
        Returns:
            - left_child: left child of this node
            - right_child: right child of this node
        '''
        self.split_feature, self.split_point = split_node(X, y, self.idx)
        left_idx = self.idx[np.where(X[self.idx, self.split_feature] <= self.split_point)[0]]
        right_idx = self.idx[np.where(X[self.idx, self.split_feature] > self.split_point)[0]]
        if len(left_idx) == 0 or len(right_idx) == 0:
            self.is_leaf = True
            return None, None
        left_child = TreeNode(X, y, left_idx, self.depth+1, self.max_depth)
        right_child = TreeNode(X, y, right_idx, self.depth+1, self.max_depth)
        return left_child, right_child
    
    def predict(self, X_ts, prob=False, verbose=False):
        '''Predict the target value of a given instance X_ts
        
        Returns:
            - response: the predicted target value of X_ts
        '''
        if self.is_leaf:
            if prob:
                if verbose: print('\nReached leaf node. The prediction is: ', self.probability)
                return self.probability
            else:
                if verbose: print('\nReached leaf node. The prediction is: ', self.response)
                return self.response
        elif X_ts[self.split_feature] <= self.split_point:
            if verbose:
                print('\nSplit feature: #', self.split_feature)
                print('x[', self.split_feature, '] = ', X_ts[self.split_feature])
                print('Split condition: <= ', self.split_point)
            return self.left_child.predict(X_ts, verbose)
        else:
            if verbose:
                print('\nSplit feature: #', self.split_feature)
                print('x[', self.split_feature, '] = ', X_ts[self.split_feature])
                print('Split condition: > ', self.split_point)
            return self.right_child.predict(X_ts, verbose)
    
    def show(self):
        '''Print the node information onto the screen.
        '''
        if self.is_leaf:
            print(' ' * self.depth, 'Leaf node: ', self.idx)
            print(' ' * self.depth, 'Leaf response: ', self.response)
        else:
            print(' ' * self.depth, 'Split condition: #', \
                  self.split_feature, ' <= ', self.split_point)
            self.left_child.show()
            self.right_child.show()
